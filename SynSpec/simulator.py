import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import pickle
import os
import traceback
import time
import gc


# --- CONFIGURATION & DEVICE SETUP ---
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available. Using GPU.')
else:
    device = torch.device('cpu')
    print('CUDA not available. Using CPU.')
# EU Regulatory Parameters (ETSI EN 300 220)
EU_BAND_PARAMS = {
    'L': {'freq': (865.0e6, 868.0e6), 'max_bw': 250e3, 'dc_limit': 0.01, 'erp': 14},
    'M': {'freq': (868.0e6, 868.6e6), 'max_bw': 250e3, 'dc_limit': 0.01, 'erp': 14},
    'P': {'freq': (869.4e6, 869.65e6), 'max_bw': 500e3, 'dc_limit': 0.1, 'erp': 27}
}

# --- PHYSICS UTILITIES ---
# USE HANN WINDOW RAMP (FOR POWER AMPLIFIER) TO PREVENT SPECTRAL SPLATTER
def apply_pa_ramp(frame, fs, ramp_time=10e-6):
    ramp_samples = int(ramp_time * fs)
    if ramp_samples * 2 >= len(frame):
        window = torch.hann_window(len(frame), device=frame.device, dtype=torch.float32)
        return frame * window
    ramp = torch.hann_window(ramp_samples * 2, device=frame.device, dtype=torch.float32)
    frame[:ramp_samples] *= ramp[:ramp_samples]
    frame[-ramp_samples:] *= ramp[ramp_samples:]
    return frame

# SINGLE TAP RAYLEIGH CHANNEL TO SIMULATE UNIFORM AMPLITUDE ATTENTUATION AND PHASE SHIFT ACROSS BAND
def apply_flat_fading(signal):
    if len(signal) < 1:
        return signal
    # Generate Rayleigh fading coefficient: h = (h_real + j*h_imag) / sqrt(2)
    h_real = torch.randn(1, device=device, dtype=torch.float32)
    h_imag = torch.randn(1, device=device, dtype=torch.float32)
    h = torch.complex(h_real, h_imag) / np.sqrt(2)
    return signal * h

# --- REGULATORY LOGIC ---
# Determines non-compliant behavior (BW, OOB, ERP)
def get_rogue_parameters(band_info, signal_type, p_rogue=0.10):
    is_rogue = np.random.rand() < p_rogue
    violations = {
        'bw_violation': False,
        'oob_violation': False,
        'erp_violation': False,
        'forced_bw': None,
        'power_scale': 1.0,
        'freq_offset': 0.0
    }
    if not is_rogue:
        return violations
    if 'max_bw' in band_info and signal_type in ['lora', 'mixture']:
        if np.random.rand() < 0.5:
            violations['forced_bw'] = 500e3
            if violations['forced_bw'] > band_info['max_bw']:
                violations['bw_violation'] = True
    if np.random.rand() < 0.4:
        violations['oob_violation'] = True
        direction = np.random.choice([-1, 1])
        violations['freq_offset'] = direction * 0.4
    if np.random.rand() < 0.4:
        violations['erp_violation'] = True
        violations['power_scale'] = np.random.uniform(1.4, 2.0)
    return violations

# Ground truth generator for regulatory violations. Checks (BW, Symbol Rate, Duty Cycle, OOB, ERP)
def check_regulatory_violations_strict(metadata, band_info, rogue_params):
    violations = {}
    max_signal_bw = 0
    if metadata.get('lora_params'):
        max_signal_bw = max(max_signal_bw, metadata['lora_params']['BW'])
    if metadata.get('ieee_params'):
        sym_rate = metadata['ieee_params']['symbol_rate']
        # Symbol rate guard band check (80% heuristic for FSK)
        if sym_rate > band_info['max_bw'] * 0.8:
            violations['bandwidth'] = f"Symbol Rate {sym_rate/1e3:.1f}kHz > Guard Limit {band_info['max_bw']*0.8/1e3:.1f}kHz"
        max_signal_bw = max(max_signal_bw, sym_rate)
    if max_signal_bw > band_info['max_bw']:
        violations['bandwidth_limit'] = f"BW {max_signal_bw/1e3:.1f}kHz > Limit {band_info['max_bw']/1e3:.1f}kHz"
    cf = metadata['center_frequency']
    signal_bw = max_signal_bw if max_signal_bw > 0 else 125e3  # Default fallback
    if (cf - signal_bw/2 < band_info['freq'][0]) or (cf + signal_bw/2 > band_info['freq'][1]):
        violations['oob'] = "Signal Spills Out-of-Band"
    if metadata.get('dc_violation', False):
        violations['duty_cycle'] = "DC Limit Exceeded"
    if rogue_params.get('erp_violation', False):
        violations['erp'] = "ERP Limit Exceeded"
    return violations


# Determine signal type with persistence and realistic probabilities
def select_lpwan_signal_type(tech_history):
    LOR_P, IEE_P, MIX_P, NOI_P = 0.85, 0.05, 0.08, 0.02
    if not tech_history:
        return np.random.choice(['lora', 'ieee', 'mixture', 'noise'], p=[LOR_P, IEE_P, MIX_P, NOI_P])
    # 75% persistence of previous technology
    if np.random.rand() < 0.75:
        return tech_history[-1]
    return np.random.choice(['lora', 'ieee', 'mixture', 'noise'], p=[LOR_P, IEE_P, MIX_P, NOI_P])

# --- WAVEFORM GENERATORS ---
# LoRa signal generator with continuous phase tracking ------
# Implements proper frequency wrapping and phase accumulation -----
def generate_lora_signal(fs, frame_len, band_info, cf, prev_tech, last_phase, dc_violation, forced_bw=None):
    if forced_bw is not None:
        BW = forced_bw
    elif dc_violation:
        BW = 125e3
    elif band_info['max_bw'] >= 500e3:
        BW = np.random.choice([125e3, 250e3, 500e3])
    else:
        BW = np.random.choice([125e3, 250e3])
    SF = np.random.randint(7, 13)
    if prev_tech == 'lora' and np.random.rand() < 0.8:
        SF = min(12, max(7, SF + np.random.choice([-1, 0, 1])))
    symbol_duration = (2**SF) / BW
    samples_per_symbol = int(symbol_duration * fs)
    num_symbols = int(np.ceil(frame_len / samples_per_symbol))
    frame = torch.zeros(frame_len, dtype=torch.complex64, device=device)
    current_phase = last_phase if last_phase is not None else 0.0
    freq_drift_rate = np.random.uniform(-20, 20)  # Hz/s
    chirp_slope = BW / symbol_duration
    for i in range(num_symbols):
        start = i * samples_per_symbol
        end = min(start + samples_per_symbol, frame_len)
        active = end - start
        if active <= 0:
            break
        k = np.random.randint(0, 2**SF)
        freq_offset = k * BW / (2**SF)
        t_sym = torch.arange(active, device=device, dtype=torch.float32) / fs
        f_base = -BW/2 + chirp_slope * t_sym + freq_offset
        f_wrapped = ((f_base + BW/2) % BW) - BW/2
        phase_inc = 2 * np.pi * f_wrapped / fs
        sym_phase = torch.cumsum(phase_inc, dim=0)
        drift = 2 * np.pi * freq_drift_rate * (t_sym**2) / 2
        total_phase = current_phase + sym_phase + drift
        frame[start:end] = torch.exp(1j * total_phase)
        current_phase = total_phase[-1].item() % (2 * np.pi)
    metadata = {
        'SF': SF,
        'BW': BW,
        'symbol_duration': symbol_duration,
        'freq_drift_rate': freq_drift_rate,
        'last_phase': current_phase
    }
    
    return frame, metadata

# --- IEEE 802.15.4g GFSK signal generator with proper Gaussian filtering. ---
# --- Symbol rate constrained to bandwidth limits. ---
def generate_ieee_signal(fs, frame_len, band_info, cf, prev_tech, last_phase, dc_violation):
    allowed_bw = band_info['max_bw']
    if allowed_bw >= 500e3:
        possible_rates = [50e3, 100e3, 150e3, 200e3]
    elif allowed_bw >= 250e3:
        possible_rates = [50e3, 100e3]
    else:
        possible_rates = [50e3]
    if dc_violation:
        possible_rates = [50e3]
    symbol_rate = np.random.choice(possible_rates)
    mod_index = np.random.choice([0.5, 1.0])
    BT = 0.5  # Bandwidth-time product
    samples_per_symbol = max(1, int(fs / symbol_rate))
    freq_dev = (mod_index * symbol_rate) / 2
    num_symbols = max(1, frame_len // samples_per_symbol)    
    symbols = torch.randint(0, 2, (num_symbols,), device=device, dtype=torch.float32) * 2 - 1
    # Create Gaussian pulse shaping filter
    t_pulse = torch.arange(-3 * samples_per_symbol, 3 * samples_per_symbol + 1,
                           device=device, dtype=torch.float32) / fs
    gaussian_std = np.sqrt(np.log(2)) / (2 * np.pi * BT * symbol_rate)
    gauss_pulse = torch.exp(-0.5 * (t_pulse / gaussian_std)**2)
    gauss_pulse = gauss_pulse / gauss_pulse.sum()
    symbols_upsampled = torch.repeat_interleave(symbols, samples_per_symbol)
    if len(symbols_upsampled) < frame_len:
        pad = torch.zeros(frame_len - len(symbols_upsampled), device=device, dtype=torch.float32)
        symbols_upsampled = torch.cat((symbols_upsampled, pad))
    else:
        symbols_upsampled = symbols_upsampled[:frame_len]
    symbols_3d = symbols_upsampled.unsqueeze(0).unsqueeze(0)
    kernel_3d = gauss_pulse.unsqueeze(0).unsqueeze(0)
    filtered_3d = F.conv1d(symbols_3d, kernel_3d, padding=len(gauss_pulse)//2)
    filtered = filtered_3d.squeeze()[:frame_len]    
    freq_waveform = freq_dev * filtered    
    phase = 2 * np.pi * torch.cumsum(freq_waveform, dim=0) / fs
    t = torch.arange(frame_len, device=device, dtype=torch.float32) / fs
    freq_drift_rate = np.random.uniform(-5, 5)
    phase += 2 * np.pi * freq_drift_rate * t**2 / 2
    if last_phase is not None:
        phase += last_phase
    frame = torch.exp(1j * phase)
    current_phase = phase[-1].item() % (2 * np.pi)
    metadata = {
        'symbol_rate': symbol_rate,
        'modulation': 'GFSK',
        'mod_index': mod_index,
        'BT_product': BT,
        'freq_dev': freq_dev,
        'freq_drift_rate': freq_drift_rate,
        'last_phase': current_phase
    }
    return frame, metadata


# Creates realistic coexistence scenarios between LoRa and IEEE signals
# Models power ratio, frequency separation, and timing offset.
def apply_coexistence_constraints(lora, ieee, lora_meta, ieee_meta, fs):
    min_len = min(len(lora), len(ieee))
    if min_len == 0:
        return ieee, {}
    lora, ieee = lora[:min_len], ieee[:min_len]    
    power_ratio_db = np.random.uniform(5, 15)
    scale_factor = torch.sqrt(torch.tensor(10**(power_ratio_db/10), device=device, dtype=torch.float32))
    if np.random.rand() < 0.7:
        ieee = ieee / scale_factor
        power_ratio = power_ratio_db
    else:
        lora = lora / scale_factor
        power_ratio = -power_ratio_db
    if np.random.rand() < 0.3:
        freq_sep = np.random.uniform(-50e3, 50e3)
        interf_type = 'co-channel'
    else:
        freq_sep = np.random.choice([-1, 1]) * np.random.uniform(100e3, 300e3)
        interf_type = 'adjacent-channel'
    t = torch.arange(min_len, device=device, dtype=torch.float32) / fs
    ieee_shifted = ieee * torch.exp(1j * 2 * np.pi * freq_sep * t)
    max_offset = int(0.1 * fs)
    offset = np.random.randint(-max_offset, max_offset) if max_offset > 0 else 0
    ieee_shifted = torch.roll(ieee_shifted, offset)
    mixed = lora + ieee_shifted
    coex_metadata = {
        'power_ratio_db': power_ratio,
        'freq_separation': freq_sep,
        'timing_offset': offset,
        'interference_type': interf_type
    }
    return mixed, coex_metadata

# --- FRAME GENERATION  ---
# Main controller for generating LPWAN signal frames. Handles LoRa, IEEE, mixture, and noise types with regulatory compliance.
def generate_lpwan_spectrum_frame(fs, signal_type, band_id, band_info, dc_violation,
                                   prev_tech, symbol_memory, frame_len):
    if signal_type == 'noise':
        meta = {
            'signal_type': 'noise',
            'regulatory_violations': {},
            'center_frequency': np.mean(band_info['freq']),
            'is_rogue': False,
            'rogue_details': {},
            'lora_params': None,
            'ieee_params': None,
            'coex_params': None
        }
        return None, meta, symbol_memory

    rogue_params = get_rogue_parameters(band_info, signal_type, p_rogue=0.50)    
    f_low, f_high = band_info['freq']
    safe_margin = 0.1e6
    center_freq = np.random.uniform(f_low + safe_margin, f_high - safe_margin)
    if rogue_params['oob_violation']:
       
        bw_est = rogue_params.get('forced_bw') or 125e3
        
        if rogue_params['freq_offset'] > 0:
            center_freq = f_high - (bw_est * 0.1)
        else:
            center_freq = f_low + (bw_est * 0.1)
    metadata = {
        'signal_type': signal_type,
        'band_id': band_id,
        'center_frequency': center_freq,
        'dc_violation': dc_violation,
        'is_rogue': any(rogue_params[k] for k in ['bw_violation', 'oob_violation', 'erp_violation']),
        'rogue_details': rogue_params,
        'regulatory_violations': {},
        'lora_params': None,
        'ieee_params': None,
        'coex_params': None
    }
    if rogue_params['erp_violation']:
        metadata['erp_violation'] = True
    frame = torch.zeros(frame_len, dtype=torch.complex64, device=device)
    forced_bw = rogue_params.get('forced_bw', None)
    lora_signal_ref = None
    if signal_type in ['lora', 'mixture']:
        l_frame, l_meta = generate_lora_signal(
            fs, frame_len, band_info, center_freq, prev_tech,
            symbol_memory.get('lora'), dc_violation, forced_bw
        )
        if signal_type == 'lora':
            frame += l_frame
        metadata['lora_params'] = l_meta
        symbol_memory['lora'] = l_meta['last_phase']
        lora_signal_ref = l_frame
    if signal_type in ['ieee', 'mixture']:
        i_frame, i_meta = generate_ieee_signal(
            fs, frame_len, band_info, center_freq, prev_tech,
            symbol_memory.get('ieee'), dc_violation
        )
        if signal_type == 'ieee':
            frame += i_frame
        metadata['ieee_params'] = i_meta
        symbol_memory['ieee'] = i_meta['last_phase']
        if signal_type == 'mixture' and lora_signal_ref is not None:
            mixed, mix_meta = apply_coexistence_constraints(
                lora_signal_ref, i_frame, l_meta, i_meta, fs
            )
            frame = mixed
            metadata['coex_params'] = mix_meta
    metadata['regulatory_violations'] = check_regulatory_violations_strict(
        metadata, band_info, rogue_params
    )
    
    return frame, metadata, symbol_memory

# --- DATASET GENERATOR ---
# Generates a complete LPWAN spectrum sequence with overlapping snapshots. 
# This Implements proper SNR control via caached buffer technique.
def generate_lpwan_spectrum_dataset_with_snapshots(
    fs=1e6, seq_len=1.0, min_frames=8, max_frames=20, region='EU',
    snr_db=10, snapshot_len_sec=0.1, overlap_sec=0.05, gap_scale=0.02, jitter_max=0.02
):
    seq_samples = int(seq_len * fs)    
    sequence = torch.zeros(seq_samples, dtype=torch.complex64, device=device)
    # initilize cached buffer
    clean_sequence = torch.zeros(seq_samples, dtype=torch.complex64, device=device)
    
    full_metadata = {
        'region': region,
        'fs': fs,
        'snr_db': snr_db,
        'sub_band': None,
        'frames': [],
        'temporal_features': [],
        'dc_violations': 0,
        'erp_violations': 0,
        'total_tx_time': 0.0,
        'noise_power': None,
        'signal_power': None
    }

    band_probs = {'M': 0.7, 'P': 0.25, 'L': 0.05}
    band_id = np.random.choice(list(band_probs.keys()), p=list(band_probs.values()))
    full_metadata['sub_band'] = band_id
    band_info = EU_BAND_PARAMS[band_id]
    limit = band_info['dc_limit']
    r = np.random.rand()
    if r < 0.7:
        dc_accumulator = np.random.uniform(0, 0.5 * limit)
    elif r < 0.9:
        dc_accumulator = np.random.uniform(0.5 * limit, 0.9 * limit)
    else:
        dc_accumulator = np.random.uniform(0.9 * limit, 1.1 * limit)

    current_sample = 0
    tech_history = deque(maxlen=5)
    symbol_memory = {'lora': None, 'ieee': None}
    frame_count = 0
    prev_tech = None

    while current_sample < seq_samples and frame_count < max_frames:
        gap = int(np.random.exponential(gap_scale * fs))
        current_sample += gap
        if current_sample >= seq_samples:
            break
        
        max_len = min(int(0.5 * fs), seq_samples - current_sample)
        min_len = int(0.01 * fs)
        if max_len < min_len:
            break
        
        frame_len = np.random.randint(min_len, max_len + 1)
        frame_duration = frame_len / fs
        dc_accumulator += frame_duration / 3600
        dc_violation = dc_accumulator > limit
        if dc_violation:
            full_metadata['dc_violations'] += 1
        violate_dc_intent = np.random.rand() < 0.10
        if (frame_count >= min_frames) and dc_violation and not violate_dc_intent:
            continue

        signal_type = select_lpwan_signal_type(tech_history)        
        if signal_type == 'noise':
            dc_accumulator -= frame_duration / 3600
            dc_accumulator = max(0.0, dc_accumulator)  # Prevent negative
            if dc_violation:
                full_metadata['dc_violations'] -= 1
            dc_violation = dc_accumulator > limit
        tech_history.append(signal_type)
        frame, frame_meta, symbol_memory = generate_lpwan_spectrum_frame(
            fs, signal_type, band_id, band_info, dc_violation,
            prev_tech, symbol_memory, frame_len
        )
        end_idx = min(current_sample + frame_len, seq_samples)

        if frame is None:
            frame_meta['boundary'] = {'start': current_sample, 'end': end_idx}
            frame_meta['frame_duration'] = frame_duration
            full_metadata['frames'].append(frame_meta)
            current_sample = end_idx
            prev_tech = 'noise'
        else:
            p_frame = torch.mean(torch.abs(frame)**2)
            if p_frame > 0:
                frame = frame / torch.sqrt(p_frame)
            if frame_meta.get('rogue_details', {}).get('erp_violation'):
                frame = frame * frame_meta['rogue_details']['power_scale']
                full_metadata['erp_violations'] += 1
            faded_frame = apply_flat_fading(frame)
            ramped_frame = apply_pa_ramp(faded_frame, fs)
            actual_len = end_idx - current_sample
            if len(ramped_frame) > actual_len:
                ramped_frame = ramped_frame[:actual_len]
            sequence[current_sample:end_idx] += ramped_frame            
            clean_sequence[current_sample:end_idx] += ramped_frame
            full_metadata['total_tx_time'] += actual_len / fs
            frame_meta['boundary'] = {'start': current_sample, 'end': end_idx}
            frame_meta['frame_duration'] = actual_len / fs
            full_metadata['frames'].append(frame_meta)
            current_sample = end_idx
            prev_tech = signal_type
        frame_count += 1
    snr_linear = 10**(snr_db / 10)
    noise_power = 1.0 / snr_linear
    noise_std = torch.sqrt(torch.tensor(noise_power / 2.0, device=device, dtype=torch.float32))
    noise_real = torch.randn(seq_samples, device=device, dtype=torch.float32) * noise_std
    noise_imag = torch.randn(seq_samples, device=device, dtype=torch.float32) * noise_std
    noise = torch.complex(noise_real, noise_imag)
    sequence += noise
    full_metadata['signal_power'] = 1.0
    full_metadata['noise_power'] = float(noise_power)
    full_metadata['duty_cycle'] = full_metadata['total_tx_time'] / seq_len
    snapshot_len_samples = int(snapshot_len_sec * fs)
    overlap_samples = int(overlap_sec * fs)
    step_samples = snapshot_len_samples - overlap_samples
    max_jitter = min(int(jitter_max * fs), step_samples // 2)
    first_start = np.random.randint(0, max_jitter + 1) if max_jitter > 0 else 0
    snapshots = []
    snapshot_metadata = []
    start_sample = first_start
    snap_idx = 0

    while start_sample + snapshot_len_samples <= seq_samples:
        end_sample = start_sample + snapshot_len_samples
        snap = sequence[start_sample:end_sample]
        clean_snap = clean_sequence[start_sample:end_sample]
        clean_power_val = float(torch.mean(torch.abs(clean_snap)**2).item())
        contained = []
        for i, fm in enumerate(full_metadata['frames']):
            b = fm['boundary']
            if b['start'] < end_sample and b['end'] > start_sample:
                ov_start = max(b['start'], start_sample)
                ov_end = min(b['end'], end_sample)
                contained.append({
                    'frame_idx': i,
                    'type': fm['signal_type'],
                    'overlap': [ov_start, ov_end],
                    'overlap_ratio': (ov_end - ov_start) / max(1, b['end'] - b['start'])
                })
        snapshots.append(snap)
        snapshot_metadata.append({
            'index': snap_idx,
            'start_sample': start_sample,
            'contained_frames': contained,
            'band_id': band_id,
            'snr_db': snr_db,
            'clean_power': clean_power_val
        })
        start_sample += step_samples
        snap_idx += 1
    if snapshots:
        snapshots = torch.stack(snapshots)
    else:
        snapshots = torch.empty((0, snapshot_len_samples), dtype=torch.complex64, device=device)
    
    return snapshots, snapshot_metadata, full_metadata

# Generate labels for snapshots based on contained frames (noise, lora, ieee, mixture-cochannel, mixture-adjacent).
def label_snapshots(snapshot_metadata, full_metadata):
    labels = []
    for snap_meta in snapshot_metadata:
        contained = snap_meta['contained_frames']
        if not contained:
            labels.append("noise")
            continue        
        types = set(c['type'] for c in contained)
        if 'mixture' in types:
            mix_frame_idx = next(c['frame_idx'] for c in contained if c['type'] == 'mixture')
            coex_params = full_metadata['frames'][mix_frame_idx].get('coex_params', {})
            interf_type = coex_params.get('interference_type', 'co-channel')
            if 'adjacent' in interf_type:
                labels.append('mixture-adjacent')
            else:
                labels.append('mixture-cochannel')
            continue
        if 'lora' in types and 'ieee' in types:
            labels.append('mixture-cochannel') 
            continue
        if len(types) == 1:
            labels.append(list(types)[0])
        else:
            durations = {}
            for c in contained:
                dur = c['overlap'][1] - c['overlap'][0]
                durations[c['type']] = durations.get(c['type'], 0) + dur
            labels.append(max(durations, key=durations.get))

    return labels


# Captures Rogues, SNR, AND Coexistence Ratios.
def extract_detailed_metadata(snap_meta, full_metadata, snapshot, fs):
    objects = []
    coex_params = None      
    for c in snap_meta['contained_frames']:
        frame_idx = c['frame_idx']
        f_meta = full_metadata['frames'][frame_idx]
        if coex_params is None and f_meta.get('coex_params'):
            coex_params = f_meta['coex_params']
        rogue_info = f_meta.get('rogue_details', {})
        violation_type = None
        if rogue_info.get('bw_violation'): violation_type = 'bw'
        elif rogue_info.get('oob_violation'): violation_type = 'oob'
        elif rogue_info.get('erp_violation'): violation_type = 'erp'
        obj = {
            "class": f_meta['signal_type'],
            "center_freq": f_meta['center_frequency'],
            "bw": (f_meta.get('lora_params') or {}).get('BW') or \
                  (f_meta.get('ieee_params') or {}).get('symbol_rate') or 0,
            "is_rogue": f_meta['is_rogue'],
            "violation_type": violation_type,
            "time_overlap_ratio": c['overlap_ratio']
        }
        objects.append(obj)
    meta = {
        "snapshot_index": snap_meta['index'],        
        "objects": objects,
        "coexistence_descriptors": coex_params,
        "snr_db": snap_meta.get('verified_snr'),
        "band_id": snap_meta.get('band_id', 'Unknown'),
        "signal_power": snap_meta.get('clean_power', 0.0),
        "label": snap_meta['label'], 
        "center_frequency": objects[0]['center_freq'] if objects else 868.0e6,
        "is_compliant": not any(o['is_rogue'] for o in objects)
    }
    return meta


 # --- VERIFICATION & TESTING ---
# Verify SNR for each snapshot using the cached buffer and returns dictionary mapping snapshot index to measured SNR.
def verify_snr_post_generation(snapshots, snapshot_metadata, full_metadata):
    snr_map = {}
    global_noise_power = full_metadata.get('noise_power', None)
    for meta in snapshot_metadata:
        has_active_frames = any(c['type'] != 'noise' for c in meta['contained_frames'])
        if not has_active_frames:
            continue
        if 'clean_power' in meta and global_noise_power is not None:
            P_signal = meta['clean_power']
            P_noise = global_noise_power
            if P_signal > 1e-12 and P_noise > 1e-12:
                snr = 10 * np.log10(P_signal / P_noise)
                snr_map[meta['index']] = snr
    return snr_map

# Realistic traffic (for example: Standard 90/10 distribution) 
# To obtain equal representation of all signal classes returns dictionary mapping snapshot index to measured SNR.
def generate_balanced_dataset(target_per_class=200, save_path="lamp_mmv_dataset.pkl"):
    all_snapshots = []
    all_metadata = []
    class_targets = {
        'noise': target_per_class,
        'lora': target_per_class,
        'ieee': target_per_class,
        'mixture-adjacent': target_per_class,
        'mixture-cochannel': target_per_class
    }
    counts = {k: 0 for k in class_targets}
    print(f"Generating balanced dataset: {target_per_class} samples per class")
    start_time = time.time()
    iteration = 0
    max_iterations = 10000
    
    while not all(counts[k] >= target_per_class for k in counts) and iteration < max_iterations:
        iteration += 1
        try:
            snaps, snap_metas, full_meta = generate_lpwan_spectrum_dataset_with_snapshots(
                seq_len=1.0, # Specify the needed sequence length
                snr_db=10 # Specify the target SNR
            )
            if len(snaps) == 0:
                continue
            labels = label_snapshots(snap_metas, full_meta)
            for i, (snap, sm, lbl) in enumerate(zip(snaps, snap_metas, labels)):
                if 'mixture' in lbl:
                    if 'co' in lbl.lower():
                        lbl = 'mixture-cochannel'
                    else:
                        lbl = 'mixture-adjacent'
                sm['label'] = lbl
                if lbl in counts and counts[lbl] < class_targets[lbl]:
                    has_active_frames = any(c['type'] != 'noise' for c in sm['contained_frames'])
                    P_noise = full_meta.get('noise_power')
                    P_signal = sm.get('clean_power')
                    if has_active_frames and P_noise is not None and P_signal is not None:
                        if P_signal > 1e-12 and P_noise > 1e-12:
                            sm['verified_snr'] = 10 * np.log10(P_signal / P_noise)
                    detailed = extract_detailed_metadata(sm, full_meta, snap, 1e6)
                    if 'verified_snr' in sm:
                        detailed['verified_snr'] = sm['verified_snr']
                    all_snapshots.append(snap.cpu().numpy())
                    all_metadata.append(detailed)
                    counts[lbl] += 1
            del snaps, snap_metas, full_meta
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if iteration % 20 == 0:
                gc.collect()

        except Exception as e:
            print(f"Error in iteration {iteration}: {e}")
            traceback.print_exc()
            continue
            
        if iteration % 50 == 0:
            elapsed = time.time() - start_time
            total = sum(counts.values())
            target_total = len(counts) * target_per_class
            pct = 100 * total / target_total if target_total > 0 else 0
            rate = iteration / elapsed if elapsed > 0 else 0
            eta_sec = (max_iterations - iteration) / rate if rate > 0 else 0
            print(f"[{pct:.1f}%] Iter {iteration}/{max_iterations} | "
                  f"Counts: {counts} | "
                  f"Rate: {rate:.1f} it/s | "
                  f"ETA: {eta_sec/60:.1f}m")
    dataset = {
        'snapshots': all_snapshots,
        'metadata': all_metadata,
        'counts': counts,
        'generation_time': time.time() - start_time
    }
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"\nDataset generation complete!")
    print(f"Total time: {dataset['generation_time']/60:.1f} minutes")
    print(f"Final counts: {counts}")
    print(f"Saved to: {save_path}")
    return dataset
 
    # --- MAIN EXECUTION ---
if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    generate_balanced_dataset(
        target_per_class=1000,
        save_path="robust10db.pkl"
    )
