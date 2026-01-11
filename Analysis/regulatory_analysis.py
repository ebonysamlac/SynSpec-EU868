import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

FILE_PATH = "robust10db.pkl"  # provide dataset path
FS = 1e6

def find_specific_example(metadata, snapshots, label_filter, condition):
    for i, meta in enumerate(metadata):
        if label_filter not in meta['label']:
            continue
            
        if label_filter == 'ieee' and condition == 'bw':
            continue
            
        is_dc = meta.get('duty_cycle_violation', False)
        is_bw = False
        is_oob = False
        is_erp = False
        if 'objects' in meta:
            for obj in meta['objects']:
                if label_filter in obj['class'] or 'mixture' in label_filter:
                    v_type = obj.get('violation_type')
                    if v_type == 'bw': is_bw = True
                    if v_type == 'oob': is_oob = True
                    if v_type == 'erp': is_erp = True
                    if obj.get('is_rogue'): 
                        pass         
        elif 'rogue_details' in meta:
            rogue = meta['rogue_details']
            if rogue.get('bw_violation'): is_bw = True
            if rogue.get('oob_violation'): is_oob = True
            if rogue.get('erp_violation'): is_erp = True
        match = False
        if condition == 'compliant':
            if not (is_dc or is_bw or is_oob or is_erp): match = True
        elif condition == 'bw':
            if is_bw: match = True
        elif condition == 'dc':
            if is_dc: match = True
        elif condition == 'oob':
            if is_oob: match = True
        elif condition == 'erp':
            if is_erp: match = True
            
        if match:
            return snapshots[i], meta
            
    return None, None

def plot_spectrogram(ax, sig, fs, title, caption_text):
    if hasattr(sig, 'numpy'):
        sig = sig.numpy()
    f, t, Sxx = signal.spectrogram(sig, fs, return_onesided=False, 
                                   nperseg=256, noverlap=200, nfft=512, mode='psd')
    Sxx = np.fft.fftshift(Sxx, axes=0)
    f = np.fft.fftshift(f)
    im = ax.pcolormesh(t*1000, f/1e3, 10 * np.log10(Sxx + 1e-12), 
                       shading='gouraud', cmap='inferno', vmin=-120, vmax=-40)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylabel('Freq (kHz)', fontsize=8)
    ax.set_xlabel('Time (ms)', fontsize=8)
    ax.set_ylim(-500, 500)
    ax.grid(alpha=0.2)
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    ax.text(0.05, 0.95, caption_text, transform=ax.transAxes, fontsize=7,
            verticalalignment='top', bbox=props, fontfamily='monospace')
            
    return im

def generate_caption(meta, condition):
    lines = []    
    lines.append(f"Band: {meta.get('band_id', '?')}")
    freq = meta.get('center_frequency', 868e6) / 1e6
    lines.append(f"Freq: {freq:.2f} MHz")
    
    if 'objects' in meta:
        for i, obj in enumerate(meta['objects']):
            cls = obj['class']
            bw = obj.get('bw', 0)            
            freq_str = ""
            if len(meta['objects']) > 1:
                offset_khz = (obj['center_freq'] - meta['center_frequency']) / 1e3
                freq_str = f"[@{offset_khz:+.0f}k] "
            if cls == 'lora':
                sf = obj.get('sf')
                sf_str = f"SF{sf}" if sf else ""
                lines.append(f"LoRa: {sf_str} BW{bw/1e3:.0f}kHz")                
            elif cls == 'ieee':
                lines.append(f"IEEE: {bw/1e3:.0f} kbps")    
    if condition == 'compliant':
        lines.append("[ COMPLIANT ]")
    elif condition == 'bw':
        lines.append("!! BW VIOL !!")
    elif condition == 'dc':
        lines.append("!! DC VIOL !!")
    elif condition == 'oob':
        lines.append("!! OOB VIOL !!")
    elif condition == 'erp':
        lines.append("!! ERP VIOL !!")
    return "\n".join(lines)


def run_visualization():
    print(f"Loading {FILE_PATH}...")
    try:
        with open(FILE_PATH, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    snapshots = data['snapshots']
    metadata = data['metadata']
    print(f"Loaded {len(snapshots)} snapshots.")
    conditions = [
        ('compliant', 'Compliant'),
        ('bw', 'Bandwidth Viol'),
        ('dc', 'Duty Cycle Viol'),
        ('oob', 'Out-of-Band Viol'),
        ('erp', 'ERP (Power) Viol')
    ]
    signal_types = [
        ('lora', 'LoRa Signal'),
        ('ieee', 'IEEE 802.15.4g'),
        ('mixture-cochannel', 'Co-Channel Mix'),
        ('mixture-adjacent', 'Adj-Channel Mix')
    ]
    fig, axes = plt.subplots(5, 4, figsize=(20, 18))
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    fig.suptitle(f"Spectrum Violation Analysis: {FILE_PATH}", fontsize=16, y=0.98)
    for row, (cond_key, cond_name) in enumerate(conditions):
        for col, (sig_key, sig_name) in enumerate(signal_types):
            ax = axes[row, col]
            sig, meta = find_specific_example(metadata, snapshots, sig_key, cond_key)
            if sig is not None:
                caption = generate_caption(meta, cond_key)
                title = f"{sig_name}\n({cond_name})"
                plot_spectrogram(ax, sig, FS, title, caption)
            else:
                ax.text(0.5, 0.5, "No Example Found", ha='center', color='gray')
                ax.set_title(f"{sig_name}\n({cond_name})")
                ax.axis('off')
                
    save_name = "10dbspectrogramplot.png" # provide name to save plot with
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"Success! Matrix plot saved to '{save_name}'")
    plt.show()

if __name__ == "__main__":
    run_visualization()
