import pickle
import numpy as np
import os

# --- CONFIGURATION ---
FILE_PATH = "lamp_mmv_robust10dbFthird1.pkl"  # Make sure this matches the dataset filename
SHOW_ALL = False          # Set to True to print every single snapshot (Warning: Huge output)
SHOW_FIRST_N = 50         # Set the first number of output to see
FILTER_LABEL = 'ieee'       # Set to 'lora', 'mixture', 'rogue', etc. to filter. None = Show all types.

def print_separator(char='-', length=80):
    print(char * length)

def inspect_dataset():
    print(f"Loading {FILE_PATH}...")
    
    if not os.path.exists(FILE_PATH):
        print(f"Error: File '{FILE_PATH}' not found.")
        return

    try:
        with open(FILE_PATH, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    snapshots = data['snapshots']
    metadata = data['metadata']
    counts = data.get('counts', {})
    
    print_separator('=')
    print(f"DATASET SUMMARY")
    print(f"Total Snapshots: {len(snapshots)}")
    print(f"Class Balance:   {counts}")
    print(f"Gen Time:        {data.get('generation_time', 0):.2f} sec")
    print_separator('=')
    
    print("\n[SNAPSHOT DETAILS]")
    
    printed_count = 0
    
    for i, meta in enumerate(metadata):
        label = meta.get('label', 'unknown')
        
        # Check specific filters
        is_rogue = meta.get('regulatory_violations', False)
        is_dc_viol = meta.get('duty_cycle_violation', False)
        
        if FILTER_LABEL:
            if FILTER_LABEL == 'rogue' and not (is_rogue or is_dc_viol): continue
            elif FILTER_LABEL != 'rogue' and FILTER_LABEL not in label: continue

        # --- PRINTING LOGIC ---
        print(f"Snapshot #{i} | Label: {label.upper()}")
        
        freq_mhz = meta.get('center_frequency', 0) / 1e6
        snr = meta.get('verified_snr', 'N/A')
        snr_str = f"{snr:.2f} dB" if isinstance(snr, (int, float)) else str(snr)
        
        print(f"  > Freq: {freq_mhz:.3f} MHz | Band: {meta.get('band_id', '?')} | SNR: {snr_str}")
        
        viol_flags = []
        if is_dc_viol: viol_flags.append("DUTY_CYCLE")
        
        # Check Rogue Details
        if 'rogue_details' in meta and meta['rogue_details']:
            rd = meta['rogue_details']
            if rd.get('bw_violation'): viol_flags.append("BANDWIDTH_VIOLATION")
            if rd.get('oob_violation'): viol_flags.append("OOB_EMISSION")
            if rd.get('erp_violation'): viol_flags.append(f"ERP_VIOLATION (Scale {rd.get('power_scale', 1.0):.2f}x)")
        
        if viol_flags:
            print(f"  > \033[91mVIOLATIONS: {', '.join(viol_flags)}\033[0m") # Red text for visibility
        else:
            print(f"  > Status: Compliant")

        if meta.get('lora_params'):
            lp = meta['lora_params']
            print(f"  > LoRa Params: SF{lp.get('SF')} | BW {lp.get('BW')/1e3:.0f} kHz")
            
        if meta.get('ieee_params'):
            ip = meta['ieee_params']
            print(f"  > IEEE Params: {ip.get('symbol_rate')/1e3:.0f} ksym/s | Mod: {ip.get('modulation')}")
            
        print_separator()
        
        printed_count += 1
        if not SHOW_ALL and printed_count >= SHOW_FIRST_N:
            print(f"... Stopping after {SHOW_FIRST_N} entries (Change configuration to see more) ...")
            break

if __name__ == "__main__":
    inspect_dataset()
