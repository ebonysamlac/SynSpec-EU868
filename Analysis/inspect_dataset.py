# inspect dataset by filtering accorrding to signal types.
import pickle
import numpy as np
import os

FILE_PATH = "robust10db.pkl"  # add file path for the dataset
SHOW_ALL = False          
SHOW_FIRST_N = 50         
FILTER_LABEL = 'ieee'  # None, 'lora', 'ieee', 'mixture', 'rogue' # choose the signal type

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
        has_rogue_obj = False
        if 'objects' in meta:
            has_rogue_obj = any(o['is_rogue'] for o in meta['objects'])
        is_dc_viol = meta.get('duty_cycle_violation', False)
        is_rogue = has_rogue_obj or is_dc_viol or meta.get('regulatory_violations', False)
        if FILTER_LABEL:
            if FILTER_LABEL == 'rogue' and not is_rogue: continue
            elif FILTER_LABEL != 'rogue' and FILTER_LABEL not in label: continue
        print(f"Snapshot #{i} | Label: {label.upper()}")
        freq_mhz = meta.get('center_frequency', 0) / 1e6     
        snr = meta.get('snr_db', meta.get('verified_snr', 'N/A'))
        snr_str = f"{snr:.2f} dB" if isinstance(snr, (int, float)) else str(snr)
        print(f"  > Anchor Freq: {freq_mhz:.3f} MHz | Band: {meta.get('band_id', '?')} | SNR: {snr_str}")        
        
        viol_flags = []
        if is_dc_viol: viol_flags.append("DUTY_CYCLE_LIMIT")
        if 'objects' in meta:
            for idx, obj in enumerate(meta['objects']):
                otype = obj['class'].upper()
                bw_val = obj.get('bw', 0) / 1e3
                v_type = obj.get('violation_type')
                if v_type:
                    viol_flags.append(f"Obj{idx}_{otype}_{v_type.upper()}_VIOLATION")
                if obj.get('is_rogue') and not v_type:
                     viol_flags.append(f"Obj{idx}_{otype}_GENERIC_ROGUE")
                if otype == 'LORA':
                    sf = obj.get('sf')
                    sf_str = f"SF{sf} | " if sf is not None else ""
                    print(f"  > [Obj {idx}] LoRa: {sf_str}BW {bw_val:.0f} kHz")
                elif otype == 'IEEE':
                    print(f"  > [Obj {idx}] IEEE: {bw_val:.0f} kbps")
                elif otype == 'MIXTURE':
                    print(f"  > [Obj {idx}] Mixture Frame (Should be decomposed)")
        elif 'rogue_details' in meta:
             rd = meta['rogue_details']
             if rd.get('bw_violation'): viol_flags.append("BANDWIDTH_VIOLATION")
             if rd.get('oob_violation'): viol_flags.append("OOB_EMISSION")
             if rd.get('erp_violation'): viol_flags.append("ERP_VIOLATION")

        if viol_flags:
            print(f"  > \033[91mVIOLATIONS: {', '.join(viol_flags)}\033[0m") 
        else:
            print(f"  > Status: Compliant")
        print_separator()
        printed_count += 1
        if not SHOW_ALL and printed_count >= SHOW_FIRST_N:
            print(f"... Stopping after {SHOW_FIRST_N} entries ...")
            break

if __name__ == "__main__":
    inspect_dataset()
