import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# --- CONFIGURATION ---
FILE_PATH = "lamp_mmv_robust10dbFthird1.pkl" 
FS = 1e6

# !!! SET THE ID YOU WANT TO VIEW HERE !!!
TARGET_SNAPSHOT_ID = 490  

def load_data(file_path):
    print(f"Loading {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return None
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def generate_caption(meta):
    lines = []    
    label = meta.get('label', 'Unknown')
    lines.append(f"Label: {label}")    
    rogue = meta.get('rogue_details', {})
    if not rogue:
        lines.append("[ COMPLIANT ]")
    else:
        if rogue.get('bw_violation'): lines.append("!! BW VIOLATION !!")
        if rogue.get('oob_violation'): lines.append("!! OOB VIOLATION !!")
        if rogue.get('erp_violation'): lines.append("!! ERP VIOLATION !!")
        if meta.get('duty_cycle_violation'): lines.append("!! DC VIOLATION !!")

    if 'lora_params' in meta and meta['lora_params']:
        sf = meta['lora_params'].get('SF')
        bw = meta['lora_params'].get('BW', 0) / 1e3
        lines.append(f"LoRa: SF{sf}, BW{bw:.0f}kHz")
        
    if 'ieee_params' in meta and meta['ieee_params']:
        rate = meta['ieee_params'].get('symbol_rate', 0) / 1e3
        lines.append(f"IEEE: {rate:.0f} kbps")

    return "\n".join(lines)

def plot_single_snapshot(sig, meta, fs, snapshot_id):    
    if hasattr(sig, 'numpy'):
        sig = sig.numpy()
    plt.figure(figsize=(10, 6))    
    f, t, Sxx = signal.spectrogram(sig, fs, return_onesided=False, 
                                   nperseg=512, noverlap=400, nfft=1024, mode='psd')
    Sxx = np.fft.fftshift(Sxx, axes=0)
    f = np.fft.fftshift(f)

    plt.pcolormesh(t*1000, f/1e3, 10 * np.log10(Sxx + 1e-12), 
                   shading='gouraud', cmap='inferno', vmin=-120, vmax=-40)
    plt.colorbar(label='Power Spectral Density (dB/Hz)')
    plt.title(f"Snapshot ID: {snapshot_id} | {meta.get('label', 'Signal')}", fontsize=14, fontweight='bold')
    plt.ylabel('Frequency (kHz)')
    plt.xlabel('Time (ms)')
    plt.ylim(-500, 500) # View window +/- 500 kHz
    plt.grid(alpha=0.3, linestyle='--')
    caption_text = generate_caption(meta)
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.text(0.02, 0.95, caption_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props, fontfamily='monospace')

    # Save and Show
    save_name = f"snapshot_{snapshot_id}_spectrogram.png"
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"Saved plot to '{save_name}'")
    plt.show()

if __name__ == "__main__":
    data = load_data(FILE_PATH)
    
    if data:
        snapshots = data['snapshots']
        metadata = data['metadata']
        
        if 0 <= TARGET_SNAPSHOT_ID < len(snapshots):
            target_sig = snapshots[TARGET_SNAPSHOT_ID]
            target_meta = metadata[TARGET_SNAPSHOT_ID]
            
            print(f"\n--- Metadata for Snapshot {TARGET_SNAPSHOT_ID} ---")
            for k, v in target_meta.items():
                print(f"{k}: {v}")
            print("----------------------------------------\n")
            
            # Visualize
            plot_single_snapshot(target_sig, target_meta, FS, TARGET_SNAPSHOT_ID)
        else:
            print(f"Error: Snapshot ID {TARGET_SNAPSHOT_ID} is out of range (Max: {len(snapshots)-1})")
