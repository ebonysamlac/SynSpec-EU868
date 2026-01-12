
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

FILE_PATH = "dataset.pkl"  # provide dataset path
FS = 1e6
TARGET_SNAPSHOT_ID = 490  # provide the snapshot id to be inspected

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
    band = meta.get('band_id', '?')
    lines.append(f"Label: {label} ({band})")    
    if meta.get('duty_cycle_violation'): 
        lines.append("!! DC VIOLATION !!")
    if 'objects' in meta:
        for i, obj in enumerate(meta['objects']):
            cls = obj['class']
            bw = obj.get('bw', 0)

            if cls == 'lora':
                sf = obj.get('sf')
                sf_str = f"SF{sf}, " if sf is not None else ""
                lines.append(f"Obj {i} [LoRa]: {sf_str}BW{bw/1e3:.0f}kHz")
            elif cls == 'ieee':
                lines.append(f"Obj {i} [IEEE]: {bw/1e3:.0f} kbps")            
            v_type = obj.get('violation_type')
            if v_type:
                lines.append(f"  -> !! {v_type.upper()} VIOLATION !!")
            elif obj.get('is_rogue'):
                lines.append(f"  -> !! ROGUE (Unspecified) !!")
    if meta.get('coexistence_descriptors'):
        coex = meta['coexistence_descriptors']
        interf = coex.get('interference_type', 'N/A')
        lines.append(f"Interference: {interf}")
    return "\n".join(lines)

def plot_single_snapshot(sig, meta, fs, snapshot_id):    
    if hasattr(sig, 'numpy'):
        sig = sig.cpu().numpy() 
    elif hasattr(sig, 'cpu'):
        sig = sig.cpu().numpy()
    plt.figure(figsize=(10, 6))    
    f, t, Sxx = signal.spectrogram(sig, fs, return_onesided=False, 
                                   nperseg=512, noverlap=400, nfft=1024, mode='psd')
    Sxx = np.fft.fftshift(Sxx, axes=0)
    f = np.fft.fftshift(f)
    
    # Plot
    plt.pcolormesh(t*1000, f/1e3, 10 * np.log10(Sxx + 1e-12), 
                   shading='gouraud', cmap='inferno', vmin=-120, vmax=-40)
    cbar = plt.colorbar()
    cbar.set_label('Power Spectral Density (dB/Hz)')
    plt.title(f"Snapshot ID: {snapshot_id} | {meta.get('label', 'Signal')}", fontsize=14, fontweight='bold')
    plt.ylabel('Frequency (kHz)')
    plt.xlabel('Time (ms)')
    plt.ylim(-500, 500) # View window +/- 500 kHz
    plt.grid(alpha=0.3, linestyle='--')
    caption_text = generate_caption(meta)
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.text(0.02, 0.95, caption_text, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=props, fontfamily='monospace')
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
            print(f"Label: {target_meta.get('label')}")
            if 'objects' in target_meta:
                for idx, obj in enumerate(target_meta['objects']):
                    print(f"Obj {idx}: {obj}")
            print("----------------------------------------\n")
            
            # Visualize
            plot_single_snapshot(target_sig, target_meta, FS, TARGET_SNAPSHOT_ID)
        else:
            print(f"Error: Snapshot ID {TARGET_SNAPSHOT_ID} is out of range (Max: {len(snapshots)-1})")
