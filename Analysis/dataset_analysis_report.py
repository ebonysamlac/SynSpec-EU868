
# Generate comprehensive dataset analysis report
import pickle
import numpy as np
import matplotlib.pyplot as plt

def get_band_from_freq(freq_hz):
    f_mhz = freq_hz / 1e6
    if 865.0 <= f_mhz < 868.0:
        return 'Band L (865-868)'
    elif 868.0 <= f_mhz < 868.6:
        return 'Band M (868-868.6)'
    elif 869.4 <= f_mhz <= 869.65:
        return 'Band P (869.4-869.65)'
    else:
        return 'Other/Rogue'

def analyze_lpwan_dataset(pkl_path):
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File '{pkl_path}' not found.")
        return
    snaps = data['snapshots']
    metas = data['metadata']
    counts = data['counts']
    print("=" * 60)
    print(f"LPWAN Dataset Report: {pkl_path}")
    print("=" * 60)
    bands = []
    for m in metas:
        if m.get('center_frequency'):
            bands.append(get_band_from_freq(m['center_frequency']))
        else:
            bands.append('Unknown')
    ieee_bws = []
    lora_bws = []
    sf_dist = []  
    rogue_count = 0
    clean_count = 0
    
    for m in metas:
        is_snapshot_rogue = False
        if 'objects' in m:
            for obj in m['objects']:
                if obj['class'] == 'ieee':
                    ieee_bws.append(obj['bw'] / 1e3) # kHz
                elif obj['class'] == 'lora':
                    lora_bws.append(obj['bw'] / 1e3) # kHz
                    if obj.get('sf'):
                        sf_dist.append(obj['sf'])                
                if obj['is_rogue']:
                    is_snapshot_rogue = True        
        if is_snapshot_rogue:
            rogue_count += 1
        else:
            clean_count += 1
    powers = [m['signal_power'] for m in metas]
    valid_powers = [10 * np.log10(p) for p in powers if p > 1e-12]
    print("\n1. Frequency Band Distribution:")
    unique_bands = sorted(list(set(bands)))
    for b in unique_bands:
        c = bands.count(b)
        print(f"   {b:25s}: {c:5d} ({c/len(bands)*100:.1f}%)")
    print("\n2. Compliance Overview:")
    print(f"   Compliant Snapshots: {clean_count:5d} ({clean_count/len(metas)*100:.1f}%)")
    print(f"   Rogue/Violating:     {rogue_count:5d} ({rogue_count/len(metas)*100:.1f}%)")

    # Visualize
    fig, axes = plt.subplots(3, 4, figsize=(24, 15)) 
    fig.suptitle(f"Dataset Analysis: {pkl_path}", fontsize=16)
    axes[0,0].bar(counts.keys(), counts.values(), color='skyblue', edgecolor='black')
    axes[0,0].set_title('1. Class Balance')
    axes[0,0].tick_params(axis='x', rotation=45)
    band_counts = [bands.count(b) for b in unique_bands]
    axes[0,1].pie(band_counts, labels=unique_bands, autopct='%1.1f%%', startangle=140, colors=plt.cm.Pastel1.colors)
    axes[0,1].set_title('2. Frequency Bands')
    axes[0,2].pie([clean_count, rogue_count], labels=['Compliant', 'Violation'], autopct='%1.1f%%', 
                  colors=['#66b3ff', '#ff9999'], wedgeprops=dict(width=0.4))
    axes[0,2].set_title('3. Regulatory Compliance')
    if valid_powers:
        axes[0,3].hist(valid_powers, bins=40, color='green', alpha=0.6)
        axes[0,3].set_title('4. Signal Power (dB)')
        axes[0,3].set_xlabel('dB')
    if lora_bws:
        axes[1,0].hist(lora_bws, bins=[100, 200, 300, 600], color='orange', edgecolor='black')
        axes[1,0].set_title('5. LoRa Bandwidths (kHz)')
        axes[1,0].set_xticks([125, 250, 500])
    else:
        axes[1,0].text(0.5, 0.5, "No LoRa Data", ha='center')
    if ieee_bws:
        axes[1,1].hist(ieee_bws, bins=10, color='purple', edgecolor='black')
        axes[1,1].set_title('6. IEEE Bandwidths/Rates (kHz)')
    else:
        axes[1,1].text(0.5, 0.5, "No IEEE Data", ha='center')
    viol_types = []
    for m in metas:
        if 'objects' in m:
            for obj in m['objects']:
                if obj.get('violation_type'):
                    viol_types.append(obj['violation_type'])
    
    if viol_types:
        v_counts = {v: viol_types.count(v) for v in set(viol_types)}
        axes[1,2].bar(v_counts.keys(), v_counts.values(), color='red', alpha=0.7)
        axes[1,2].set_title('7. Violation Types')
    else:
        axes[1,2].text(0.5, 0.5, "No Violations Found", ha='center')
    snap_idx = 0
    for i, m in enumerate(metas):
        if m['label'] != 'noise': snap_idx = i; break
    raw = snaps[snap_idx]
    snap_np = raw.numpy() if hasattr(raw, 'numpy') else np.array(raw)
    fft = np.fft.fftshift(np.fft.fft(snap_np))
    freqs = np.linspace(-0.5, 0.5, len(fft))
    axes[1,3].plot(freqs, 20*np.log10(np.abs(fft) + 1e-9))
    axes[1,3].set_title(f'8. Example: {metas[snap_idx]["label"]}')
    axes[1,3].grid(alpha=0.3)
    axes[1,3].set_ylim(-40, 60)
    if sf_dist:
        axes[2,0].hist(sf_dist, bins=np.arange(6.5, 13.5, 1), color='gold', edgecolor='black', alpha=0.7)
        axes[2,0].set_title('9. LoRa Spreading Factors')
        axes[2,0].set_xlabel('SF')
        axes[2,0].set_xticks(range(7, 13))
    else:
        axes[2,0].text(0.5, 0.5, "No SF Data", ha='center')        
    axes[2,1].axis('off')
    axes[2,2].axis('off')
    axes[2,3].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_name = 'dataset_analysis10db.png'
    plt.savefig(save_name, dpi=150)
    print(f"\nSaved analysis to '{save_name}'")
    plt.show()
if __name__ == '__main__':
    analyze_lpwan_dataset('dataset.pkl') # add dataset path
