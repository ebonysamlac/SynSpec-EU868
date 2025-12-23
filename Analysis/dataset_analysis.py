import pickle
import numpy as np
import matplotlib.pyplot as plt

def analyze_lpwan_dataset(pkl_path):
    """Generate comprehensive dataset report matching the new simulator structure"""
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File '{pkl_path}' not found.")
        return

    if not data['snapshots']:
        print("Dataset is empty.")
        return

    snaps = data['snapshots']
    metas = data['metadata']
    counts = data['counts']
    
    print("=" * 60)
    print(f"LPWAN Dataset Report: {pkl_path}")
    print("=" * 60)
    
    # 1. Class Balance
    print("\n1. Class Distribution:")
    total_snaps = sum(counts.values())
    for cls, cnt in counts.items():
        print(f"   {cls:20s}: {cnt:6d} ({cnt/total_snaps*100:.1f}%)")

    print("\n2. LoRa Parameters:")
    sf_dist = []
    for m in metas:
        if m.get('lora_params'):
            sf_dist.append(m['lora_params']['SF'])
            
    if sf_dist:
        for sf in range(7, 13):
            count = sf_dist.count(sf)
            pct = count/len(sf_dist)*100 if len(sf_dist) > 0 else 0
            print(f"   SF{sf}: {count:5d} ({pct:.1f}%)")
    else:
        print("   No LoRa frames found.")
  
    print("\n3. Regulatory Compliance:")
    dc_viol = sum(1 for m in metas if m.get('duty_cycle_violation'))
    reg_viol = sum(1 for m in metas if m.get('regulatory_violations'))
    
    print(f"   Duty cycle violations: {dc_viol:5d} ({dc_viol/len(metas)*100:.1f}%)")
    print(f"   Regulatory (Rogue):    {reg_viol:5d} ({reg_viol/len(metas)*100:.1f}%)")
    print("\n4. Measured Power Distribution (Cached Buffer):")
    powers = [m['signal_power'] for m in metas]
    if len(powers) > 0:
        valid_powers = [p for p in powers if p > 1e-12]
        if valid_powers:
            powers_db = 10 * np.log10(valid_powers)
            print(f"   Mean: {np.mean(powers_db):.2f} dB")
            print(f"   Std:  {np.std(powers_db):.2f} dB")
            print(f"   Min:  {np.min(powers_db):.2f} dB")
            print(f"   Max:  {np.max(powers_db):.2f} dB")
        else:
            print("   (All signals are effectively zero power/noise)")
    
    # --- VISUALIZATIONS ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    #Class Distribution
    axes[0,0].bar(counts.keys(), counts.values(), color='skyblue')
    axes[0,0].set_title('Class Distribution')
    axes[0,0].set_ylabel('Count')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    #SF Distribution
    if sf_dist:
        axes[0,1].hist(sf_dist, bins=np.arange(6.5, 13.5, 1), color='orange', edgecolor='black', alpha=0.7)
        axes[0,1].set_title('LoRa Spreading Factor Distribution')
        axes[0,1].set_xlabel('Spreading Factor (SF)')
        axes[0,1].set_xticks(range(7, 13))
    else:
        axes[0,1].text(0.5, 0.5, "No LoRa Data", ha='center')

    #Power Distribution
    if powers:
        # Plot in linear scale or log scale depending on preference. Here: Linear
        axes[1,0].hist(powers, bins=50, color='green', alpha=0.7)
        axes[1,0].set_title('Signal Power Distribution (Linear)')
        axes[1,0].set_xlabel('Power Magnitude')
    
    # Spectrogram (First Non-Noise Snapshot)
    snap_idx = 0
    for i, m in enumerate(metas):
        if m['label'] != 'noise':
            snap_idx = i
            break
            
    snap = snaps[snap_idx]
    snap_np = np.array(snap) # Ensure numpy
    fft = np.fft.fftshift(np.fft.fft(snap_np))
    freqs = np.linspace(-0.5, 0.5, len(fft))
    
    axes[1,1].plot(freqs, 20*np.log10(np.abs(fft) + 1e-9))
    axes[1,1].set_title(f'Example Spectrum: {metas[snap_idx]["label"]}')
    axes[1,1].set_xlabel('Normalized Frequency')
    axes[1,1].set_ylabel('Magnitude (dB)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=150)
    print(f"\nVisualization saved to dataset_analysis.png")
    plt.show()

if __name__ == '__main__':
    analyze_lpwan_dataset('lamp_mmv_robust0dbFirst1.pkl')
