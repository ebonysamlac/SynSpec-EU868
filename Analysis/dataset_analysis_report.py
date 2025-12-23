
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

    # --- DATA EXTRACTION ---
    bands = []
    for m in metas:
        if m.get('center_frequency'):
            bands.append(get_band_from_freq(m['center_frequency']))
        else:
            bands.append('Unknown')
            
    ieee_rates = []
    for m in metas:
        if m.get('ieee_params'):
            ieee_rates.append(int(m['ieee_params']['symbol_rate'] / 1000))

    sf_dist = []
    for m in metas:
        if m.get('lora_params'):
            sf_dist.append(m['lora_params']['SF'])


    dc_status = ['Violation' if m.get('duty_cycle_violation') else 'Compliant' for m in metas]

    powers = [m['signal_power'] for m in metas]
    valid_powers = [10 * np.log10(p) for p in powers if p > 1e-12]

    # TEXT REPORT 
    print("\n1. Frequency Band Distribution:")
    unique_bands = sorted(list(set(bands)))
    for b in unique_bands:
        c = bands.count(b)
        print(f"   {b:25s}: {c:5d} ({c/len(bands)*100:.1f}%)")

    print("\n2. Duty Cycle Compliance:")
    c_viol = dc_status.count('Violation')
    print(f"   Compliant: {len(dc_status) - c_viol:5d} ({(len(dc_status) - c_viol)/len(dc_status)*100:.1f}%)")
    print(f"   Violation: {c_viol:5d} ({c_viol/len(dc_status)*100:.1f}%)")

    # VISUALIZATION 
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle(f"Dataset Analysis: {pkl_path}", fontsize=16)

    # Class Distribution
    axes[0,0].bar(counts.keys(), counts.values(), color='skyblue', edgecolor='black')
    axes[0,0].set_title('1. Class Balance')
    axes[0,0].tick_params(axis='x', rotation=45)

    # Frequency Band Distribution (Pie Chart)
    band_counts = [bands.count(b) for b in unique_bands]
    axes[0,1].pie(band_counts, labels=unique_bands, autopct='%1.1f%%', startangle=140, colors=plt.cm.Pastel1.colors)
    axes[0,1].set_title('2. Frequency Bands')

    # Duty Cycle Status (Donut Chart)
    dc_counts = [dc_status.count('Compliant'), dc_status.count('Violation')]
    axes[0,2].pie(dc_counts, labels=['Compliant', 'Violation'], autopct='%1.1f%%', 
                  colors=['#66b3ff', '#ff9999'], wedgeprops=dict(width=0.4))
    axes[0,2].set_title('3. Duty Cycle Status')

    # Regulatory Violations (Bar)
    reg_counts = {'Rogue': sum(1 for m in metas if m.get('regulatory_violations')),
                  'Clean': sum(1 for m in metas if not m.get('regulatory_violations'))}
    axes[0,3].bar(reg_counts.keys(), reg_counts.values(), color=['red', 'green'], alpha=0.7)
    axes[0,3].set_title('4. Rogue Signal Presence')

    # LoRa SF Distribution
    if sf_dist:
        axes[1,0].hist(sf_dist, bins=np.arange(6.5, 13.5, 1), color='orange', edgecolor='black')
        axes[1,0].set_title('5. LoRa Spreading Factors')
        axes[1,0].set_xticks(range(7, 13))
    else:
        axes[1,0].text(0.5, 0.5, "No LoRa Data", ha='center')

    # IEEE Symbol Rates
    if ieee_rates:
        axes[1,1].hist(ieee_rates, bins=[25, 75, 125, 175, 225], color='purple', edgecolor='black')
        axes[1,1].set_title('6. IEEE Symbol Rates')
        axes[1,1].set_xticks([50, 100, 150, 200])
        axes[1,1].set_xlabel('ksym/s')
    else:
        axes[1,1].text(0.5, 0.5, "No IEEE Data", ha='center')

    # Signal Power
    if valid_powers:
        axes[1,2].hist(valid_powers, bins=40, color='green', alpha=0.6)
        axes[1,2].set_title('7. Ghost Buffer Power (dB)')
        axes[1,2].set_xlabel('dB')

    # Example Spectrum
    snap_idx = 0
    for i, m in enumerate(metas):
        if m['label'] != 'noise': snap_idx = i; break
    
    raw = snaps[snap_idx]
    snap_np = raw.numpy() if hasattr(raw, 'numpy') else np.array(raw)
    fft = np.fft.fftshift(np.fft.fft(snap_np))
    freqs = np.linspace(-0.5, 0.5, len(fft))
    
    axes[1,3].plot(freqs, 20*np.log10(np.abs(fft) + 1e-9))
    axes[1,3].set_title(f'8. Spectrum Example: {metas[snap_idx]["label"]}')
    axes[1,3].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    plt.savefig('dataset_comprehensive_analysis.png', dpi=150)
    print(f"\nSaved analysis to 'dataset_comprehensive_analysis.png'")
    plt.show()

if __name__ == '__main__':
    analyze_lpwan_dataset('lamp_mmv_robust10dbFthird1.pkl')
