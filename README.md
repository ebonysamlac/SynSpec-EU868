# SynSpec-EU868: Regulatory-Aware Synthetic Spectrum Generation for LoRa and IEEE 802.15.4g Coexistence
SynSpec-EU868 is a regulatory-aware, cross-layer spectrum simulator designed to address data scarcity in sub-GHz spectrum monitoring. Unlike traditional physical-layer signal generators, this framework integrates MAC-layer regulatory logic with PHY-layer waveform synthesis to generate realistic, stochastic datasets of compliant and non-compliant LPWAN transmissions (LoRa and IEEE 802.15.4g). 

The tool generates continuous-time complex baseband I/Q sequences driven by stochastic traffic models and constrained by ETSI regulations. It was developed to support research in **Data-Driven Wireless Spectrum Supervision**, enabling the training of robust deep learning models for interference classification and rogue device detection.


# 🚀 Key Features
- **Cross-Layer Synthesis:** Models interactions between PHY parameters (waveform shape) and MAC constraints (duty cycle, timestamps).
- **Stochastic Violation Generation:** Automatically generates "Rogue" devices with continuous deviations in:
  - **Bandwidth:** $B > B_{limit}$
  - **Frequency/OOB:** Carrier Frequency Offset violating band edges.
  - **ERP/Power:** Effective Radiated Power excesses.
  - **Duty Cycle:** Time-domain persistence violations.
- **Cached Buffer Architecture:** A memory-optimized engine that separates deterministic signal generation from stochastic noise/channel addition, enabling massive dataset generation.
- **Realistic Channel Impairments:** Includes Rayleigh fading, PA ramping, and AWGN.
- **Coexistence Modeling:** Simulates uncoordinated collisions (Co-channel) and neighbor interference (Adjacent-channel).

# 📦 Output Format
The simulator exports datasets as serialized Python objects (`.pkl` files) containing raw IQ samples and rich, parametric ground truth metadata.

## Output Data Structure
Each pickle file contains a dictionary with two main keys:

```python
data = {
    'snapshots': [ ... ],  # List of raw IQ samples (complex64), Shape: (N_samples, sequence_length)

    'metadata': [          # List of dictionaries (one per snapshot)
        {
            'snapshot_index': 42,
            'label': 'mixture-cochannel',     # Classification label
            'snr_db': 10.5,
            'center_frequency': 868000000.0,  # Anchor frequency
            'is_compliant': False,            # Global compliance flag
            
            # THE CORE GROUND TRUTH: List of all active signals in the window
            'objects': [
                {
                    'class': 'lora',
                    'center_freq': 868000000.0,
                    'bw': 125000.0,
                    'sf': 9,             # Spreading Factor (LoRa only)
                    'is_rogue': True,
                    'violation_type': 'bw',
                    'time_overlap_ratio': 1.0
                },
                {
                    'class': 'ieee',
                    'center_freq': 868050000.0,
                    'bw': 50000.0,       # Symbol Rate (Hz)
                    'sf': None,
                    'is_rogue': False,
                    'violation_type': None,
                    'time_overlap_ratio': 0.8
                }
            ],
            
            # Mixture Context
            'coexistence_descriptors': { 
                'power_ratio_db': 12.0, 
                'freq_separation': 50000.0 
            }, 
            'band_id': 'M'
        },
        # ... more snapshots
    ]
}

## Metadata Dictionary Keys
Each entry provides the exact parametric ground truth required for training.

   | Key                   | Type     | Description                                                            |
   |-----------------------|----------|------------------------------------------------------------------------|
   | label                 | str      | Signal class: lora, ieee, mixture-cochannel, mixture-adjacent, noise.  |
   | snr_db                | float    | Verified Signal-to-Noise Ratio (dB) of the snapshot.                            |
   | is_compliant          | bool     | True if all objects in the snapshot comply with ETSI regulations. |
   | objects               | list     | A list of dictionaries where each dictionary describes a single signal instance (see below).|
   | center_frequency      | float    | The center frequency (Hz) of the receiver window.|
   | coexistence_descriptors| dict    | Contains delta P (power ratio) and delta f (freq separation) forr mixture frames.|

## Object Dictionary Keys (Inside 'objects' list)

   | Key                   | Description                                                            |
   |-----------------------|------------------------------------------------------------------------|
   | class                 | Signal teechnology (lora, ieee).                                |
   | center_freq           | Exact center frequency of this specific signal instance.          |
   | bw                    | Bandwidth (Hz) or Symbol Rate (Hz) for IEEE 802.15.4g.                                       |
   | sf                    | Spreading Factor (LoRa only).                       |
   | is_rogue              | True if this specific object violates a regulation.                     |
   | violation_type        | Specifi violation category 'bw', 'oob', 'erp', or None.                     |
   

# Installation
Please follow the instructions.

## Install dependencies
pip install -r requirements.txt

## Requirements:
 - numpy
 - scipy
 - torch (for GPU acceleration and tensor operations)
 - matplotlib (for visualization)
 - pandas (for logging)

## Data Structure
1. Snapshots (snapshots)
   
    - Format: Complex-valued NumPy arrays (or PyTorch tensors).

    - Content: Raw IQ samples ($I + jQ$) representing the baseband signal after channel impairments.
  
    - Shape: (Sequence_Length,) (e.g., 20,000 samples).

2. Metadata (metadata)
   Each entry provides the exact parametric ground truth required.


## Usage
Clone the repository: git clone https://github.com/ebonysamlac/SynSpec-EU868.git

To run the default main simulation script to generate dataset.

use the simulator.py file to generate the dataset.

python SynSpec/simulator.py --target_per_class 1000 --save_path dataset.pkl

## Dataset
The dataset generated for validation of the simulator is hosted on Zenodo:
https://doi.org/10.5281/zenodo.18036026

# 📜 Citation & License
This project is open-source under the MIT License.

If you use this simulator or the generated datasets in your research, please cite:

      @simdataset {ebo2025lpwan,

      author       = {Ebo Ife Olalekan, Idowu, Ajayi, Lina Mroueh, and Youmni Ziade},
  
      title        = {Regulatory-Aware Synthetic Spectrum Generation for LoRa and IEEE 802.15.4g Coexistence},
  
      year         = {2025},
  
      doi          = {10.5281/zenodo.18036026},
  
      publisher    = {Zenodo}
      }

### Contact: 
For questions, email: ioebo@isep.fr
