# SynSpec-EU868: Regulatory-Aware Synthetic Spectrum Generation for LoRa and IEEE 802.15.4g Coexistence
SynSpec-EU868 is a regulatory-aware, cross-layer spectrum simulator designed to address data scarcity in sub-GHz spectrum monitoring. Unlike traditional physical-layer signal generators, this framework integrates MAC-layer regulatory logic with PHY-layer waveform synthesis to generate realistic, stochastic datasets of compliant and non-compliant LPWAN transmissions (LoRa and IEEE 802.15.4g). The proposed framework generates continuous-time complex baseband I/Q sequences driven by stochastic traffic models and constrained by ETSI regulations.

This tool was developed to support research in Data-Driven Wireless Spectrum Supervision, enabling the training of data-driven spectrum monitoring and compliance detection models, and providing a practical tool for next-generation regulatory-aware sensing systems.

# 🚀 Key Features
- Cross-Layer Synthesis: Models interactions between PHY parameters (waveform shape) and MAC constraints (duty cycle, timestamps).
- Stochastic Violation Generation: Automatically generates "Rogue" devices with continuous deviations in:
  - Bandwidth ($B > B_{limit}$)
  - Frequency/OOB (Carrier Frequency Offset)
  - ERP/Power (Effective Radiated Power excesses)
  - Duty Cycle (Time-domain persistence violations)
- Cached Buffer Architecture: A memory-optimized engine that separates deterministic signal generation from stochastic noise/channel addition, enabling massive dataset generation.
- Realistic Channel Impairments: Includes Rayleigh fading, AWGN, and spectral leakage artifacts.
- Coexistence Modeling: Simulates uncoordinated collisions (Co-channel) and neighbor interference (Adjacent-channel).

# 📦 Output Format
The simulator exports datasets as serialized Python objects (.pkl files) containing raw IQ samples and rich, parametric ground truth metadata.

## Output Data Structure
Each pickle file contains a dictionary with two main keys:

data = {

    'snapshots': [ ... ],  # List of raw IQ tensors/arrays
    
    'metadata':  [ ... ]   # List of dictionaries containing ground truth
    
}

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

   | Key                   | Type     | Description                                                            |
   |-----------------------|----------|------------------------------------------------------------------------|
   | label                 | str      | Signal class: lora, ieee, mixture-cochannel, mixture-adjacent, noise.  |
   | snr_db                | float    | Signal-to-Noise Ratio (dB) of the snapshot.                            |
   | regulatory_violations | bool     | Boolean flag (True/False) if any violation is present.                 |
   | rogue_details         | dict     | Dictionary containing exact violation metrics.                         |
   | bw_violation bool     | bool     | True if bandwidth exceeds limit.                                       |
   | forced_bw             | float    | The actual bandwidth used (e.g., 500 kHz for a 250 kHz limit).         |
   | freq_offset           | float    | Frequency drift in MHz (e.g., 0.4 MHz).                                |
   | power_scale           | float    | ERP scaling factor (e.g., 1.5x limit).                                 |
   | lora_params           | dict     | Ground truth PHY params: SF, BW.                                       |
   | ieee_params           | dict     | Ground truth PHY params: symbol_rate, mod_index.                       |
   | coex_params           | dict     | For mixtures: Power ratio and frequency separation.                     |

   

## Usage
Clone the repository: git clone https://github.com/ebonysamlac/SynSpec-EU868.git

To run the default main simulation script to generate dataset.

use the simulator.py file to generate the dataset.

python SynSpec/simulator.py --target_per_class 1000 --save_path dataset.pkl

## Dataset
The dataset generated for validation is hosted on Zenodo:
https://doi.org/10.5281/zenodo.18036026

# 📜 Citation & License
This project is open-source under the MIT License.

If you use this simulator or the generated datasets in your research, please cite:

      @simdataset {ebo2025lpwan,

      author       = {Ebo, Ife Olalekan},
  
      title        = {Regulatory-Aware Synthetic Spectrum Generation for LoRa and IEEE 802.15.4g Coexistence},
  
      year         = {2025},
  
      doi          = {10.5281/zenodo.18036026},
  
      publisher    = {Zenodo}
      }
