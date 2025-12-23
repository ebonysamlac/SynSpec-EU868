# SynSpec-EU868: Regulatory-Aware Synthetic Spectrum Generation for LoRa and IEEE 802.15.4g Coexistence
This paper introduces a regulatory-aware, cross-layer spectrum simulator for the coexistence of LoRa and IEEE 802.15.4g systems in the 868 MHz band. The proposed framework generates continuous-time complex baseband I/Q sequences driven by stochastic traffic models and constrained by ETSI regulations. It combines PHY-accurate waveform synthesis with realistic channel effects and explicitly injects regulatory violations, including bandwidth, duty-cycle, out-of-band, and power exceedance.
The simulator, SynSpec-EU868, produces labeled spectrum snapshots suitable for training data-driven spectrum monitoring and compliance detection models, providing a practical tool for next-generation regulatory-aware sensing systems.

Installation
Please follow the instructions.

Usage
Clone the repository: git clone https://github.com/ebonysamlac/SynSpec-EU868.git

use the simulator.py file to generate the dataset.
