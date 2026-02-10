# Mamba-RUL: A Selective State Space Approach for Remaining Useful Life Estimation

## Project Overview

**Title:** Mamba-RUL: A Selective State Space Approach for Remaining Useful Life Estimation of High-Speed Rotating Machinery under Non-Stationary Loads

### The Core Concept

This project applies the Mamba (Selective State Space Model - S6) architecture to predict the Remaining Useful Life (RUL) of industrial bearings using high-frequency vibration data.

**Why Mamba?**
- **Transformers** crash on long sequences (quadratic complexity)
- **RNNs/LSTMs** forget long-term dependencies
- **Mamba S6** processes million-point vibration signals in linear time with selective memory

### Problem Statement

Modern industrial systems (Wind Turbines, CNC Machines, Jet Engines) generate high-frequency vibration data sampled at >20kHz. Traditional Deep Learning models fail to:
1. Process extremely long temporal sequences efficiently
2. Model long-term degradation trajectories
3. Handle non-stationary operating conditions (varying loads/speeds)

### Proposed Solution

Mamba-RUL uses the Selective State Space Model (S6) to:
- Ingest raw, long-sequence vibration data without extensive preprocessing
- Selectively remember fault frequencies while forgetting background noise
- Disentangle operational noise from structural degradation
- Predict continuous RUL under varying load conditions

### Expected Outcome

- **Accuracy:** >95% RUL prediction on NASA IMS Dataset
- **Speed:** 5x faster than Transformer baselines
- **Capability:** Superior long-term trend retention vs LSTM

## Dataset

**NASA IMS (Intelligent Maintenance Systems) Dataset**
- Real recordings of bearings from "Brand New" to "Complete Failure"
- Run-to-failure data over ~30 days
- Vibration sensors at 20kHz sampling rate
- Open source and IEEE-standard

## Project Structure

```
Predictive Project/
├── data/
│   └── IMS/                    # NASA IMS dataset
├── src/
│   ├── section0_setup.py       # Environment verification
│   ├── section1_load_data.py   # Data loading utilities
│   ├── section2_preprocessing.py # Data cleaning & normalization
│   ├── section3_sequence_gen.py  # Windowing & RUL labeling
│   ├── section4_dataset.py     # PyTorch Dataset & DataLoader
│   └── section5_mamba_model.py # Mamba RUL model architecture
├── notebooks/
│   └── exploratory_analysis.ipynb
├── requirements.txt
└── README.md
```

## Installation

See `requirements.txt` for dependencies.

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Each section can be run independently:

```python
# Section 0: Verify installation
python src/section0_setup.py

# Section 1: Load NASA IMS data
python src/section1_load_data.py

# Section 2: Preprocess data
python src/section2_preprocessing.py

# Section 3: Generate sequences
python src/section3_sequence_gen.py

# Section 4: Create PyTorch datasets
python src/section4_dataset.py

# Section 5: Build and test Mamba model
python src/section5_mamba_model.py
```

## Key Features

- **Modular Design:** Each processing step is isolated
- **Efficient Processing:** Handles million-point sequences
- **Reproducible:** Fixed random seeds and documented parameters
- **Visualizations:** Comprehensive plotting at each stage
- **Best Practices:** Type hints, docstrings, and comments

## Technical Details

- **Architecture:** Mamba SSM (Selective State Space Model)
- **Framework:** PyTorch
- **Data:** NASA IMS bearing vibration dataset
- **Task:** Regression (RUL prediction)
- **Input:** Raw time-series vibration signals
- **Output:** Continuous RUL value

## References

- Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- NASA IMS Bearing Dataset
- Prognostics and Health Management (PHM) literature
