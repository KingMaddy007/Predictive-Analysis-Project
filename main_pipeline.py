"""
COMPLETE PIPELINE — MAMBA-RUL END-TO-END

This script runs the complete Mamba-RUL pipeline from data loading to model testing.
It demonstrates the integration of all sections.

Usage:
    python main_pipeline.py

Author: Mamba-RUL Project
Date: 2026-02-10
"""

import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from section1_load_data import IMSDataLoader
from section2_preprocessing import VibrationPreprocessor
from section3_sequence_gen import SequenceGenerator
from section4_dataset import create_dataloaders
from section5_mamba_model import MambaRULModel, test_model_forward_pass


def print_section_header(section_num: int, title: str):
    """Print a formatted section header."""
    print("\n")
    print("=" * 80)
    print(f"SECTION {section_num}: {title}")
    print("=" * 80)
    print()


def main():
    """
    Run the complete Mamba-RUL pipeline.
    """
    
    print("=" * 80)
    print(" " * 20 + "MAMBA-RUL COMPLETE PIPELINE")
    print("=" * 80)
    print()
    print("Project: Mamba-RUL - Selective State Space Approach for RUL Estimation")
    print("Dataset: NASA IMS Bearing Dataset")
    print("Task: Remaining Useful Life Prediction")
    print()
    
    # Configuration
    config = {
        # Data loading
        'dataset_path': Path(__file__).parent / "IMS",
        'test_name': '1st_test',
        'bearing_column': 0,
        'max_files': None,  # None = load all files
        
        # Preprocessing
        'normalization_method': 'standard',  # 'standard' or 'minmax'
        
        # Sequence generation
        'window_size': 2048,
        'stride': 512,
        'normalize_rul': True,
        
        # DataLoader
        'train_split': 0.8,
        'batch_size': 32,
        'shuffle_train': True,
        'num_workers': 0,
        
        # Model
        'input_dim': 1,
        'd_model': 128,
        'd_state': 16,
        'd_conv': 4,
        'expand': 2,
        'num_layers': 4,
        'dropout': 0.1,
        'use_mamba': True,
        
        # General
        'random_seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("Configuration:")
    print("-" * 80)
    for key, value in config.items():
        print(f"  {key:<25s}: {value}")
    print("-" * 80)
    
    # Set random seeds
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['random_seed'])
    
    # ========================================================================
    # SECTION 1: LOAD DATA
    # ========================================================================
    print_section_header(1, "LOAD NASA IMS DATASET")
    
    try:
        loader = IMSDataLoader(config['dataset_path'])
        
        # Show available tests
        tests = loader.get_available_tests()
        print(f"Available tests: {tests}")
        print()
        
        # Load data
        signal, files = loader.load_test_data(
            test_name=config['test_name'],
            bearing_column=config['bearing_column'],
            max_files=config['max_files']
        )
        
        print(f"\n✓ Loaded {len(files)} files")
        print(f"✓ Total signal length: {len(signal):,} samples")
        
        # Plot sample
        loader.plot_signal_sample(signal, sample_length=10000)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure the IMS dataset is in the correct location.")
        return
    
    # ========================================================================
    # SECTION 2: PREPROCESS DATA
    # ========================================================================
    print_section_header(2, "DATA PREPROCESSING")
    
    preprocessor = VibrationPreprocessor(
        normalization_method=config['normalization_method']
    )
    
    original_signal = signal.copy()
    signal, preprocess_stats = preprocessor.preprocess(signal, fit_scaler=True)
    
    print("\nPreprocessing Statistics:")
    print("-" * 80)
    for key, value in preprocess_stats.items():
        if isinstance(value, float):
            print(f"  {key:<30s}: {value:.6f}")
        else:
            print(f"  {key:<30s}: {value}")
    
    # Visualization
    preprocessor.plot_comparison(original_signal, signal, sample_length=5000)
    
    # ========================================================================
    # SECTION 3: GENERATE SEQUENCES
    # ========================================================================
    print_section_header(3, "SEQUENCE GENERATION (WINDOWING)")
    
    generator = SequenceGenerator(
        window_size=config['window_size'],
        stride=config['stride'],
        normalize_rul=config['normalize_rul']
    )
    
    X, y = generator.generate_sequences(signal, verbose=True)
    
    print(f"\n✓ Generated {len(X):,} sequences")
    print(f"✓ X shape: {X.shape}")
    print(f"✓ y shape: {y.shape}")
    
    # Visualizations
    generator.visualize_windows(X, y, num_samples=6)
    generator.plot_rul_distribution(y)
    
    # ========================================================================
    # SECTION 4: CREATE DATALOADERS
    # ========================================================================
    print_section_header(4, "DATASET & DATALOADER CREATION")
    
    train_loader, val_loader, dataloader_info = create_dataloaders(
        X, y,
        train_split=config['train_split'],
        batch_size=config['batch_size'],
        shuffle_train=config['shuffle_train'],
        num_workers=config['num_workers'],
        add_channel_dim=True,
        random_seed=config['random_seed']
    )
    
    print(f"\n✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}")
    
    # Test batch loading
    batch_x, batch_y = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  X shape: {batch_x.shape}")
    print(f"  y shape: {batch_y.shape}")
    
    # ========================================================================
    # SECTION 5: BUILD AND TEST MODEL
    # ========================================================================
    print_section_header(5, "MAMBA MODEL IMPLEMENTATION")
    
    model = MambaRULModel(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        d_state=config['d_state'],
        d_conv=config['d_conv'],
        expand=config['expand'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        use_mamba=config['use_mamba']
    )
    
    print(model.get_model_summary())
    
    # Test forward pass
    print("\nTesting model with real batch from DataLoader:")
    print("-" * 80)
    
    model = model.to(config['device'])
    batch_x = batch_x.to(config['device'])
    
    model.eval()
    with torch.no_grad():
        predictions = model(batch_x)
    
    print(f"✓ Forward pass successful!")
    print(f"  Input shape: {batch_x.shape}")
    print(f"  Output shape: {predictions.shape}")
    print(f"  Predictions range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
    print(f"  True RUL range: [{batch_y.min().item():.4f}, {batch_y.max().item():.4f}]")
    
    # ========================================================================
    # PIPELINE COMPLETE
    # ========================================================================
    print("\n")
    print("=" * 80)
    print(" " * 25 + "PIPELINE COMPLETE!")
    print("=" * 80)
    print()
    print("✓ All sections executed successfully")
    print("✓ Model is ready for training")
    print()
    print("Next Steps:")
    print("  1. Implement training loop with loss function (MSE, MAE, etc.)")
    print("  2. Add validation metrics (RMSE, R², etc.)")
    print("  3. Implement early stopping and model checkpointing")
    print("  4. Experiment with hyperparameters")
    print("  5. Compare with baseline models (LSTM, Transformer)")
    print()
    print("=" * 80)
    
    return {
        'model': model,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'config': config,
        'preprocessor': preprocessor,
        'generator': generator
    }


if __name__ == "__main__":
    results = main()
