"""
SECTION 4 — DATASET & DATALOADER CREATION

This module creates PyTorch Dataset and DataLoader objects for training
the Mamba RUL model.

Author: Mamba-RUL Project
Date: 2026-02-10
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class RULDataset(Dataset):
    """
    PyTorch Dataset for Remaining Useful Life prediction.
    
    Wraps vibration sequences and RUL labels for use with PyTorch DataLoader.
    """
    
    def __init__(
        self,
        sequences: np.ndarray,
        rul_labels: np.ndarray,
        add_channel_dim: bool = True
    ):
        """
        Initialize the RUL Dataset.
        
        Args:
            sequences: Array of shape (num_samples, sequence_length)
            rul_labels: Array of shape (num_samples,)
            add_channel_dim: Whether to add channel dimension for compatibility
                           with certain architectures (makes shape: [batch, 1, seq_len])
        """
        self.sequences = torch.FloatTensor(sequences)
        self.rul_labels = torch.FloatTensor(rul_labels)
        self.add_channel_dim = add_channel_dim
        
        # Validate shapes
        assert len(self.sequences) == len(self.rul_labels), \
            "Number of sequences must match number of labels"
        
        print(f"✓ Dataset initialized with {len(self)} samples")
        print(f"  Sequence shape: {self.sequences.shape}")
        print(f"  Label shape: {self.rul_labels.shape}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Tuple of (sequence, rul_label)
        """
        sequence = self.sequences[idx]
        rul = self.rul_labels[idx]
        
        # Add channel dimension if requested: [seq_len] -> [1, seq_len]
        if self.add_channel_dim:
            sequence = sequence.unsqueeze(0)
        
        return sequence, rul
    
    def get_statistics(self) -> dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'num_samples': len(self),
            'sequence_length': self.sequences.shape[1],
            'sequence_mean': float(self.sequences.mean()),
            'sequence_std': float(self.sequences.std()),
            'rul_mean': float(self.rul_labels.mean()),
            'rul_std': float(self.rul_labels.std()),
            'rul_min': float(self.rul_labels.min()),
            'rul_max': float(self.rul_labels.max()),
        }
        return stats


def create_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    train_split: float = 0.8,
    batch_size: int = 32,
    shuffle_train: bool = True,
    num_workers: int = 0,
    add_channel_dim: bool = True,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Create training and validation DataLoaders.
    
    Args:
        X: Sequence array of shape (num_samples, sequence_length)
        y: RUL labels of shape (num_samples,)
        train_split: Fraction of data to use for training
        batch_size: Batch size for DataLoader
        shuffle_train: Whether to shuffle training data
        num_workers: Number of worker processes for data loading
        add_channel_dim: Whether to add channel dimension
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, info_dict)
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Calculate split index
    num_samples = len(X)
    split_idx = int(num_samples * train_split)
    
    print(f"Creating train/validation split...")
    print(f"  Total samples: {num_samples:,}")
    print(f"  Train samples: {split_idx:,} ({train_split*100:.0f}%)")
    print(f"  Val samples: {num_samples - split_idx:,} ({(1-train_split)*100:.0f}%)")
    print()
    
    # Split data
    # Note: For time-series, we typically use temporal split (not random)
    # Early data = training, later data = validation
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Create datasets
    train_dataset = RULDataset(X_train, y_train, add_channel_dim=add_channel_dim)
    val_dataset = RULDataset(X_val, y_val, add_channel_dim=add_channel_dim)
    
    print()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Gather info
    info = {
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'batch_size': batch_size,
        'num_train_batches': len(train_loader),
        'num_val_batches': len(val_loader),
        'train_stats': train_dataset.get_statistics(),
        'val_stats': val_dataset.get_statistics(),
    }
    
    print(f"✓ DataLoaders created successfully")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    return train_loader, val_loader, info


def visualize_batch(
    dataloader: DataLoader,
    num_samples: int = 4
):
    """
    Visualize a batch of samples from the DataLoader.
    
    Args:
        dataloader: PyTorch DataLoader
        num_samples: Number of samples to visualize
    """
    # Get one batch
    batch_sequences, batch_labels = next(iter(dataloader))
    
    # Limit to num_samples
    num_samples = min(num_samples, len(batch_sequences))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    
    for i in range(num_samples):
        sequence = batch_sequences[i]
        rul = batch_labels[i].item()
        
        # Remove channel dimension if present
        if sequence.dim() == 2:
            sequence = sequence.squeeze(0)
        
        sequence = sequence.numpy()
        time = np.arange(len(sequence)) / 20000
        
        axes[i].plot(time, sequence, linewidth=0.7, color='#2E86AB')
        axes[i].set_title(f'Sample {i+1} | RUL: {rul:.4f}', fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Time (s)', fontsize=10)
        axes[i].set_ylabel('Amplitude', fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Batch Samples from DataLoader', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def main():
    """Main execution function for demonstration."""
    
    print("=" * 70)
    print("SECTION 4: DATASET & DATALOADER CREATION")
    print("=" * 70)
    print()
    
    # Load sequences from Section 3
    try:
        from section1_load_data import IMSDataLoader
        from section2_preprocessing import VibrationPreprocessor
        from section3_sequence_gen import SequenceGenerator
        from pathlib import Path
        
        # Load and preprocess data
        dataset_path = Path(__file__).parent.parent / "IMS"
        loader = IMSDataLoader(dataset_path)
        
        print("Loading data...")
        signal, _ = loader.load_test_data(
            test_name='1st_test',
            bearing_column=0,
            max_files=50
        )
        
        print("\nPreprocessing...")
        preprocessor = VibrationPreprocessor(normalization_method='standard')
        signal, _ = preprocessor.preprocess(signal, fit_scaler=True)
        
        print("\nGenerating sequences...")
        generator = SequenceGenerator(window_size=2048, stride=512, normalize_rul=True)
        X, y = generator.generate_sequences(signal, verbose=True)
        print()
        
    except Exception as e:
        print(f"Could not load data: {e}")
        print("Creating synthetic data for demonstration...")
        np.random.seed(42)
        X = np.random.randn(1000, 2048).astype(np.float32)
        y = np.linspace(1.0, 0.0, 1000).astype(np.float32)
    
    # Create dataloaders
    print()
    train_loader, val_loader, info = create_dataloaders(
        X, y,
        train_split=0.8,
        batch_size=32,
        shuffle_train=True,
        add_channel_dim=True,
        random_seed=42
    )
    
    print()
    print("DataLoader Information:")
    print("-" * 70)
    print(f"Training set size:       {info['train_size']:,}")
    print(f"Validation set size:     {info['val_size']:,}")
    print(f"Batch size:              {info['batch_size']}")
    print(f"Training batches:        {info['num_train_batches']}")
    print(f"Validation batches:      {info['num_val_batches']}")
    print()
    print("Training Set Statistics:")
    for key, value in info['train_stats'].items():
        if isinstance(value, float):
            print(f"  {key:<25s}: {value:.6f}")
        else:
            print(f"  {key:<25s}: {value}")
    print("-" * 70)
    
    print()
    print("=" * 70)
    
    # Test batch loading
    print("\nTesting batch loading...")
    batch_x, batch_y = next(iter(train_loader))
    print(f"✓ Batch loaded successfully")
    print(f"  Batch X shape: {batch_x.shape}")
    print(f"  Batch y shape: {batch_y.shape}")
    print(f"  Device: {batch_x.device}")
    
    # Visualize
    print("\nGenerating batch visualization...")
    visualize_batch(train_loader, num_samples=4)
    
    return train_loader, val_loader, info


if __name__ == "__main__":
    train_loader, val_loader, info = main()
