"""
SECTION 3 — SEQUENCE GENERATION (WINDOWING)

This module converts long vibration signals into training samples using
sliding windows and generates RUL (Remaining Useful Life) labels.

The RUL is calculated as: RUL = Total_Length - Current_Window_End

Author: Mamba-RUL Project
Date: 2026-02-10
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from tqdm import tqdm


class SequenceGenerator:
    """
    Generator for creating windowed sequences with RUL labels.
    
    Converts a long time-series signal into fixed-length windows suitable
    for training deep learning models.
    """
    
    def __init__(
        self, 
        window_size: int = 2048,
        stride: int = 512,
        normalize_rul: bool = True
    ):
        """
        Initialize the sequence generator.
        
        Args:
            window_size: Length of each window (number of samples)
            stride: Step size between consecutive windows
            normalize_rul: Whether to normalize RUL values to [0, 1]
        """
        self.window_size = window_size
        self.stride = stride
        self.normalize_rul = normalize_rul
        self.max_rul = None
    
    def generate_sequences(
        self, 
        signal: np.ndarray,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate windowed sequences and corresponding RUL labels.
        
        The signal is assumed to represent a run-to-failure scenario where:
        - Beginning of signal = healthy bearing (high RUL)
        - End of signal = failed bearing (low RUL)
        
        Args:
            signal: Preprocessed vibration signal
            verbose: Whether to print progress information
        
        Returns:
            Tuple of (X, y) where:
                X: Array of shape (num_windows, window_size)
                y: Array of RUL values of shape (num_windows,)
        """
        signal_length = len(signal)
        
        if signal_length < self.window_size:
            raise ValueError(
                f"Signal length ({signal_length}) is shorter than window size ({self.window_size})"
            )
        
        # Calculate number of windows
        num_windows = (signal_length - self.window_size) // self.stride + 1
        
        if verbose:
            print(f"Generating sequences...")
            print(f"  Signal length: {signal_length:,} samples")
            print(f"  Window size: {self.window_size:,} samples")
            print(f"  Stride: {self.stride:,} samples")
            print(f"  Number of windows: {num_windows:,}")
        
        # Pre-allocate arrays
        X = np.zeros((num_windows, self.window_size), dtype=np.float32)
        y = np.zeros(num_windows, dtype=np.float32)
        
        # Generate windows
        iterator = range(num_windows)
        if verbose:
            iterator = tqdm(iterator, desc="Creating windows")
        
        for i in iterator:
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            
            # Extract window
            X[i] = signal[start_idx:end_idx]
            
            # Calculate RUL: remaining samples from end of this window to end of signal
            # RUL decreases as we approach failure
            remaining_samples = signal_length - end_idx
            y[i] = remaining_samples
        
        # Store max RUL for normalization
        self.max_rul = float(np.max(y))
        
        # Normalize RUL if requested
        if self.normalize_rul:
            y = y / self.max_rul
            if verbose:
                print(f"  ✓ RUL normalized to [0, 1] range (max RUL: {self.max_rul:,.0f})")
        
        if verbose:
            print(f"  ✓ Generated {num_windows:,} sequences")
            print(f"  ✓ X shape: {X.shape}")
            print(f"  ✓ y shape: {y.shape}")
            print(f"  ✓ RUL range: [{np.min(y):.4f}, {np.max(y):.4f}]")
        
        return X, y
    
    def visualize_windows(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_samples: int = 6
    ):
        """
        Visualize sample windows at different stages of degradation.
        
        Args:
            X: Window sequences
            y: RUL labels
            num_samples: Number of windows to visualize
        """
        # Select windows evenly spaced across the degradation timeline
        indices = np.linspace(0, len(X) - 1, num_samples, dtype=int)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx, ax_idx in enumerate(indices):
            window = X[ax_idx]
            rul = y[ax_idx]
            
            # Create time axis
            time = np.arange(len(window)) / 20000  # Convert to seconds
            
            # Plot
            axes[idx].plot(time, window, linewidth=0.7, color='#2E86AB')
            
            # Title with RUL information
            if self.normalize_rul:
                rul_display = f"{rul:.3f} (normalized)"
                actual_rul = rul * self.max_rul if self.max_rul else rul
                title = f"Window {ax_idx} | RUL: {rul_display}\nActual: {actual_rul:,.0f} samples"
            else:
                title = f"Window {ax_idx} | RUL: {rul:,.0f} samples"
            
            axes[idx].set_title(title, fontsize=10, fontweight='bold')
            axes[idx].set_xlabel('Time (s)', fontsize=9)
            axes[idx].set_ylabel('Amplitude', fontsize=9)
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle(
            'Sample Windows Across Degradation Timeline\n(Early Life → End of Life)',
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()
        plt.show()
    
    def plot_rul_distribution(self, y: np.ndarray):
        """
        Plot the distribution of RUL values.
        
        Args:
            y: RUL labels
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(y, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('RUL Value', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('RUL Distribution', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # RUL over time (degradation curve)
        axes[1].plot(y, linewidth=1.5, color='#A23B72')
        axes[1].set_xlabel('Window Index', fontsize=12)
        axes[1].set_ylabel('RUL Value', fontsize=12)
        axes[1].set_title('RUL Degradation Curve', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].fill_between(range(len(y)), y, alpha=0.3, color='#A23B72')
        
        plt.tight_layout()
        plt.show()


def main():
    """Main execution function for demonstration."""
    
    print("=" * 70)
    print("SECTION 3: SEQUENCE GENERATION (WINDOWING)")
    print("=" * 70)
    print()
    
    # Load preprocessed data from Section 2
    try:
        from section1_load_data import IMSDataLoader
        from section2_preprocessing import VibrationPreprocessor
        from pathlib import Path
        
        # Load data
        dataset_path = Path(__file__).parent.parent / "IMS"
        loader = IMSDataLoader(dataset_path)
        
        print("Loading and preprocessing data...")
        signal, _ = loader.load_test_data(
            test_name='1st_test',
            bearing_column=0,
            max_files=50  # Use first 50 files for demo
        )
        
        # Preprocess
        preprocessor = VibrationPreprocessor(normalization_method='standard')
        signal, _ = preprocessor.preprocess(signal, fit_scaler=True)
        print()
        
    except Exception as e:
        print(f"Could not load data: {e}")
        print("Creating synthetic signal for demonstration...")
        # Create synthetic degradation signal
        np.random.seed(42)
        signal = np.random.randn(500000).astype(np.float32)
    
    # Initialize sequence generator
    # Adjust parameters as needed:
    # - window_size: Length of each sequence (e.g., 2048, 4096, 8192)
    # - stride: Overlap between windows (smaller = more overlap)
    # - normalize_rul: Whether to scale RUL to [0, 1]
    
    generator = SequenceGenerator(
        window_size=2048,
        stride=512,
        normalize_rul=True
    )
    
    # Generate sequences
    print()
    X, y = generator.generate_sequences(signal, verbose=True)
    
    print()
    print("Dataset Summary:")
    print("-" * 70)
    print(f"Input sequences (X):     {X.shape}")
    print(f"RUL labels (y):          {y.shape}")
    print(f"Memory usage (X):        {X.nbytes / 1024**2:.2f} MB")
    print(f"Memory usage (y):        {y.nbytes / 1024**2:.2f} MB")
    print(f"Total memory:            {(X.nbytes + y.nbytes) / 1024**2:.2f} MB")
    print("-" * 70)
    
    print()
    print("=" * 70)
    
    # Visualizations
    print("\nGenerating visualizations...")
    generator.visualize_windows(X, y, num_samples=6)
    generator.plot_rul_distribution(y)
    
    return X, y, generator


if __name__ == "__main__":
    X, y, generator = main()
