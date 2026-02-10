"""
SECTION 2 — DATA PREPROCESSING

This module cleans and normalizes vibration data for neural network input.
It handles NaN values, converts to proper numerical format, and applies
normalization techniques.

Author: Mamba-RUL Project
Date: 2026-02-10
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, Optional


class VibrationPreprocessor:
    """
    Preprocessor for vibration signal data.
    
    Handles cleaning, normalization, and validation of time-series data
    for deep learning models.
    """
    
    def __init__(self, normalization_method: str = 'standard'):
        """
        Initialize the preprocessor.
        
        Args:
            normalization_method: 'standard' (z-score) or 'minmax' (0-1 scaling)
        """
        self.normalization_method = normalization_method
        
        if normalization_method == 'standard':
            self.scaler = StandardScaler()
        elif normalization_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {normalization_method}")
        
        self.is_fitted = False
    
    def remove_nan_values(self, signal: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Remove NaN and infinite values from signal.
        
        Args:
            signal: Input vibration signal
        
        Returns:
            Tuple of (cleaned signal, number of removed values)
        """
        original_length = len(signal)
        
        # Remove NaN and infinite values
        mask = np.isfinite(signal)
        cleaned_signal = signal[mask]
        
        removed_count = original_length - len(cleaned_signal)
        
        if removed_count > 0:
            print(f"⚠ Removed {removed_count} NaN/Inf values ({removed_count/original_length*100:.2f}%)")
        
        return cleaned_signal, removed_count
    
    def convert_to_float(self, signal: np.ndarray) -> np.ndarray:
        """
        Ensure signal is in float32 format for neural networks.
        
        Args:
            signal: Input signal
        
        Returns:
            Signal as float32 array
        """
        return signal.astype(np.float32)
    
    def normalize(self, signal: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize the signal using the specified method.
        
        Args:
            signal: Input signal
            fit: Whether to fit the scaler (True for training data)
        
        Returns:
            Normalized signal
        """
        # Reshape for sklearn (needs 2D array)
        signal_2d = signal.reshape(-1, 1)
        
        if fit:
            normalized = self.scaler.fit_transform(signal_2d)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler must be fitted before transform. Use fit=True first.")
            normalized = self.scaler.transform(signal_2d)
        
        # Reshape back to 1D
        return normalized.flatten()
    
    def preprocess(
        self, 
        signal: np.ndarray, 
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """
        Complete preprocessing pipeline.
        
        Args:
            signal: Raw vibration signal
            fit_scaler: Whether to fit the scaler (True for training data)
        
        Returns:
            Tuple of (preprocessed signal, statistics dictionary)
        """
        stats = {}
        
        print("Starting preprocessing pipeline...")
        print("-" * 70)
        
        # Step 1: Remove NaN values
        print("Step 1: Removing NaN/Inf values...")
        signal, removed = self.remove_nan_values(signal)
        stats['removed_values'] = removed
        stats['length_after_cleaning'] = len(signal)
        
        # Step 2: Convert to float
        print("Step 2: Converting to float32...")
        signal = self.convert_to_float(signal)
        
        # Step 3: Store original statistics
        stats['original_mean'] = float(np.mean(signal))
        stats['original_std'] = float(np.std(signal))
        stats['original_min'] = float(np.min(signal))
        stats['original_max'] = float(np.max(signal))
        
        # Step 4: Normalize
        print(f"Step 3: Normalizing using {self.normalization_method} method...")
        signal = self.normalize(signal, fit=fit_scaler)
        
        # Step 5: Final statistics
        stats['normalized_mean'] = float(np.mean(signal))
        stats['normalized_std'] = float(np.std(signal))
        stats['normalized_min'] = float(np.min(signal))
        stats['normalized_max'] = float(np.max(signal))
        
        print("✓ Preprocessing complete!")
        print("-" * 70)
        
        return signal, stats
    
    def plot_comparison(
        self, 
        original: np.ndarray, 
        preprocessed: np.ndarray,
        sample_length: int = 5000
    ):
        """
        Visualize signal before and after preprocessing.
        
        Args:
            original: Original signal
            preprocessed: Preprocessed signal
            sample_length: Number of samples to plot
        """
        # Take samples
        orig_sample = original[:sample_length]
        prep_sample = preprocessed[:sample_length]
        time = np.arange(sample_length) / 20000  # 20kHz sampling
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Original signal
        axes[0].plot(time, orig_sample, linewidth=0.5, color='#A23B72', alpha=0.8)
        axes[0].set_title('Original Signal', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Time (seconds)', fontsize=11)
        axes[0].set_ylabel('Amplitude', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].text(
            0.02, 0.95, 
            f'Mean: {np.mean(orig_sample):.2f}, Std: {np.std(orig_sample):.2f}',
            transform=axes[0].transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        # Preprocessed signal
        axes[1].plot(time, prep_sample, linewidth=0.5, color='#2E86AB', alpha=0.8)
        axes[1].set_title('Preprocessed Signal (Normalized)', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Time (seconds)', fontsize=11)
        axes[1].set_ylabel('Normalized Amplitude', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].text(
            0.02, 0.95,
            f'Mean: {np.mean(prep_sample):.2f}, Std: {np.std(prep_sample):.2f}',
            transform=axes[1].transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        )
        
        plt.tight_layout()
        plt.show()


def main():
    """Main execution function for demonstration."""
    
    print("=" * 70)
    print("SECTION 2: DATA PREPROCESSING")
    print("=" * 70)
    print()
    
    # For demonstration, we'll create a sample signal
    # In practice, this would come from section1_load_data.py
    
    # Option 1: Load from Section 1
    try:
        from section1_load_data import IMSDataLoader
        from pathlib import Path
        
        dataset_path = Path(__file__).parent.parent / "IMS"
        loader = IMSDataLoader(dataset_path)
        
        print("Loading sample data from IMS dataset...")
        signal, _ = loader.load_test_data(
            test_name='1st_test',
            bearing_column=0,
            max_files=10  # Load only first 10 files for demo
        )
        print()
        
    except Exception as e:
        print(f"Could not load IMS data: {e}")
        print("Creating synthetic signal for demonstration...")
        # Create synthetic signal
        np.random.seed(42)
        signal = np.random.randn(100000) * 50 + 100
        signal = signal.astype(np.float64)
    
    # Store original for comparison
    original_signal = signal.copy()
    
    # Initialize preprocessor
    # Try 'standard' for z-score normalization or 'minmax' for 0-1 scaling
    preprocessor = VibrationPreprocessor(normalization_method='standard')
    
    # Preprocess the signal
    preprocessed_signal, stats = preprocessor.preprocess(signal, fit_scaler=True)
    
    # Display statistics
    print()
    print("Preprocessing Statistics:")
    print("-" * 70)
    print(f"{'Metric':<30s} {'Original':<15s} {'Preprocessed':<15s}")
    print("-" * 70)
    print(f"{'Mean':<30s} {stats['original_mean']:<15.6f} {stats['normalized_mean']:<15.6f}")
    print(f"{'Std Dev':<30s} {stats['original_std']:<15.6f} {stats['normalized_std']:<15.6f}")
    print(f"{'Min':<30s} {stats['original_min']:<15.6f} {stats['normalized_min']:<15.6f}")
    print(f"{'Max':<30s} {stats['original_max']:<15.6f} {stats['normalized_max']:<15.6f}")
    print("-" * 70)
    
    print()
    print("=" * 70)
    
    # Visualize
    preprocessor.plot_comparison(original_signal, preprocessed_signal, sample_length=5000)
    
    return preprocessed_signal, stats, preprocessor


if __name__ == "__main__":
    preprocessed_signal, stats, preprocessor = main()
