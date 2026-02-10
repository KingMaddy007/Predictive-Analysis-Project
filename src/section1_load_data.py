"""
SECTION 1 — LOAD NASA IMS DATASET

This module loads vibration sensor data from the NASA IMS bearing dataset.
It reads multiple time-series files, maintains chronological order, and
combines them into a single continuous signal.

Dataset Structure:
    IMS/
    ├── 1st_test/
    ├── 2nd_test/
    └── 3rd_test/

Each test contains multiple files with vibration measurements from 4 bearings.

Author: Mamba-RUL Project
Date: 2026-02-10
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional
from tqdm import tqdm


class IMSDataLoader:
    """
    Loader for NASA IMS Bearing Dataset.
    
    The IMS dataset contains run-to-failure vibration data from bearings.
    Each file contains measurements from 4 bearing sensors at 20kHz sampling rate.
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the IMS data loader.
        
        Args:
            dataset_path: Path to the IMS dataset directory
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    def get_available_tests(self) -> List[str]:
        """
        Get list of available test directories.
        
        Returns:
            List of test directory names
        """
        tests = []
        for item in self.dataset_path.iterdir():
            if item.is_dir() and 'test' in item.name.lower():
                tests.append(item.name)
        return sorted(tests)
    
    def load_single_file(self, file_path: Path, bearing_column: int = 0) -> np.ndarray:
        """
        Load a single vibration data file.
        
        Args:
            file_path: Path to the data file
            bearing_column: Which bearing sensor to extract (0-3)
        
        Returns:
            NumPy array of vibration measurements
        """
        try:
            # IMS files are tab-separated with 4 columns (one per bearing)
            data = pd.read_csv(file_path, sep='\t', header=None)
            
            # Extract specified bearing column
            if bearing_column >= data.shape[1]:
                raise ValueError(f"Bearing column {bearing_column} not found. File has {data.shape[1]} columns.")
            
            vibration = data.iloc[:, bearing_column].values
            return vibration
        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return np.array([])
    
    def load_test_data(
        self, 
        test_name: str, 
        bearing_column: int = 0,
        max_files: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Load all vibration data from a specific test run.
        
        Args:
            test_name: Name of the test directory (e.g., '1st_test')
            bearing_column: Which bearing sensor to extract (0-3)
            max_files: Maximum number of files to load (None = all files)
        
        Returns:
            Tuple of:
                - Combined vibration signal as NumPy array
                - List of loaded file names
        """
        test_path = self.dataset_path / test_name
        
        if not test_path.exists():
            raise FileNotFoundError(f"Test directory not found: {test_path}")
        
        # Get all data files and sort chronologically
        data_files = sorted([f for f in test_path.iterdir() if f.is_file()])
        
        if max_files is not None:
            data_files = data_files[:max_files]
        
        print(f"Loading {len(data_files)} files from {test_name}...")
        print(f"Extracting bearing column: {bearing_column}")
        
        # Load all files
        all_vibrations = []
        loaded_files = []
        
        for file_path in tqdm(data_files, desc="Loading files"):
            vibration = self.load_single_file(file_path, bearing_column)
            
            if len(vibration) > 0:
                all_vibrations.append(vibration)
                loaded_files.append(file_path.name)
        
        # Combine into single continuous signal
        combined_signal = np.concatenate(all_vibrations)
        
        print(f"\n✓ Successfully loaded {len(loaded_files)} files")
        print(f"✓ Total signal length: {len(combined_signal):,} samples")
        print(f"✓ Duration: {len(combined_signal) / 20000:.2f} seconds (at 20kHz)")
        
        return combined_signal, loaded_files
    
    def plot_signal_sample(
        self, 
        signal: np.ndarray, 
        sample_length: int = 10000,
        title: str = "Vibration Signal Sample"
    ):
        """
        Plot a sample of the vibration signal.
        
        Args:
            signal: Vibration signal array
            sample_length: Number of samples to plot
            title: Plot title
        """
        sample = signal[:sample_length]
        time = np.arange(len(sample)) / 20000  # Convert to seconds (20kHz sampling)
        
        plt.figure(figsize=(14, 5))
        plt.plot(time, sample, linewidth=0.5, color='#2E86AB')
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Vibration Amplitude', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_signal_statistics(self, signal: np.ndarray) -> dict:
        """
        Calculate basic statistics of the vibration signal.
        
        Args:
            signal: Vibration signal array
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'length': len(signal),
            'mean': np.mean(signal),
            'std': np.std(signal),
            'min': np.min(signal),
            'max': np.max(signal),
            'median': np.median(signal),
        }
        return stats


def main():
    """Main execution function for demonstration."""
    
    # Set dataset path (adjust as needed)
    dataset_path = Path(__file__).parent.parent / "IMS"
    
    print("=" * 70)
    print("SECTION 1: NASA IMS DATASET LOADING")
    print("=" * 70)
    print()
    
    # Initialize loader
    loader = IMSDataLoader(dataset_path)
    
    # Show available tests
    tests = loader.get_available_tests()
    print(f"Available tests: {tests}")
    print()
    
    # Load data from first test, bearing 0
    # You can change these parameters:
    # - test_name: '1st_test', '2nd_test', '3rd_test'
    # - bearing_column: 0, 1, 2, 3
    # - max_files: None (all files) or a number to limit
    
    signal, files = loader.load_test_data(
        test_name='1st_test',
        bearing_column=0,
        max_files=None  # Load all files
    )
    
    print()
    
    # Display statistics
    stats = loader.get_signal_statistics(signal)
    print("Signal Statistics:")
    print("-" * 70)
    for key, value in stats.items():
        if key == 'length':
            print(f"{key:15s}: {value:,}")
        else:
            print(f"{key:15s}: {value:.6f}")
    
    print()
    print("=" * 70)
    
    # Plot sample
    loader.plot_signal_sample(signal, sample_length=10000)
    
    return signal, files


if __name__ == "__main__":
    signal, files = main()
