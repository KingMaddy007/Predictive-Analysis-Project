"""
SECTION 11-12 — UTILITIES: ADVANCED IMPROVEMENTS & REPRODUCIBILITY

This module provides utilities for:
- Reproducibility (random seeds, configuration management)
- Advanced data augmentation
- Frequency-domain transformations
- Experiment tracking

Author: Mamba-RUL Project
Date: 2026-02-10
"""

import torch
import numpy as np
import random
import json
import yaml
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import hashlib


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"✓ Random seed set to {seed} for reproducibility")


class ExperimentConfig:
    """
    Manage experiment configuration for reproducibility.
    """
    
    def __init__(self, config_dict: Optional[Dict] = None):
        """
        Initialize configuration.
        
        Args:
            config_dict: Configuration dictionary
        """
        self.config = config_dict or {}
        self.config['timestamp'] = datetime.now().isoformat()
        self.config['config_hash'] = None
    
    def add(self, key: str, value):
        """Add a configuration parameter."""
        self.config[key] = value
    
    def get(self, key: str, default=None):
        """Get a configuration parameter."""
        return self.config.get(key, default)
    
    def compute_hash(self) -> str:
        """Compute hash of configuration for versioning."""
        # Create a stable string representation
        config_str = json.dumps(self.config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        self.config['config_hash'] = config_hash
        return config_hash
    
    def save(self, filepath: str):
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save configuration
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(exist_ok=True, parents=True)
        
        # Compute hash before saving
        self.compute_hash()
        
        # Save as JSON
        if filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(self.config, f, indent=2)
        # Save as YAML
        elif filepath.suffix in ['.yaml', '.yml']:
            with open(filepath, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        print(f"✓ Configuration saved to {filepath}")
        print(f"  Config hash: {self.config['config_hash']}")
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load configuration from file.
        
        Args:
            filepath: Path to configuration file
        
        Returns:
            ExperimentConfig instance
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        elif filepath.suffix in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        print(f"✓ Configuration loaded from {filepath}")
        return cls(config_dict)


class DataAugmentation:
    """
    Data augmentation techniques for time-series vibration data.
    """
    
    @staticmethod
    def add_noise(signal: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """
        Add Gaussian noise to signal.
        
        Args:
            signal: Input signal
            noise_level: Standard deviation of noise
        
        Returns:
            Noisy signal
        """
        noise = np.random.normal(0, noise_level, signal.shape)
        return signal + noise
    
    @staticmethod
    def time_warp(signal: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """
        Apply time warping to signal.
        
        Args:
            signal: Input signal
            sigma: Warping strength
        
        Returns:
            Warped signal
        """
        from scipy.interpolate import interp1d
        
        orig_steps = np.arange(len(signal))
        
        # Create random warping curve
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(len(signal),))
        warp_steps = np.cumsum(random_warps)
        warp_steps = warp_steps / warp_steps[-1] * (len(signal) - 1)
        
        # Interpolate
        f = interp1d(orig_steps, signal, kind='linear', fill_value='extrapolate')
        warped_signal = f(warp_steps)
        
        return warped_signal
    
    @staticmethod
    def magnitude_scale(signal: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """
        Scale signal magnitude.
        
        Args:
            signal: Input signal
            sigma: Scaling factor standard deviation
        
        Returns:
            Scaled signal
        """
        scale = np.random.normal(loc=1.0, scale=sigma)
        return signal * scale
    
    @staticmethod
    def time_shift(signal: np.ndarray, max_shift: int = 100) -> np.ndarray:
        """
        Shift signal in time.
        
        Args:
            signal: Input signal
            max_shift: Maximum shift in samples
        
        Returns:
            Shifted signal
        """
        shift = np.random.randint(-max_shift, max_shift)
        return np.roll(signal, shift)


class FrequencyTransforms:
    """
    Frequency-domain transformations for vibration signals.
    """
    
    @staticmethod
    def fft_transform(signal: np.ndarray, sampling_rate: float = 20000) -> tuple:
        """
        Compute FFT of signal.
        
        Args:
            signal: Time-domain signal
            sampling_rate: Sampling rate in Hz
        
        Returns:
            Tuple of (frequencies, magnitudes)
        """
        n = len(signal)
        fft_values = np.fft.fft(signal)
        fft_magnitudes = np.abs(fft_values)[:n//2]
        frequencies = np.fft.fftfreq(n, 1/sampling_rate)[:n//2]
        
        return frequencies, fft_magnitudes
    
    @staticmethod
    def power_spectrum(signal: np.ndarray, sampling_rate: float = 20000) -> tuple:
        """
        Compute power spectral density.
        
        Args:
            signal: Time-domain signal
            sampling_rate: Sampling rate in Hz
        
        Returns:
            Tuple of (frequencies, power)
        """
        from scipy import signal as scipy_signal
        
        frequencies, power = scipy_signal.welch(signal, fs=sampling_rate)
        return frequencies, power
    
    @staticmethod
    def wavelet_transform(signal: np.ndarray, wavelet: str = 'db4', level: int = 5):
        """
        Compute wavelet transform.
        
        Args:
            signal: Time-domain signal
            wavelet: Wavelet type
            level: Decomposition level
        
        Returns:
            Wavelet coefficients
        """
        import pywt
        
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        return coeffs


class ExperimentLogger:
    """
    Log experiment details for reproducibility.
    """
    
    def __init__(self, log_dir: str = 'experiments'):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Create experiment ID
        self.experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = self.log_dir / self.experiment_id
        self.experiment_dir.mkdir(exist_ok=True)
        
        self.log_file = self.experiment_dir / 'experiment.log'
        
        print(f"✓ Experiment logger initialized")
        print(f"  Experiment ID: {self.experiment_id}")
        print(f"  Log directory: {self.experiment_dir}")
    
    def log(self, message: str):
        """
        Log a message.
        
        Args:
            message: Message to log
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        
        print(log_entry.strip())
    
    def save_config(self, config: ExperimentConfig):
        """Save experiment configuration."""
        config.save(self.experiment_dir / 'config.json')
    
    def save_results(self, results: Dict, filename: str = 'results.json'):
        """
        Save experiment results.
        
        Args:
            results: Results dictionary
            filename: Filename to save
        """
        filepath = self.experiment_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved to {filepath}")


def create_default_config() -> ExperimentConfig:
    """
    Create default experiment configuration.
    
    Returns:
        Default configuration
    """
    config = ExperimentConfig({
        # Data
        'dataset': 'NASA_IMS',
        'test_name': '1st_test',
        'bearing_column': 0,
        'max_files': None,
        
        # Preprocessing
        'normalization_method': 'standard',
        
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
        'model_type': 'mamba',
        'input_dim': 1,
        'd_model': 128,
        'd_state': 16,
        'd_conv': 4,
        'expand': 2,
        'num_layers': 4,
        'dropout': 0.1,
        
        # Training
        'num_epochs': 100,
        'optimizer': 'adam',
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'scheduler': 'plateau',
        'grad_clip': 1.0,
        'early_stopping_patience': 20,
        
        # Reproducibility
        'random_seed': 42,
        
        # Hardware
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })
    
    return config


def main():
    """Demo utilities module."""
    print("Section 11-12: Advanced Utilities & Reproducibility")
    print()
    
    # Set seed
    set_seed(42)
    
    # Create config
    config = create_default_config()
    config.save('sample_config.json')
    
    # Load config
    loaded_config = ExperimentConfig.load('sample_config.json')
    print(f"\nLoaded config hash: {loaded_config.config['config_hash']}")
    
    # Demo augmentation
    print("\nTesting data augmentation...")
    signal = np.random.randn(1000)
    
    aug = DataAugmentation()
    noisy = aug.add_noise(signal, 0.01)
    scaled = aug.magnitude_scale(signal, 0.1)
    
    print("✓ Augmentation working")
    
    # Demo logger
    print("\nTesting experiment logger...")
    logger = ExperimentLogger(log_dir='sample_experiments')
    logger.log("Experiment started")
    logger.save_config(config)
    
    print("\n✓ All utilities working correctly!")


if __name__ == "__main__":
    main()
