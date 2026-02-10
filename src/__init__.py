"""
Mamba-RUL: A Selective State Space Approach for Remaining Useful Life Estimation

This package contains all modules for the Mamba-RUL project.
"""

__version__ = "1.0.0"
__author__ = "Mamba-RUL Project"

# Import key classes for easy access
from .section0_setup import verify_environment
from .section1_load_data import IMSDataLoader
from .section2_preprocessing import VibrationPreprocessor
from .section3_sequence_gen import SequenceGenerator
from .section4_dataset import RULDataset, create_dataloaders
from .section5_mamba_model import MambaRULModel
from .section6_training import RULTrainer
from .section7_evaluation import RULEvaluator, load_model_from_checkpoint
from .section8_baselines import LSTMBaseline, TransformerBaseline, GRUBaseline
from .section9_comparison import ModelComparator, measure_inference_time
from .section10_visualization import RULVisualizer
from .section11_12_utils import (
    set_seed,
    ExperimentConfig,
    DataAugmentation,
    FrequencyTransforms,
    ExperimentLogger,
    create_default_config
)

__all__ = [
    # Section 0
    'verify_environment',
    # Section 1
    'IMSDataLoader',
    # Section 2
    'VibrationPreprocessor',
    # Section 3
    'SequenceGenerator',
    # Section 4
    'RULDataset',
    'create_dataloaders',
    # Section 5
    'MambaRULModel',
    # Section 6
    'RULTrainer',
    # Section 7
    'RULEvaluator',
    'load_model_from_checkpoint',
    # Section 8
    'LSTMBaseline',
    'TransformerBaseline',
    'GRUBaseline',
    # Section 9
    'ModelComparator',
    'measure_inference_time',
    # Section 10
    'RULVisualizer',
    # Section 11-12
    'set_seed',
    'ExperimentConfig',
    'DataAugmentation',
    'FrequencyTransforms',
    'ExperimentLogger',
    'create_default_config',
]
