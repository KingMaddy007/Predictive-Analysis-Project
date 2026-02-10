"""
COMPLETE TRAINING SCRIPT — MAMBA-RUL

This script implements the complete training pipeline for the Mamba-RUL model,
including data loading, preprocessing, training, evaluation, and comparison with baselines.

Usage:
    python train_complete.py --config config.json

Author: Mamba-RUL Project
Date: 2026-02-10
"""

import sys
from pathlib import Path
import argparse
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src import *


def train_mamba_model(config: ExperimentConfig):
    """
    Train the Mamba-RUL model.
    
    Args:
        config: Experiment configuration
    
    Returns:
        Trained model and results
    """
    print("\n" + "="*80)
    print("TRAINING MAMBA-RUL MODEL")
    print("="*80 + "\n")
    
    # Set seed for reproducibility
    set_seed(config.get('random_seed', 42))
    
    # 1. Load Data
    # 1. Load Data
    print("Step 1: Loading Data (2nd_test only)...")
    loader = IMSDataLoader(dataset_path=config.get('dataset_path', 'IMS'))
    signal, files = loader.load_test_data(
        test_name='2nd_test',
        bearing_column=0,  # Bearing 1 failed in 2nd_test
        max_files=config.get('max_files', None)
    )
    print(f"✓ Loaded {len(signal):,} samples from {len(files)} files\n")
    
    # 2. Preprocess
    print("Step 2: Preprocessing...")
    preprocessor = VibrationPreprocessor(normalization_method=config.get('normalization_method', 'standard'))
    signal, stats = preprocessor.preprocess(signal, fit_scaler=True)
    print(f"✓ Preprocessed signal (method: {config.get('normalization_method')})\n")
    
    # 3. Generate Sequences
    print("Step 3: Generating Sequences...")
    seq_gen = SequenceGenerator(
        window_size=config.get('window_size', 2048),
        stride=config.get('stride', 512),
        normalize_rul=config.get('normalize_rul', True)
    )
    X, y = seq_gen.generate_sequences(signal, verbose=True)
    print(f"✓ Generated {len(X):,} sequences\n")
    
    # 4. Create DataLoaders
    print("Step 4: Creating DataLoaders...")
    train_loader, val_loader, info = create_dataloaders(
        X, y,
        train_split=config.get('train_split', 0.8),
        batch_size=config.get('batch_size', 32),
        shuffle_train=config.get('shuffle_train', True),
        num_workers=config.get('num_workers', 0)
    )
    print(f"✓ Train: {info['train_size']} samples, Val: {info['val_size']} samples\n")
    
    # 5. Initialize Model
    print("Step 5: Initializing Mamba-RUL Model...")
    model = MambaRULModel(
        input_dim=config.get('input_dim', 1),
        d_model=config.get('d_model', 128),
        d_state=config.get('d_state', 16),
        d_conv=config.get('d_conv', 4),
        expand=config.get('expand', 2),
        num_layers=config.get('num_layers', 4),
        dropout=config.get('dropout', 0.1),
        use_mamba=True
    )
    
    # ---------------------------------------------------------
    # TRANSFER LEARNING: Load Pre-trained Weights
    # ---------------------------------------------------------
    previous_checkpoint = "checkpoints/mamba_e3b1b2e4/best_model.pth"
    if Path(previous_checkpoint).exists():
        print(f"\n🔄 TRANSFER LEARNING ENABLED!")
        print(f"  Loading pre-trained weights from: {previous_checkpoint}")
        try:
            checkpoint = torch.load(previous_checkpoint, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("  ✓ Weights loaded successfully! Fine-tuning on 2nd_test...")
        except Exception as e:
            print(f"  ⚠️ Warning: Could not load weights: {e}")
            print("  Starting from scratch instead.")
    else:
        print(f"\n⚠️ Pre-trained checkpoint not found at: {previous_checkpoint}")
        print("  Starting from scratch.")
    # ---------------------------------------------------------
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized ({total_params:,} parameters)\n")
    
    # 6. Train
    print("Step 6: Training...")
    trainer = RULTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.config,
        checkpoint_dir=f"checkpoints/mamba_{config.config.get('config_hash', 'default')}",
        device=config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    trainer.train(
        num_epochs=config.get('num_epochs', 50),
        early_stopping_patience=config.get('early_stopping_patience', 10)
    )
    
    # 7. Evaluate
    print("\nStep 7: Evaluating...")
    evaluator = RULEvaluator(model=model, device=trainer.device)
    
    val_results = evaluator.evaluate(val_loader, dataset_name='Validation')
    evaluator.save_results(val_results, save_dir=f"results/mamba_{config.config.get('config_hash', 'default')}")
    evaluator.plot_predictions(val_results, save_dir=f"results/mamba_{config.config.get('config_hash', 'default')}")
    
    return model, val_results, trainer.history


def train_baseline_models(config: ExperimentConfig, train_loader, val_loader):
    """
    Train baseline models for comparison.
    
    Args:
        config: Experiment configuration
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
    
    Returns:
        Dictionary of baseline results
    """
    print("\n" + "="*80)
    print("TRAINING BASELINE MODELS")
    print("="*80 + "\n")
    
    baseline_results = {}
    
    # LSTM Baseline
    print("Training LSTM Baseline...")
    lstm_model = LSTMBaseline(
        input_dim=config.get('input_dim', 1),
        hidden_dim=config.get('d_model', 128),
        num_layers=config.get('num_layers', 4),
        dropout=config.get('dropout', 0.1)
    )
    
    lstm_trainer = RULTrainer(
        model=lstm_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.config,
        checkpoint_dir=f"checkpoints/lstm_{config.config.get('config_hash', 'default')}",
        device=config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    lstm_trainer.train(
        num_epochs=config.get('num_epochs', 50),
        early_stopping_patience=config.get('early_stopping_patience', 10)
    )
    
    lstm_evaluator = RULEvaluator(model=lstm_model, device=lstm_trainer.device)
    lstm_results = lstm_evaluator.evaluate(val_loader, dataset_name='LSTM_Validation')
    lstm_evaluator.save_results(lstm_results, save_dir=f"results/lstm_{config.config.get('config_hash', 'default')}")
    
    baseline_results['LSTM'] = {
        'metrics': lstm_results['metrics'],
        'training_time': sum(lstm_trainer.history['epoch_time']),
        'num_parameters': sum(p.numel() for p in lstm_model.parameters())
    }
    
    # Transformer Baseline
    print("\nTraining Transformer Baseline...")
    transformer_model = TransformerBaseline(
        input_dim=config.get('input_dim', 1),
        d_model=config.get('d_model', 128),
        nhead=8,
        num_layers=config.get('num_layers', 4),
        dropout=config.get('dropout', 0.1)
    )
    
    transformer_trainer = RULTrainer(
        model=transformer_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.config,
        checkpoint_dir=f"checkpoints/transformer_{config.config.get('config_hash', 'default')}",
        device=config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    transformer_trainer.train(
        num_epochs=config.get('num_epochs', 50),
        early_stopping_patience=config.get('early_stopping_patience', 10)
    )
    
    transformer_evaluator = RULEvaluator(model=transformer_model, device=transformer_trainer.device)
    transformer_results = transformer_evaluator.evaluate(val_loader, dataset_name='Transformer_Validation')
    transformer_evaluator.save_results(transformer_results, save_dir=f"results/transformer_{config.config.get('config_hash', 'default')}")
    
    baseline_results['Transformer'] = {
        'metrics': transformer_results['metrics'],
        'training_time': sum(transformer_trainer.history['epoch_time']),
        'num_parameters': sum(p.numel() for p in transformer_model.parameters())
    }
    
    return baseline_results


def compare_models(mamba_results, baseline_results, config):
    """
    Compare Mamba with baseline models.
    
    Args:
        mamba_results: Mamba model results
        baseline_results: Baseline model results
        config: Experiment configuration
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80 + "\n")
    
    comparator = ModelComparator(results_dir=f"comparison_{config.config.get('config_hash', 'default')}")
    
    # Add Mamba results
    comparator.add_model_results(
        'Mamba-RUL',
        metrics=mamba_results['metrics'],
        training_time=mamba_results.get('training_time'),
        num_parameters=mamba_results.get('num_parameters')
    )
    
    # Add baseline results
    for model_name, results in baseline_results.items():
        comparator.add_model_results(
            model_name,
            metrics=results['metrics'],
            training_time=results.get('training_time'),
            num_parameters=results.get('num_parameters')
        )
    
    # Generate comparison report
    comparator.generate_report()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Mamba-RUL model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--train-baselines', action='store_true', help='Also train baseline models')
    args = parser.parse_args()
    
    # Load or create config
    if args.config and Path(args.config).exists():
        config = ExperimentConfig.load(args.config)
    else:
        config = create_default_config()
        config.save('default_config.json')
    
    # Initialize experiment logger
    logger = ExperimentLogger(log_dir='experiments')
    logger.save_config(config)
    logger.log("Experiment started")
    
    # Train Mamba model
    mamba_model, mamba_val_results, mamba_history = train_mamba_model(config)
    
    mamba_results = {
        'metrics': mamba_val_results['metrics'],
        'training_time': sum(mamba_history['epoch_time']),
        'num_parameters': sum(p.numel() for p in mamba_model.parameters())
    }
    
    logger.log(f"Mamba training complete. Val RMSE: {mamba_val_results['metrics']['RMSE']:.4f}")
    
    # Train baselines if requested
    if args.train_baselines:
        # Recreate dataset for baselines (2nd_test only)
        print("\nUsing 2nd_test dataset for baselines...")
        loader = IMSDataLoader(dataset_path=config.get('dataset_path', 'IMS'))
        signal, _ = loader.load_test_data(
            test_name='2nd_test',
            bearing_column=0,  # Bearing 1 failed in 2nd_test
            max_files=config.get('max_files', None)
        )
        
        preprocessor = VibrationPreprocessor(normalization_method=config.get('normalization_method', 'standard'))
        signal, _ = preprocessor.preprocess(signal, fit_scaler=True)
        
        seq_gen = SequenceGenerator(
            window_size=config.get('window_size', 2048),
            stride=config.get('stride', 512)
        )
        X, y = seq_gen.generate_sequences(signal, verbose=False)
        
        train_loader, val_loader, _ = create_dataloaders(
            X, y,
            train_split=config.get('train_split', 0.8),
            batch_size=config.get('batch_size', 32)
        )
        
        baseline_results = train_baseline_models(config, train_loader, val_loader)
        logger.log("Baseline training complete")
        
        # Compare models
        compare_models(mamba_results, baseline_results, config)
        logger.log("Comparison complete")
    
    logger.log("Experiment finished successfully")
    
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: experiments/{logger.experiment_id}/")
    print(f"Checkpoints saved to: checkpoints/")
    print(f"Visualizations saved to: results/")


if __name__ == "__main__":
    main()
