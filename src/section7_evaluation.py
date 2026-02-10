"""
SECTION 7 — MODEL VALIDATION AND TESTING

This module implements comprehensive model evaluation including
prediction generation, metric calculation, and result analysis.

Author: Mamba-RUL Project
Date: 2026-02-10
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import json


class RULEvaluator:
    """
    Evaluator for RUL prediction models.
    
    Provides comprehensive evaluation metrics and prediction analysis.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: The trained RUL model
            device: Device to run on (auto-detect if None)
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model = model.to(self.device)
        self.model.eval()
    
    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for a dataset.
        
        Args:
            dataloader: DataLoader for the dataset
        
        Returns:
            Tuple of (predictions, true_values)
        """
        all_predictions = []
        all_true_values = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in tqdm(dataloader, desc='Predicting'):
                batch_x = batch_x.to(self.device)
                
                # Forward pass
                predictions = self.model(batch_x)
                
                # Move to CPU and convert to numpy
                all_predictions.append(predictions.cpu().numpy())
                all_true_values.append(batch_y.numpy())
        
        predictions = np.concatenate(all_predictions)
        true_values = np.concatenate(all_true_values)
        
        return predictions, true_values
    
    def calculate_metrics(
        self,
        predictions: np.ndarray,
        true_values: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            predictions: Predicted RUL values
            true_values: True RUL values
        
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        mse = np.mean((predictions - true_values) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - true_values))
        
        # R² Score
        ss_res = np.sum((true_values - predictions) ** 2)
        ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Mean Absolute Percentage Error (MAPE)
        # Avoid division by zero
        mask = true_values != 0
        if np.any(mask):
            mape = np.mean(np.abs((true_values[mask] - predictions[mask]) / true_values[mask])) * 100
        else:
            mape = float('inf')
        
        # NASA Scoring Function
        # Penalizes late predictions more than early predictions
        errors = predictions - true_values
        nasa_score = self.nasa_scoring_function(errors)
        
        # Max error
        max_error = np.max(np.abs(errors))
        
        metrics = {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2),
            'MAPE': float(mape),
            'NASA_Score': float(nasa_score),
            'Max_Error': float(max_error)
        }
        
        return metrics
    
    @staticmethod
    def nasa_scoring_function(errors: np.ndarray) -> float:
        """
        NASA Prognostics scoring function.
        
        Penalizes late predictions (negative errors) more heavily.
        
        Args:
            errors: prediction - true (positive = early, negative = late)
        
        Returns:
            NASA score (lower is better)
        """
        scores = np.where(
            errors < 0,
            np.exp(-errors / 13) - 1,  # Late predictions (heavily penalized)
            np.exp(errors / 10) - 1     # Early predictions (lightly penalized)
        )
        return np.sum(scores)
    
    def evaluate(
        self,
        dataloader: DataLoader,
        dataset_name: str = 'Test'
    ) -> Dict[str, any]:
        """
        Complete evaluation pipeline.
        
        Args:
            dataloader: DataLoader for evaluation
            dataset_name: Name of the dataset (for display)
        
        Returns:
            Dictionary containing predictions, true values, and metrics
        """
        print(f"\n{'='*70}")
        print(f"EVALUATING ON {dataset_name.upper()} SET")
        print(f"{'='*70}\n")
        
        # Generate predictions
        predictions, true_values = self.predict(dataloader)
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, true_values)
        
        # Print metrics
        print("Evaluation Metrics:")
        print("-" * 70)
        for metric_name, value in metrics.items():
            if metric_name == 'MAPE':
                print(f"  {metric_name:<20s}: {value:.2f}%")
            else:
                print(f"  {metric_name:<20s}: {value:.6f}")
        print("-" * 70)
        
        results = {
            'predictions': predictions,
            'true_values': true_values,
            'metrics': metrics,
            'dataset_name': dataset_name
        }
        
        return results
    
    def save_results(
        self,
        results: Dict,
        save_dir: str = 'results'
    ):
        """
        Save evaluation results to disk.
        
        Args:
            results: Results dictionary from evaluate()
            save_dir: Directory to save results
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        dataset_name = results['dataset_name'].lower()
        
        # Save predictions and true values
        np.save(save_path / f'{dataset_name}_predictions.npy', results['predictions'])
        np.save(save_path / f'{dataset_name}_true_values.npy', results['true_values'])
        
        # Save metrics
        with open(save_path / f'{dataset_name}_metrics.json', 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        
        # Create comparison DataFrame
        df = pd.DataFrame({
            'True_RUL': results['true_values'],
            'Predicted_RUL': results['predictions'],
            'Error': results['predictions'] - results['true_values'],
            'Absolute_Error': np.abs(results['predictions'] - results['true_values'])
        })
        
        # Save to CSV
        df.to_csv(save_path / f'{dataset_name}_comparison.csv', index=False)
        
        print(f"\n✓ Results saved to {save_path}/")
        print(f"  - {dataset_name}_predictions.npy")
        print(f"  - {dataset_name}_true_values.npy")
        print(f"  - {dataset_name}_metrics.json")
        print(f"  - {dataset_name}_comparison.csv")
    
    def plot_predictions(
        self,
        results: Dict,
        save_dir: str = 'results',
        show: bool = False
    ):
        """
        Plot prediction vs true RUL.
        
        Args:
            results: Results dictionary from evaluate()
            save_dir: Directory to save plots
            show: Whether to display plots
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        predictions = results['predictions']
        true_values = results['true_values']
        dataset_name = results['dataset_name']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Scatter plot: Predicted vs True
        axes[0, 0].scatter(true_values, predictions, alpha=0.5, s=10)
        
        # Perfect prediction line
        min_val = min(true_values.min(), predictions.min())
        max_val = max(true_values.max(), predictions.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        axes[0, 0].set_xlabel('True RUL')
        axes[0, 0].set_ylabel('Predicted RUL')
        axes[0, 0].set_title(f'{dataset_name} Set: Predicted vs True RUL')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Time series plot
        indices = np.arange(len(predictions))
        axes[0, 1].plot(indices, true_values, label='True RUL', alpha=0.7, linewidth=1)
        axes[0, 1].plot(indices, predictions, label='Predicted RUL', alpha=0.7, linewidth=1)
        axes[0, 1].set_xlabel('Sample Index')
        axes[0, 1].set_ylabel('RUL')
        axes[0, 1].set_title('RUL Prediction Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Error distribution
        errors = predictions - true_values
        axes[1, 0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(0, color='r', linestyle='--', label='Zero Error')
        axes[1, 0].set_xlabel('Prediction Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Error Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Absolute error over time
        abs_errors = np.abs(errors)
        axes[1, 1].plot(indices, abs_errors, alpha=0.7, linewidth=1)
        axes[1, 1].axhline(np.mean(abs_errors), color='r', linestyle='--', label=f'Mean: {np.mean(abs_errors):.4f}')
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Absolute Error')
        axes[1, 1].set_title('Absolute Error Over Time')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        plot_path = save_path / f'{dataset_name.lower()}_predictions.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved prediction plot to {plot_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_class,
    model_config: Dict,
    device: Optional[str] = None
) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_class: Model class to instantiate
        model_config: Model configuration dictionary
        device: Device to load on
    
    Returns:
        Loaded model
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model
    model = model_class(**model_config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from {checkpoint_path}")
    if 'best_val_loss' in checkpoint:
        print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    return model


def main():
    """Demo evaluation script."""
    print("Section 7: Model Validation and Testing")
    print("This module provides the RULEvaluator class for model evaluation.")
    print("\nSee main_evaluate.py for a complete evaluation example.")


if __name__ == "__main__":
    main()
