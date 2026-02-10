"""
SECTION 10 — VISUALIZATION MODULE

This module provides comprehensive visualization tools for RUL prediction
results, degradation trends, and model performance.

Author: Mamba-RUL Project
Date: 2026-02-10
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class RULVisualizer:
    """
    Comprehensive visualization tools for RUL prediction.
    """
    
    def __init__(self, save_dir: str = 'visualizations'):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
    
    def plot_rul_prediction(
        self,
        true_rul: np.ndarray,
        pred_rul: np.ndarray,
        title: str = 'RUL Prediction',
        save_name: Optional[str] = None
    ):
        """
        Plot true vs predicted RUL.
        
        Args:
            true_rul: True RUL values
            pred_rul: Predicted RUL values
            title: Plot title
            save_name: Filename to save (None = don't save)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        axes[0].scatter(true_rul, pred_rul, alpha=0.5, s=20, edgecolor='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(true_rul.min(), pred_rul.min())
        max_val = max(true_rul.max(), pred_rul.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        axes[0].set_xlabel('True RUL', fontsize=12)
        axes[0].set_ylabel('Predicted RUL', fontsize=12)
        axes[0].set_title(f'{title}: Scatter Plot', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Time series
        indices = np.arange(len(true_rul))
        axes[1].plot(indices, true_rul, label='True RUL', alpha=0.8, linewidth=1.5)
        axes[1].plot(indices, pred_rul, label='Predicted RUL', alpha=0.8, linewidth=1.5)
        axes[1].fill_between(indices, true_rul, pred_rul, alpha=0.2)
        
        axes[1].set_xlabel('Sample Index', fontsize=12)
        axes[1].set_ylabel('RUL', fontsize=12)
        axes[1].set_title(f'{title}: Time Series', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, bbox_inches='tight')
            print(f"✓ Saved to {self.save_dir / save_name}")
        
        plt.show()
    
    def plot_degradation_trajectory(
        self,
        rul_values: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        title: str = 'Degradation Trajectory',
        save_name: Optional[str] = None
    ):
        """
        Plot degradation trajectory over time.
        
        Args:
            rul_values: RUL values over time
            timestamps: Optional timestamps
            title: Plot title
            save_name: Filename to save
        """
        if timestamps is None:
            timestamps = np.arange(len(rul_values))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot RUL trajectory
        ax.plot(timestamps, rul_values, linewidth=2, color='#2E86AB')
        ax.fill_between(timestamps, 0, rul_values, alpha=0.3, color='#2E86AB')
        
        # Add failure threshold
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Failure Threshold')
        
        ax.set_xlabel('Time / Sample Index', fontsize=12)
        ax.set_ylabel('Remaining Useful Life', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, bbox_inches='tight')
            print(f"✓ Saved to {self.save_dir / save_name}")
        
        plt.show()
    
    def plot_training_history(
        self,
        history: dict,
        metrics: List[str] = ['loss', 'mae', 'rmse'],
        save_name: Optional[str] = None
    ):
        """
        Plot training and validation curves.
        
        Args:
            history: Training history dictionary
            metrics: Metrics to plot
            save_name: Filename to save
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        epochs = range(1, len(history.get('train_loss', [])) + 1)
        
        for idx, metric in enumerate(metrics):
            train_key = f'train_{metric}'
            val_key = f'val_{metric}'
            
            if train_key in history:
                axes[idx].plot(epochs, history[train_key], label='Train', marker='o', markersize=3)
            if val_key in history:
                axes[idx].plot(epochs, history[val_key], label='Validation', marker='s', markersize=3)
            
            axes[idx].set_xlabel('Epoch', fontsize=11)
            axes[idx].set_ylabel(metric.upper(), fontsize=11)
            axes[idx].set_title(f'{metric.upper()} Over Epochs', fontsize=12, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, bbox_inches='tight')
            print(f"✓ Saved to {self.save_dir / save_name}")
        
        plt.show()
    
    def plot_error_distribution(
        self,
        errors: np.ndarray,
        title: str = 'Prediction Error Distribution',
        save_name: Optional[str] = None
    ):
        """
        Plot error distribution histogram.
        
        Args:
            errors: Prediction errors (pred - true)
            title: Plot title
            save_name: Filename to save
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[0].axvline(np.mean(errors), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
        axes[0].set_xlabel('Prediction Error', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'{title}', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(errors, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
        axes[1].set_ylabel('Prediction Error', fontsize=12)
        axes[1].set_title('Error Distribution (Box Plot)', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        stats_text = f"Mean: {np.mean(errors):.4f}\nStd: {np.std(errors):.4f}\nMedian: {np.median(errors):.4f}"
        axes[1].text(1.15, np.median(errors), stats_text, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, bbox_inches='tight')
            print(f"✓ Saved to {self.save_dir / save_name}")
        
        plt.show()
    
    def plot_residuals(
        self,
        true_rul: np.ndarray,
        pred_rul: np.ndarray,
        save_name: Optional[str] = None
    ):
        """
        Plot residual analysis.
        
        Args:
            true_rul: True RUL values
            pred_rul: Predicted RUL values
            save_name: Filename to save
        """
        residuals = pred_rul - true_rul
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(pred_rul, residuals, alpha=0.5, s=20)
        axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Predicted RUL')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted Values')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals vs True
        axes[0, 1].scatter(true_rul, residuals, alpha=0.5, s=20)
        axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('True RUL')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs True Values')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals over time
        axes[1, 0].plot(residuals, alpha=0.7, linewidth=1)
        axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residuals Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, bbox_inches='tight')
            print(f"✓ Saved to {self.save_dir / save_name}")
        
        plt.show()
    
    def create_report_figures(
        self,
        true_rul: np.ndarray,
        pred_rul: np.ndarray,
        history: Optional[dict] = None
    ):
        """
        Create all figures for research report.
        
        Args:
            true_rul: True RUL values
            pred_rul: Predicted RUL values
            history: Training history (optional)
        """
        print("Generating report figures...")
        
        # Main prediction plot
        self.plot_rul_prediction(true_rul, pred_rul, save_name='fig1_rul_prediction.png')
        
        # Error distribution
        errors = pred_rul - true_rul
        self.plot_error_distribution(errors, save_name='fig2_error_distribution.png')
        
        # Residual analysis
        self.plot_residuals(true_rul, pred_rul, save_name='fig3_residual_analysis.png')
        
        # Degradation trajectory
        self.plot_degradation_trajectory(true_rul, save_name='fig4_degradation_trajectory.png')
        
        # Training history
        if history:
            self.plot_training_history(history, save_name='fig5_training_history.png')
        
        print(f"\n✓ All report figures saved to {self.save_dir}/")


def main():
    """Demo visualization module."""
    print("Section 10: Visualization Module")
    print("\nThis module provides comprehensive visualization tools.")
    print("\nGenerating sample visualizations...")
    
    # Create sample data
    np.random.seed(42)
    true_rul = np.linspace(1.0, 0.0, 1000)
    pred_rul = true_rul + np.random.normal(0, 0.05, 1000)
    
    # Initialize visualizer
    viz = RULVisualizer(save_dir='sample_visualizations')
    
    # Create sample plots
    viz.plot_rul_prediction(true_rul, pred_rul, save_name='sample_prediction.png')
    viz.plot_error_distribution(pred_rul - true_rul, save_name='sample_errors.png')
    
    print("\n✓ Sample visualizations created!")


if __name__ == "__main__":
    main()
