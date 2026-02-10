"""
SECTION 9 — PERFORMANCE COMPARISON AND ANALYSIS

This module provides tools for comparing multiple models and analyzing
their performance across various metrics.

Author: Mamba-RUL Project
Date: 2026-02-10
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import json
import time


class ModelComparator:
    """
    Compare performance of multiple RUL prediction models.
    """
    
    def __init__(self, results_dir: str = 'comparison_results'):
        """
        Initialize comparator.
        
        Args:
            results_dir: Directory to save comparison results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        self.model_results = {}
    
    def add_model_results(
        self,
        model_name: str,
        metrics: Dict[str, float],
        training_time: float = None,
        inference_time: float = None,
        memory_usage: float = None,
        num_parameters: int = None
    ):
        """
        Add results for a model.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of evaluation metrics
            training_time: Training time in seconds
            inference_time: Inference time in seconds
            memory_usage: Memory usage in MB
            num_parameters: Number of model parameters
        """
        self.model_results[model_name] = {
            'metrics': metrics,
            'training_time': training_time,
            'inference_time': inference_time,
            'memory_usage': memory_usage,
            'num_parameters': num_parameters
        }
    
    def create_comparison_table(self) -> pd.DataFrame:
        """
        Create a comparison table of all models.
        
        Returns:
            DataFrame with comparison results
        """
        data = []
        
        for model_name, results in self.model_results.items():
            row = {'Model': model_name}
            
            # Add metrics
            for metric_name, value in results['metrics'].items():
                row[metric_name] = value
            
            # Add performance metrics
            if results['training_time']:
                row['Training_Time_s'] = results['training_time']
            if results['inference_time']:
                row['Inference_Time_s'] = results['inference_time']
            if results['memory_usage']:
                row['Memory_MB'] = results['memory_usage']
            if results['num_parameters']:
                row['Parameters'] = results['num_parameters']
            
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    def save_comparison_table(self, df: pd.DataFrame):
        """Save comparison table to CSV."""
        csv_path = self.results_dir / 'model_comparison.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved comparison table to {csv_path}")
    
    def plot_metric_comparison(
        self,
        metrics: List[str] = ['RMSE', 'MAE', 'R2'],
        save: bool = True
    ):
        """
        Plot comparison of metrics across models.
        
        Args:
            metrics: List of metrics to plot
            save: Whether to save the plot
        """
        df = self.create_comparison_table()
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in df.columns]
        
        if not available_metrics:
            print("⚠ No metrics available for plotting")
            return
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(available_metrics):
            # Sort by metric value
            df_sorted = df.sort_values(metric)
            
            # Create bar plot
            axes[idx].barh(df_sorted['Model'], df_sorted[metric], color='skyblue', edgecolor='black')
            axes[idx].set_xlabel(metric)
            axes[idx].set_title(f'{metric} Comparison')
            axes[idx].grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, v in enumerate(df_sorted[metric]):
                axes[idx].text(v, i, f' {v:.4f}', va='center')
        
        plt.tight_layout()
        
        if save:
            save_path = self.results_dir / 'metric_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved metric comparison to {save_path}")
        
        plt.close()
    
    def plot_performance_comparison(self, save: bool = True):
        """
        Plot training time and inference time comparison.
        
        Args:
            save: Whether to save the plot
        """
        df = self.create_comparison_table()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training time
        if 'Training_Time_s' in df.columns:
            df_sorted = df.sort_values('Training_Time_s')
            axes[0].barh(df_sorted['Model'], df_sorted['Training_Time_s'], color='coral', edgecolor='black')
            axes[0].set_xlabel('Training Time (seconds)')
            axes[0].set_title('Training Time Comparison')
            axes[0].grid(True, alpha=0.3, axis='x')
        
        # Inference time
        if 'Inference_Time_s' in df.columns:
            df_sorted = df.sort_values('Inference_Time_s')
            axes[1].barh(df_sorted['Model'], df_sorted['Inference_Time_s'], color='lightgreen', edgecolor='black')
            axes[1].set_xlabel('Inference Time (seconds)')
            axes[1].set_title('Inference Time Comparison')
            axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save:
            save_path = self.results_dir / 'performance_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved performance comparison to {save_path}")
        
        plt.close()
    
    def plot_efficiency_analysis(self, save: bool = True):
        """
        Plot efficiency analysis (accuracy vs speed).
        
        Args:
            save: Whether to save the plot
        """
        df = self.create_comparison_table()
        
        if 'RMSE' not in df.columns or 'Inference_Time_s' not in df.columns:
            print("⚠ Missing required columns for efficiency analysis")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot: RMSE vs Inference Time
        ax.scatter(df['Inference_Time_s'], df['RMSE'], s=200, alpha=0.6, edgecolor='black')
        
        # Annotate points
        for idx, row in df.iterrows():
            ax.annotate(
                row['Model'],
                (row['Inference_Time_s'], row['RMSE']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10
            )
        
        ax.set_xlabel('Inference Time (seconds)', fontsize=12)
        ax.set_ylabel('RMSE (lower is better)', fontsize=12)
        ax.set_title('Model Efficiency: Accuracy vs Speed', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add quadrant lines
        median_time = df['Inference_Time_s'].median()
        median_rmse = df['RMSE'].median()
        ax.axvline(median_time, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(median_rmse, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save:
            save_path = self.results_dir / 'efficiency_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved efficiency analysis to {save_path}")
        
        plt.close()
    
    def generate_report(self):
        """Generate a comprehensive comparison report."""
        print("\n" + "="*70)
        print("MODEL COMPARISON REPORT")
        print("="*70 + "\n")
        
        df = self.create_comparison_table()
        
        # Display table
        print("Comparison Table:")
        print("-"*70)
        print(df.to_string(index=False))
        print("-"*70 + "\n")
        
        # Find best models
        if 'RMSE' in df.columns:
            best_rmse = df.loc[df['RMSE'].idxmin()]
            print(f"Best RMSE: {best_rmse['Model']} ({best_rmse['RMSE']:.4f})")
        
        if 'MAE' in df.columns:
            best_mae = df.loc[df['MAE'].idxmin()]
            print(f"Best MAE: {best_mae['Model']} ({best_mae['MAE']:.4f})")
        
        if 'R2' in df.columns:
            best_r2 = df.loc[df['R2'].idxmax()]
            print(f"Best R²: {best_r2['Model']} ({best_r2['R2']:.4f})")
        
        if 'Inference_Time_s' in df.columns:
            fastest = df.loc[df['Inference_Time_s'].idxmin()]
            print(f"Fastest Inference: {fastest['Model']} ({fastest['Inference_Time_s']:.4f}s)")
        
        print("\n" + "="*70)
        
        # Save table
        self.save_comparison_table(df)
        
        # Generate plots
        self.plot_metric_comparison()
        self.plot_performance_comparison()
        self.plot_efficiency_analysis()
        
        # Save results as JSON
        json_path = self.results_dir / 'comparison_results.json'
        with open(json_path, 'w') as f:
            # Convert to serializable format
            serializable_results = {}
            for model_name, results in self.model_results.items():
                serializable_results[model_name] = {
                    k: v for k, v in results.items()
                }
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n✓ Complete comparison report saved to {self.results_dir}/")


def measure_inference_time(model, dataloader, device='cpu', num_runs=3):
    """
    Measure average inference time for a model.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for inference
        device: Device to run on
        num_runs: Number of runs to average
    
    Returns:
        Average inference time in seconds
    """
    import torch
    
    model.eval()
    model = model.to(device)
    
    times = []
    
    with torch.no_grad():
        for run in range(num_runs):
            start_time = time.time()
            
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(device)
                _ = model(batch_x)
            
            elapsed = time.time() - start_time
            times.append(elapsed)
    
    return np.mean(times)


def main():
    """Demo comparison module."""
    print("Section 9: Performance Comparison and Analysis")
    print("\nThis module provides tools for comparing multiple models.")
    print("\nExample usage:")
    print("""
    comparator = ModelComparator()
    
    # Add Mamba results
    comparator.add_model_results(
        'Mamba-RUL',
        metrics={'RMSE': 0.0234, 'MAE': 0.0189, 'R2': 0.9567},
        training_time=120.5,
        inference_time=2.3
    )
    
    # Add LSTM results
    comparator.add_model_results(
        'LSTM',
        metrics={'RMSE': 0.0312, 'MAE': 0.0256, 'R2': 0.9234},
        training_time=180.2,
        inference_time=3.1
    )
    
    # Generate report
    comparator.generate_report()
    """)


if __name__ == "__main__":
    main()
