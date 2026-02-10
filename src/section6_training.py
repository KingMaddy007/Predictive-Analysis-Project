"""
SECTION 6 — MODEL TRAINING PIPELINE

This module implements the complete training pipeline for the Mamba-RUL model
with support for GPU training, early stopping, and checkpoint management.

Author: Mamba-RUL Project
Date: 2026-02-10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import json
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt


class RULTrainer:
    """
    Comprehensive trainer for RUL prediction models.
    
    Features:
    - GPU/CPU support
    - Early stopping
    - Model checkpointing
    - Training history tracking
    - Progress visualization
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        checkpoint_dir: str = 'checkpoints',
        device: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The RUL prediction model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
            checkpoint_dir: Directory to save checkpoints
            device: Device to train on (auto-detect if None)
        """
        # Device setup
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Training on: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Loss functions
        self.criterion_mse = nn.MSELoss()
        self.criterion_mae = nn.L1Loss()
        
        # Optimizer
        optimizer_name = config.get('optimizer', 'adam').lower()
        lr = config.get('learning_rate', 1e-3)
        weight_decay = config.get('weight_decay', 1e-5)
        
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Learning rate scheduler
        scheduler_type = config.get('scheduler', 'plateau')
        if scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.get('num_epochs', 50)
            )
        else:
            self.scheduler = None
        
        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'train_rmse': [],
            'val_rmse': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Save config
        with open(self.checkpoint_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        total_rmse = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for batch_x, batch_y in pbar:
            # Move to device
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            predictions = self.model(batch_x)
            
            # Calculate losses
            loss_mse = self.criterion_mse(predictions, batch_y)
            loss_mae = self.criterion_mae(predictions, batch_y)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss_mse.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config['grad_clip']
                )
            
            # Update weights
            self.optimizer.step()
            
            # Calculate RMSE
            rmse = torch.sqrt(loss_mse)
            
            # Accumulate metrics
            total_loss += loss_mse.item()
            total_mae += loss_mae.item()
            total_rmse += rmse.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_mse.item():.4f}',
                'mae': f'{loss_mae.item():.4f}'
            })
        
        metrics = {
            'loss': total_loss / num_batches,
            'mae': total_mae / num_batches,
            'rmse': total_rmse / num_batches
        }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_rmse = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(self.val_loader, desc='Validation', leave=False):
                # Move to device
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                predictions = self.model(batch_x)
                
                # Calculate losses
                loss_mse = self.criterion_mse(predictions, batch_y)
                loss_mae = self.criterion_mae(predictions, batch_y)
                rmse = torch.sqrt(loss_mse)
                
                total_loss += loss_mse.item()
                total_mae += loss_mae.item()
                total_rmse += rmse.item()
                num_batches += 1
        
        metrics = {
            'loss': total_loss / num_batches,
            'mae': total_mae / num_batches,
            'rmse': total_rmse / num_batches
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: Optional[int] = None
    ):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Stop if no improvement for this many epochs
        """
        print("=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Optimizer: {self.config.get('optimizer', 'adam')}")
        print(f"Learning rate: {self.config.get('learning_rate', 1e-3)}")
        print()
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 80)
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['train_rmse'].append(train_metrics['rmse'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            self.history['learning_rate'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)
            
            # Print metrics
            print(f"\nResults:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f} | MAE: {train_metrics['mae']:.4f} | RMSE: {train_metrics['rmse']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f} | MAE: {val_metrics['mae']:.4f} | RMSE: {val_metrics['rmse']:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Epoch Time: {epoch_time:.2f}s")
            
            # Check for improvement
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if early_stopping_patience and self.epochs_without_improvement >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered (no improvement for {early_stopping_patience} epochs)")
                break
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Total training time: {total_time/60:.2f} minutes")
        print(f"Average epoch time: {np.mean(self.history['epoch_time']):.2f}s")
        
        # Save final history
        self.save_history()
        
        # Plot training curves
        self.plot_training_curves()
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Saved training history to {history_path}")
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train', marker='o', markersize=3)
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Validation', marker='s', markersize=3)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (MSE)')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE plot
        axes[0, 1].plot(epochs, self.history['train_mae'], label='Train', marker='o', markersize=3)
        axes[0, 1].plot(epochs, self.history['val_mae'], label='Validation', marker='s', markersize=3)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # RMSE plot
        axes[1, 0].plot(epochs, self.history['train_rmse'], label='Train', marker='o', markersize=3)
        axes[1, 0].plot(epochs, self.history['val_rmse'], label='Validation', marker='s', markersize=3)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].set_title('Root Mean Square Error')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1, 1].plot(epochs, self.history['learning_rate'], marker='o', markersize=3, color='green')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.checkpoint_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training curves to {save_path}")
        plt.close()


def main():
    """Demo training script."""
    print("Section 6: Model Training Pipeline")
    print("This module provides the RULTrainer class for training RUL models.")
    print("\nSee main_train.py for a complete training example.")


if __name__ == "__main__":
    main()
