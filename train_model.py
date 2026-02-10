"""
TRAINING SCRIPT TEMPLATE — MAMBA-RUL

This is a template for training the Mamba-RUL model.
Implement the training loop, validation, and model checkpointing here.

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
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Import project modules
import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from section1_load_data import IMSDataLoader
from section2_preprocessing import VibrationPreprocessor
from section3_sequence_gen import SequenceGenerator
from section4_dataset import create_dataloaders
from section5_mamba_model import MambaRULModel


class RULTrainer:
    """
    Trainer class for Mamba-RUL model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        checkpoint_dir: str = 'checkpoints'
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The Mamba-RUL model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate for optimizer
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function (Mean Squared Error for regression)
        self.criterion = nn.MSELoss()
        
        # Optimizer (Adam)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average loss, average MAE)
        """
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        
        for batch_x, batch_y in pbar:
            # Move to device
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            predictions = self.model(batch_x)
            
            # Calculate loss
            loss = self.criterion(predictions, batch_y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (optional but recommended)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Calculate MAE
            mae = torch.mean(torch.abs(predictions - batch_y))
            
            # Accumulate metrics
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mae': f'{mae.item():.4f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average loss, average MAE)
        """
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                predictions = self.model(batch_x)
                
                # Calculate metrics
                loss = self.criterion(predictions, batch_y)
                mae = torch.mean(torch.abs(predictions - batch_y))
                
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
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
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def train(self, num_epochs: int, early_stopping_patience: int = 10):
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
        print()
        
        epochs_without_improvement = 0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 80)
            
            # Train
            train_loss, train_mae = self.train_epoch()
            
            # Validate
            val_loss, val_mae = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['learning_rate'].append(current_lr)
            
            # Print metrics
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val MAE:   {val_mae:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered (no improvement for {early_stopping_patience} epochs)")
                break
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Plot training history
        self.plot_history()
    
    def plot_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        axes[0].plot(epochs, self.history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(epochs, self.history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE plot
        axes[1].plot(epochs, self.history['train_mae'], label='Train MAE', marker='o')
        axes[1].plot(epochs, self.history['val_mae'], label='Val MAE', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Training and Validation MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'training_history.png', dpi=150)
        plt.show()


def main():
    """
    Main training function.
    
    TODO: Implement the complete training pipeline:
    1. Load and preprocess data
    2. Create dataloaders
    3. Initialize model
    4. Create trainer
    5. Train model
    6. Evaluate on test set
    """
    
    print("=" * 80)
    print("MAMBA-RUL TRAINING SCRIPT")
    print("=" * 80)
    print()
    print("⚠ This is a template. Implement the training pipeline below.")
    print()
    
    # TODO: Add your training code here
    # Example:
    # 1. Load data using IMSDataLoader
    # 2. Preprocess using VibrationPreprocessor
    # 3. Generate sequences using SequenceGenerator
    # 4. Create dataloaders using create_dataloaders
    # 5. Initialize MambaRULModel
    # 6. Create RULTrainer
    # 7. Call trainer.train(num_epochs=50)
    
    print("Template loaded successfully!")
    print("Modify this script to implement your training pipeline.")


if __name__ == "__main__":
    main()
