"""
SECTION 5 — MAMBA MODEL IMPLEMENTATION

This module implements the Mamba-RUL model for Remaining Useful Life prediction.
It uses the Selective State Space Model (S6) from the mamba-ssm library.

Architecture:
    Input → Projection → Mamba Block → Temporal Pooling → FC → RUL Output

Author: Mamba-RUL Project
Date: 2026-02-10
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

# Note: mamba_ssm will be imported conditionally to handle installation issues


class MambaRULModel(nn.Module):
    """
    Mamba-based model for Remaining Useful Life prediction.
    
    This model uses the Selective State Space Model (Mamba) to process
    long vibration sequences and predict continuous RUL values.
    
    Key Features:
    - Linear-time complexity for long sequences
    - Selective memory mechanism
    - End-to-end trainable
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_mamba: bool = True
    ):
        """
        Initialize the Mamba-RUL model.
        
        Args:
            input_dim: Input feature dimension (1 for univariate time series)
            d_model: Model dimension (embedding size)
            d_state: SSM state dimension
            d_conv: Local convolution width
            expand: Expansion factor for inner dimension
            num_layers: Number of Mamba blocks to stack
            dropout: Dropout rate
            use_mamba: Whether to use actual Mamba blocks (requires mamba-ssm)
                      If False, uses LSTM as fallback for testing
        """
        super(MambaRULModel, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_mamba = use_mamba
        
        # Input projection: map from input_dim to d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Mamba blocks (or LSTM fallback)
        if use_mamba:
            try:
                from mamba_ssm import Mamba
                
                # Stack multiple Mamba blocks
                self.mamba_blocks = nn.ModuleList([
                    Mamba(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand
                    )
                    for _ in range(num_layers)
                ])
                
                print(f"✓ Using Mamba SSM blocks (d_model={d_model}, layers={num_layers})")
                
            except ImportError:
                print("⚠ Warning: mamba-ssm not installed. Using LSTM fallback.")
                self.use_mamba = False
        
        if not self.use_mamba:
            # Fallback to LSTM for testing without mamba-ssm
            self.lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            print(f"✓ Using LSTM fallback (hidden_size={d_model}, layers={num_layers})")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Temporal pooling (reduce sequence to single vector)
        # We'll use both max and average pooling
        self.pool_type = 'both'  # 'max', 'avg', or 'both'
        
        # Output head for RUL regression
        if self.pool_type == 'both':
            fc_input_dim = d_model * 2
        else:
            fc_input_dim = d_model
        
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Single RUL output
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length)
               or (batch_size, sequence_length)
        
        Returns:
            RUL predictions of shape (batch_size, 1)
        """
        batch_size = x.shape[0]
        
        # Handle different input shapes
        if x.dim() == 3:
            # (batch, channels, seq_len) -> (batch, seq_len, channels)
            x = x.transpose(1, 2)
        elif x.dim() == 2:
            # (batch, seq_len) -> (batch, seq_len, 1)
            x = x.unsqueeze(-1)
        
        # Input projection: (batch, seq_len, input_dim) -> (batch, seq_len, d_model)
        x = self.input_projection(x)
        
        # Apply Mamba blocks or LSTM
        if self.use_mamba:
            for mamba_block in self.mamba_blocks:
                # Mamba block with residual connection
                residual = x
                x = mamba_block(x)
                x = self.layer_norm(x + residual)
                x = self.dropout(x)
        else:
            # LSTM fallback
            x, _ = self.lstm(x)
            x = self.layer_norm(x)
            x = self.dropout(x)
        
        # Temporal pooling: (batch, seq_len, d_model) -> (batch, d_model)
        if self.pool_type == 'max':
            x = torch.max(x, dim=1)[0]
        elif self.pool_type == 'avg':
            x = torch.mean(x, dim=1)
        elif self.pool_type == 'both':
            max_pool = torch.max(x, dim=1)[0]
            avg_pool = torch.mean(x, dim=1)
            x = torch.cat([max_pool, avg_pool], dim=1)
        
        # Regression head: (batch, d_model) -> (batch, 1)
        rul = self.fc_layers(x)
        
        return rul.squeeze(-1)  # (batch,)
    
    def get_model_summary(self) -> str:
        """
        Get a summary of the model architecture.
        
        Returns:
            String summary of the model
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        summary = f"""
{'='*70}
MAMBA-RUL MODEL SUMMARY
{'='*70}
Architecture Type:    {'Mamba SSM' if self.use_mamba else 'LSTM (Fallback)'}
Input Dimension:      {self.input_dim}
Model Dimension:      {self.d_model}
Number of Layers:     {self.num_layers}
Pooling Strategy:     {self.pool_type}

Total Parameters:     {total_params:,}
Trainable Parameters: {trainable_params:,}
Model Size:           {total_params * 4 / 1024**2:.2f} MB (float32)
{'='*70}
"""
        return summary


def test_model_forward_pass(
    model: nn.Module,
    input_shape: Tuple[int, int, int],
    device: str = 'cpu'
) -> bool:
    """
    Test the model with a dummy forward pass.
    
    Args:
        model: The model to test
        input_shape: Shape of input tensor (batch, channels, seq_len)
        device: Device to run on ('cpu' or 'cuda')
    
    Returns:
        True if test passes, False otherwise
    """
    try:
        model = model.to(device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape).to(device)
        
        print(f"Testing forward pass...")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Device: {device}")
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"  ✓ Forward pass successful!")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        return False


def main():
    """Main execution function for demonstration."""
    
    print("=" * 70)
    print("SECTION 5: MAMBA MODEL IMPLEMENTATION")
    print("=" * 70)
    print()
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Model hyperparameters
    config = {
        'input_dim': 1,          # Univariate time series
        'd_model': 128,          # Model dimension
        'd_state': 16,           # SSM state dimension
        'd_conv': 4,             # Convolution width
        'expand': 2,             # Expansion factor
        'num_layers': 4,         # Number of Mamba blocks
        'dropout': 0.1,          # Dropout rate
        'use_mamba': True        # Try to use Mamba, fallback to LSTM if not available
    }
    
    print("Model Configuration:")
    print("-" * 70)
    for key, value in config.items():
        print(f"  {key:<20s}: {value}")
    print()
    
    # Initialize model
    print("Initializing model...")
    model = MambaRULModel(**config)
    
    # Print model summary
    print(model.get_model_summary())
    
    # Test forward pass with different input shapes
    print("\nTesting Model Forward Pass:")
    print("-" * 70)
    
    # Test 1: With channel dimension
    print("\nTest 1: Input with channel dimension (batch=8, channels=1, seq_len=2048)")
    success1 = test_model_forward_pass(
        model,
        input_shape=(8, 1, 2048),
        device=device
    )
    
    # Test 2: Without channel dimension
    print("\nTest 2: Input without channel dimension (batch=8, seq_len=2048)")
    success2 = test_model_forward_pass(
        model,
        input_shape=(8, 2048),
        device=device
    )
    
    # Test 3: Single sample
    print("\nTest 3: Single sample (batch=1, channels=1, seq_len=2048)")
    success3 = test_model_forward_pass(
        model,
        input_shape=(1, 1, 2048),
        device=device
    )
    
    print()
    print("=" * 70)
    
    if all([success1, success2, success3]):
        print("✓ ALL TESTS PASSED - Model is ready for training!")
    else:
        print("✗ SOME TESTS FAILED - Please check the errors above")
    
    print("=" * 70)
    
    return model


if __name__ == "__main__":
    model = main()
