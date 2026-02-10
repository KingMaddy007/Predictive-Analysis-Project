"""
SECTION 8 — BASELINE MODEL IMPLEMENTATION

This module implements baseline models (LSTM and Transformer) for comparison
with the Mamba-RUL model.

Author: Mamba-RUL Project
Date: 2026-02-10
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class LSTMBaseline(nn.Module):
    """
    LSTM-based baseline model for RUL prediction.
    
    Architecture:
        Input → Projection → LSTM Layers → Pooling → FC → RUL Output
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM baseline.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMBaseline, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Layer normalization
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output head
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, seq_len) or (batch, seq_len)
        
        Returns:
            RUL predictions of shape (batch,)
        """
        # Handle input shapes
        if x.dim() == 3:
            x = x.transpose(1, 2)  # (batch, seq_len, channels)
        elif x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        
        # Input projection
        x = self.input_projection(x)  # (batch, seq_len, hidden_dim)
        
        # LSTM
        x, _ = self.lstm(x)  # (batch, seq_len, hidden_dim * directions)
        
        # Layer norm
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Global pooling (take last timestep or mean)
        # Using mean pooling for better stability
        x = torch.mean(x, dim=1)  # (batch, hidden_dim * directions)
        
        # Regression head
        rul = self.fc_layers(x)  # (batch, 1)
        
        return rul.squeeze(-1)  # (batch,)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerBaseline(nn.Module):
    """
    Transformer-based baseline model for RUL prediction.
    
    Architecture:
        Input → Projection → Positional Encoding → Transformer Encoder → Pooling → FC → RUL
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize Transformer baseline.
        
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
        """
        super(TransformerBaseline, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output head
        self.fc_layers = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, seq_len) or (batch, seq_len)
        
        Returns:
            RUL predictions of shape (batch,)
        """
        # Handle input shapes
        if x.dim() == 3:
            x = x.transpose(1, 2)  # (batch, seq_len, channels)
        elif x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        
        # Input projection
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Layer norm
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Global pooling (mean pooling)
        x = torch.mean(x, dim=1)  # (batch, d_model)
        
        # Regression head
        rul = self.fc_layers(x)  # (batch, 1)
        
        return rul.squeeze(-1)  # (batch,)


class GRUBaseline(nn.Module):
    """
    GRU-based baseline model for RUL prediction.
    
    Similar to LSTM but with GRU cells.
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        """
        Initialize GRU baseline.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of GRU layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional GRU
        """
        super(GRUBaseline, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Layer normalization
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.layer_norm = nn.LayerNorm(gru_output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output head
        self.fc_layers = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Handle input shapes
        if x.dim() == 3:
            x = x.transpose(1, 2)
        elif x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # Input projection
        x = self.input_projection(x)
        
        # GRU
        x, _ = self.gru(x)
        
        # Layer norm
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Global pooling
        x = torch.mean(x, dim=1)
        
        # Regression head
        rul = self.fc_layers(x)
        
        return rul.squeeze(-1)


def get_model_summary(model: nn.Module, model_name: str) -> str:
    """
    Get a summary of the model.
    
    Args:
        model: PyTorch model
        model_name: Name of the model
    
    Returns:
        String summary
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"""
{'='*70}
{model_name.upper()} MODEL SUMMARY
{'='*70}
Total Parameters:     {total_params:,}
Trainable Parameters: {trainable_params:,}
Model Size:           {total_params * 4 / 1024**2:.2f} MB (float32)
{'='*70}
"""
    return summary


def main():
    """Demo baseline models."""
    print("Section 8: Baseline Model Implementation")
    print()
    
    # Test LSTM
    lstm_model = LSTMBaseline(input_dim=1, hidden_dim=128, num_layers=4)
    print(get_model_summary(lstm_model, "LSTM Baseline"))
    
    # Test Transformer
    transformer_model = TransformerBaseline(input_dim=1, d_model=128, num_layers=4)
    print(get_model_summary(transformer_model, "Transformer Baseline"))
    
    # Test GRU
    gru_model = GRUBaseline(input_dim=1, hidden_dim=128, num_layers=4)
    print(get_model_summary(gru_model, "GRU Baseline"))
    
    # Test forward pass
    dummy_input = torch.randn(8, 1, 2048)
    
    print("\nTesting forward pass:")
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        lstm_out = lstm_model(dummy_input)
        transformer_out = transformer_model(dummy_input)
        gru_out = gru_model(dummy_input)
    
    print(f"LSTM output shape: {lstm_out.shape}")
    print(f"Transformer output shape: {transformer_out.shape}")
    print(f"GRU output shape: {gru_out.shape}")
    
    print("\n✓ All baseline models working correctly!")


if __name__ == "__main__":
    main()
