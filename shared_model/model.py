import torch
import torch.nn as nn

class MambaRULModel(nn.Module):
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
        super(MambaRULModel, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_mamba = use_mamba
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        if use_mamba:
            try:
                from mamba_ssm import Mamba
                self.mamba_blocks = nn.ModuleList([
                    Mamba(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand
                    )
                    for _ in range(num_layers)
                ])
            except ImportError:
                print("Warning: mamba-ssm not installed. Using LSTM fallback.")
                self.use_mamba = False
        
        if not self.use_mamba:
            self.lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # FC layers for RUL regression (using both max and avg pooling as per training)
        self.fc_layers = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, channels, seq_len) -> (batch, seq_len, channels)
        if x.dim() == 3:
            x = x.transpose(1, 2)
        elif x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # Projection
        x = self.input_projection(x)
        
        # Mamba/LSTM
        if self.use_mamba:
            for mamba_block in self.mamba_blocks:
                residual = x
                x = mamba_block(x)
                x = self.layer_norm(x + residual)
                x = self.dropout(x)
        else:
            x, _ = self.lstm(x)
            x = self.layer_norm(x)
            x = self.dropout(x)
        
        # Pooling (Both Max and Avg)
        max_pool = torch.max(x, dim=1)[0]
        avg_pool = torch.mean(x, dim=1)
        x = torch.cat([max_pool, avg_pool], dim=1)
        
        # Regression
        rul = self.fc_layers(x)
        return rul.squeeze(-1)
