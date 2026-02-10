import json
import joblib
import torch
import numpy as np
from pathlib import Path
from model import MambaRULModel

class RULPredictor:
    def __init__(self, model_dir: str = "."):
        self.model_dir = Path(model_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 1. Load Configuration
        with open(self.model_dir / "config.json", "r") as f:
            self.config = json.load(f)
        
        # 2. Load Scaler
        self.scaler = joblib.load(self.model_dir / "scaler.save")
        
        # 3. Initialize Model
        self.model = MambaRULModel(
            input_dim=self.config.get('input_dim', 1),
            d_model=self.config.get('d_model', 128),
            d_state=self.config.get('d_state', 16),
            d_conv=self.config.get('d_conv', 4),
            expand=self.config.get('expand', 2),
            num_layers=self.config.get('num_layers', 4),
            dropout=self.config.get('dropout', 0.1),
            use_mamba=True
        )
        
        # 4. Load Weights
        weights_path = self.model_dir / "mamba_rul_model.pth"
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        # Handle state dict structure
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {weights_path}")

    def preprocess(self, signal: np.ndarray) -> torch.Tensor:
        # 1. Clean (remove NaNs)
        signal = signal[np.isfinite(signal)]
        
        # 2. Reshape for scaler (N, 1)
        signal = signal.reshape(-1, 1)
        
        # 3. Normalize
        signal = self.scaler.transform(signal).flatten()
        
        # 4. Create sequences (simple sliding window for inference)
        # Using the window size from config
        window_size = self.config.get('window_size', 2048)
        
        if len(signal) < window_size:
            raise ValueError(f"Signal length {len(signal)} is smaller than window size {window_size}")
            
        # Take the last window for the most recent RUL prediction
        # (Or you could slide across the whole signal)
        last_window = signal[-window_size:]
        
        # Convert to Tensor (Batch=1, Channels=1, Time)
        tensor = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, signal: np.ndarray) -> float:
        input_tensor = self.preprocess(signal)
        
        with torch.no_grad():
            rul_pred = self.model(input_tensor)
            
        return rul_pred.item()

if __name__ == "__main__":
    # Example Usage
    predictor = RULPredictor(model_dir=".")
    
    # Simulate a vibration signal (replace with real data loading)
    print("Generating dummy signal...")
    dummy_signal = np.random.randn(5000) * 0.1  # Random vibration noise
    
    try:
        rul = predictor.predict(dummy_signal)
        print(f"Predicted Remaining Useful Life (RUL): {rul:.2f} time units")
    except Exception as e:
        print(f"Prediction failed: {e}")
