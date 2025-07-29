"""GRU-based feature predictor stub."""
import torch
import torch.nn as nn

class GRUPredictor(nn.Module):
    def __init__(self, input_dim: int = 256+64, hidden_dim: int = 256):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2)
    def forward(self, feat: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([feat, aux], dim=-1).unsqueeze(0)
        out, _ = self.gru(inp)
        return out.squeeze(0)
