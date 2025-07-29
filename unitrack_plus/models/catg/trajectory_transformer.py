"""Trajectory transformer encoder stub."""
import torch
import torch.nn as nn

class TrajectoryTransformer(nn.Module):
    def __init__(self, d_model: int = 256, num_layers: int = 1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return self.encoder(seq).mean(dim=0)
