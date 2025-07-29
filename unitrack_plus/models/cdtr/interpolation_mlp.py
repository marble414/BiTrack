"""MLP for interpolation refinement."""
import torch
import torch.nn as nn

class InterpolationMLP(nn.Module):
    def __init__(self, in_dim: int = 256+14+1, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 7)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
