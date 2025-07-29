"""Multi-Scale Multi-D Conv fusion network stub."""
import torch
import torch.nn as nn

class MSMDFusion(nn.Module):
    """Three-branch 3D conv network."""
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.small = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.med = nn.Conv3d(in_channels, 64, kernel_size=5, padding=2)
        self.large = nn.Conv3d(in_channels, 64, kernel_size=7, padding=3)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_small = self.small(x)
        f_med = self.med(x)
        f_large = self.large(x)
        return torch.cat([f_small, f_med, f_large], dim=1)
