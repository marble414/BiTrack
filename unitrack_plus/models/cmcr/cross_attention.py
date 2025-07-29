"""Simplified cross-attention module."""
import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, dim: int, alpha: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.scale = dim ** -0.5
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        attn = torch.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1)
        return q + self.alpha * (attn @ k)
