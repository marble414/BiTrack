"""Pseudo-LiDAR generation placeholder."""
import torch

class PseudoLiDARGenerator:
    def __init__(self):
        pass
    def __call__(self, mask: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        # Produce dummy pseudo points
        points = torch.stack([mask.nonzero()[:,0].float(), mask.nonzero()[:,1].float(), depth[mask.bool()].float()], dim=-1)
        return points
