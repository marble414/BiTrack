"""InfoNCE loss."""
import torch
import torch.nn.functional as F

def infonce(anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor, tau: float) -> torch.Tensor:
    pos = torch.exp((anchor * positive).sum() / tau)
    neg = torch.exp((anchor.unsqueeze(1) * negatives).sum(dim=-1) / tau).sum()
    return -torch.log(pos / (pos + neg + 1e-6))
