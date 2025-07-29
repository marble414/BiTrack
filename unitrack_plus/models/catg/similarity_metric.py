"""Similarity metrics for CATG."""
import torch

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_n = a / (a.norm(dim=-1, keepdim=True) + 1e-6)
    b_n = b / (b.norm(dim=-1, keepdim=True) + 1e-6)
    return (a_n * b_n).sum(dim=-1)
