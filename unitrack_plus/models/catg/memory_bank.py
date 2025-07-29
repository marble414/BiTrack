"""Memory bank for trajectory features."""
from collections import deque
import torch

class MemoryBank:
    def __init__(self, max_len: int):
        self.max_len = max_len
        self.mem = deque(maxlen=max_len)
    def add(self, feat: torch.Tensor):
        self.mem.append(feat.detach())
    def similarity(self, feat: torch.Tensor) -> float:
        if not self.mem:
            return 0.0
        sims = [ (feat * m).sum()/(feat.norm()*m.norm()+1e-6) for m in self.mem ]
        return max(sims)
