"""Simple Gaussian smoother."""
import torch
import torch.nn.functional as F

class GaussianSmoother:
    def __init__(self, window: int = 2, sigma: float = 1.0):
        self.window = window
        self.sigma = sigma
        self.kernel = self._gaussian_kernel()
    def _gaussian_kernel(self) -> torch.Tensor:
        rng = torch.arange(-self.window, self.window+1)
        kernel = torch.exp(-0.5*(rng.float()/self.sigma)**2)
        kernel = kernel/kernel.sum()
        return kernel.view(1,1,-1)
    def smooth(self, seq: torch.Tensor) -> torch.Tensor:
        seq = seq.unsqueeze(0).unsqueeze(0)
        pad = (self.window, self.window)
        out = F.conv1d(F.pad(seq, pad, mode='replicate'), self.kernel)
        return out.squeeze()
