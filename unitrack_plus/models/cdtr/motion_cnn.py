"""1D CNN for motion smoothness score."""
import torch
import torch.nn as nn

class MotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 1, 3, padding=1),
        )
        self.fc = nn.Linear(1, 1)
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.cnn(seq.transpose(1,2))
        g = x.mean(dim=-1)
        return torch.sigmoid(self.fc(g.squeeze(-1)))
