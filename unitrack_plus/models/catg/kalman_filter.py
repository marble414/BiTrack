"""Minimal Kalman filter implementation."""
import torch

class KalmanFilter:
    def __init__(self):
        self.state = torch.zeros(11)
        self.P = torch.eye(11)
    def predict(self, dt: float) -> torch.Tensor:
        F = torch.eye(11)
        F[0,3] = F[1,4] = F[2,5] = dt
        self.state = F @ self.state
        self.P = F @ self.P @ F.t()
        return self.state
    def update(self, measurement: torch.Tensor):
        self.state[:7] = measurement
