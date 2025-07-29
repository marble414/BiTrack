"""Consistency-Aware Trajectory Generation stub."""
from typing import List
import torch
from .kalman_filter import KalmanFilter
from .gru_predictor import GRUPredictor
from .memory_bank import MemoryBank

class CATG:
    def __init__(self):
        self.kf = KalmanFilter()
        self.pred = GRUPredictor()
        self.stm = MemoryBank(5)
    def step(self, feat: torch.Tensor, bbox: torch.Tensor, dt: float):
        self.kf.predict(dt)
        self.kf.update(bbox)
        self.stm.add(feat)
        aux = torch.zeros(64)
        self.pred(feat, aux)
