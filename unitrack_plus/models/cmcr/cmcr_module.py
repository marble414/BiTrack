"""Cross-Modal Consistency Registration (CMCR) module stub."""
from typing import List
import torch
import torch.nn as nn
from .feature_extractor import TwoDFeatureExtractor, ThreeDFeatureExtractor

class CMCR(nn.Module):
    """CMCR main class."""
    def __init__(self):
        super().__init__()
        self.img_extractor = TwoDFeatureExtractor()
        self.pc_extractor = ThreeDFeatureExtractor()
    def forward(self, images: torch.Tensor, boxes2d: torch.Tensor, points: torch.Tensor) -> List[dict]:
        """Runs CMCR on inputs and returns dummy outputs."""
        img_feat = self.img_extractor(images, boxes2d)
        pc_feat = self.pc_extractor(points)
        result = []
        for i in range(img_feat.size(0)):
            result.append({"bbox_3d": torch.zeros(7), "score": 1.0, "class": 0,
                           "feature": (img_feat[i] + pc_feat[i]).detach()})
        return result
