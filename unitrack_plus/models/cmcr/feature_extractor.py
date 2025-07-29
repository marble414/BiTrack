"""Feature extractors for CMCR module."""
from typing import Tuple
import torch
import torch.nn as nn
from torchvision.ops import roi_align

class ResNetFPNBackbone(nn.Module):
    """Simplified ResNet-50 FPN backbone placeholder."""
    def __init__(self):
        super().__init__()
        # Placeholder layers
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class TwoDFeatureExtractor(nn.Module):
    """Extracts 256-dim ROI features from images."""
    def __init__(self):
        super().__init__()
        self.backbone = ResNetFPNBackbone()
        self.fc = nn.Linear(64, 256)
    def forward(self, images: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(images)
        rois = roi_align(feats, boxes, output_size=(7,7))
        pooled = rois.mean(dim=[2,3])
        return self.fc(pooled)

class ThreeDFeatureExtractor(nn.Module):
    """Voxelizes cropped point clouds and outputs features."""
    def __init__(self):
        super().__init__()
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # Placeholder: return dummy tensor
        return torch.zeros((points.size(0), 256))
