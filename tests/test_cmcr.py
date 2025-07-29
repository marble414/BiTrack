import torch
from unitrack_plus.models.cmcr.cmcr_module import CMCR

def test_cmcr_forward():
    model = CMCR()
    imgs = torch.randn(1,3,224,224)
    boxes = torch.tensor([[0,0,100,100]], dtype=torch.float)
    points = torch.randn(1,100,3)
    out = model(imgs, boxes, points)
    assert isinstance(out, list)
