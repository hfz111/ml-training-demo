import torch
from model import SimpleModel

def test_model_forward():

    model = SimpleModel()

    x = torch.randn(4, 10)

    y = model(x)

    assert y.shape == (4,1)
