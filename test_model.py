import torch
from src.model.unet import MarsUNet

model = MarsUNet(encoder_weights=None)
x = torch.randn(1, 3, 512, 512)
print(model.predict(x).shape)