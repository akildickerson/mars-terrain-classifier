import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNet(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", num_classes=5, in_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.model = smp.Unet(
            encoder_name=encoder_name, 
            encoder_weights=encoder_weights, 
            classes=self.num_classes, 
            in_channels=in_channels
        )
