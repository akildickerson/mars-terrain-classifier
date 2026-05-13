import torch.nn as nn
import torch

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        preds = predictions.view(-1)
        targs = targets.view(-1)
        intersection = (preds * targs).sum()
        dice = (2 * intersection + self.smooth) / (preds.sum() + targs.sum() + self.smooth)
        return 1 - dice
    
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = 1 - dice_weight
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        return self.dice_weight * self.dice(predictions, targets) + self.ce_weight * self.ce(predictions, targets)