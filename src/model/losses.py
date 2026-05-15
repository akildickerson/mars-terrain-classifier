import torch.nn as nn
import torch.nn.functional as F
import torch

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = torch.softmax(predictions, dim=1)
        targets = torch.nn.functional.one_hot(targets, num_classes=5)
        targets = targets.permute(0, 3, 1, 2).float()
        preds = predictions.reshape(-1)
        targs = targets.reshape(-1)
        intersection = (preds * targs).sum()
        dice = (2 * intersection + self.smooth) / (preds.sum() + targs.sum() + self.smooth)
        return 1 - dice
    
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, class_weights=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = 1 - dice_weight
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss(class_weights=class_weights)

    def forward(self, predictions, targets):
        return self.dice_weight * self.dice(predictions, targets) + self.ce_weight * self.ce(predictions, targets)