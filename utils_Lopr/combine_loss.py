import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_Lopr.dice_loss import DiceLoss
from utils_Lopr.cross_entropy import LabelSmoothing


class CELabelSmoothingPlusDice(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A list of two tensors
        target: A list of two tensors
        other args pass to BinaryDiceLoss
    Return:
        combination loss, dice plus cross entropy with label smoothing
    """
    def __init__(self, smoothing=0.0, weight=None, ignore_index=None, **kwargs):
        super(CELabelSmoothingPlusDice, self).__init__()
        self.kwargs = kwargs
        self.smoothing = smoothing
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        # print(predict.size())
        # print(target.size())
        assert predict.size() == target.size()
        dice = DiceLoss(weight=self.weight,ignore_index=self.ignore_index,**self.kwargs)
        dice_loss = dice(predict,target)

        cels = LabelSmoothing(smoothing=self.smoothing,weight=self.weight)
        cels_loss = cels(predict,target)
        
        total_loss = cels_loss + dice_loss

        return total_loss
    
