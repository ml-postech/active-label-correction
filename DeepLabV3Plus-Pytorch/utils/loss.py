import torch.nn as nn
import torch.nn.functional as F
import torch 
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()
        
class PGCE(nn.Module):
    def __init__(self, size_average=True, ignore_index=255):
        super(PGCE, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, targets, q_mask):
        n, c, h, w = inputs.size()
        print(inputs.size())

        p = F.softmax(inputs, dim=1)
        yg = torch.gather(p, 1, torch.unsqueeze(targets.long(), 1))
        loss = (1 - yg ** q_mask) / q_mask
        loss = torch.mean(loss)

        if self.size_average:
            loss /= n

        return loss

    def forward(self, inputs, targets, q_masks):
        

        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()
        
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2

        score = (
            2.0
            * (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) + self.smooth)
        )
        score = 1 - score.sum() / num
        return score
    
class DiceScore(torch.nn.Module):
    def __init__(self, smooth=1):
        super(DiceScore, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets, sigmoid=True):
        num = targets.shape[0]

        # probs = torch.sigmoid(logits)
        preds = preds.reshape(num, -1)
        m2 = targets.reshape(num, -1)
        intersection = preds * m2

        score = (
            2.0
            * (intersection.sum(1) + self.smooth)
            / (preds.sum(1) + m2.sum(1) + self.smooth)
        )
        score = score.sum() / num
        return score