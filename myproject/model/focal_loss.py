import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, y_pred, y, alpha=1, gamma=2):
        super().__init__()

        self.gamma = gamma
        self.y = y
        self.y_pred = y_pred
        self.y_pred = y_pred
        self.alpha = alpha

    def forward(self, y_pred, y):
        
        requires_grad = True

        CE = F.cross_entropy(self.y_pred, self.y)
        pt = torch.exp(-CE)
        loss = self.alpha*(1-pt)**self.gamma*CE 

        return loss.mean()
