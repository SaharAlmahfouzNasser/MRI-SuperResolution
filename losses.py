import torch
from torch import nn


class AdversarialLoss(nn.Module):
    def __init__(self, logits=False):
        super().__init__()
        if logits:
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.BCELoss()

    @staticmethod
    def get_labels(predictions, is_real):
        if is_real:
            return torch.ones_like(predictions)
        else:
            return torch.zeros_like(predictions)

    def forward(self, predictions, is_real):
        ground_truth = AdversarialLoss.get_labels(predictions, is_real)
        return self.loss(predictions, ground_truth)
