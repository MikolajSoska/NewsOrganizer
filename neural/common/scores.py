import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch import Tensor


class Accuracy(nn.Module):
    @staticmethod
    def forward(predictions: Tensor, target: Tensor) -> float:
        labels = torch.argmax(predictions, dim=-1)
        return metrics.accuracy_score(torch.flatten(target.cpu()), torch.flatten(labels.cpu()))


class F1Score(nn.Module):
    def __init__(self, average: str):
        super().__init__()
        self.average = average

    def forward(self, predictions: Tensor, target: Tensor) -> float:
        labels = torch.argmax(predictions, dim=-1)
        return metrics.f1_score(torch.flatten(target.cpu()), torch.flatten(labels.cpu()), average=self.average)
