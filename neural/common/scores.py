import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch import Tensor


class Accuracy(nn.Module):
    @staticmethod
    def forward(predictions: Tensor, target: Tensor) -> float:
        labels = torch.argmax(predictions, dim=-1)
        accuracy = []
        for i in range(target.shape[1]):
            accuracy.append(accuracy_score(target[:, i].cpu(), labels[:, i].cpu()))

        return torch.mean(torch.tensor(accuracy)).item()
