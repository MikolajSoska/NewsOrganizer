from typing import Any

import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch import Tensor


class Score:
    def __init__(self, **kwargs: Any):
        for name, value in kwargs.items():
            self.__dict__[name] = value

    def __str__(self) -> str:
        return ', '.join({f'{name}: {value}' for name, value in self.__dict__.items()})

    def __repr__(self) -> str:
        return self.__str__()


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
