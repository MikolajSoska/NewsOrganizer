import abc
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class LossWithReduction(nn.Module, abc.ABC):
    def __init__(self, reduction: str):
        super().__init__()
        if reduction == 'mean':
            self.reduction = torch.mean
        elif reduction == 'sum':
            self.reduction = torch.sum
        elif reduction == 'none':
            self.reduction = None
        else:
            raise ValueError(f'{reduction} is not a valid value for reduction')

    @abc.abstractmethod
    def forward(self, *args: Any) -> Tensor:
        pass


class SummarizationLoss(LossWithReduction):
    def __init__(self, epsilon: float = 1e-12, reduction: str = 'mean'):
        super().__init__(reduction)
        self.epsilon = epsilon

    def forward(self, predictions: Tensor, targets: Tensor, target_lengths: Tensor) -> Tensor:
        padding_mask = torch.clip(targets, min=0, max=1)
        gathered_probabilities = torch.gather(predictions, 2, targets.unsqueeze(2)).squeeze()
        loss = -torch.log(gathered_probabilities + self.epsilon) * padding_mask
        loss = torch.sum(loss, dim=0) / target_lengths

        return self.reduction(loss)


class CoverageLoss(LossWithReduction):
    def __init__(self, weight: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction)
        self.weight = weight

    def forward(self, attention: Tensor, coverage: Tensor, targets: Tensor) -> Tensor:
        padding_mask = torch.clip(targets, min=0, max=1)
        loss = self.weight * torch.sum(torch.min(attention, coverage), dim=1) * padding_mask
        if self.reduction is not None:
            return self.reduction(loss)
        else:
            return loss


class LabelSmoothingCrossEntropy(LossWithReduction):
    def __init__(self, smoothing: float, reduction: str = 'mean'):
        super().__init__(reduction)
        assert 0 <= smoothing < 1, 'Smoothing value has to be from 0 (inclusively) to 1 (exclusively)'
        self.smoothing = smoothing
        self.confidence = 1 - smoothing

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        predictions = torch.flatten(predictions, end_dim=1)
        targets = torch.flatten(targets)
        class_number = predictions.shape[-1]

        predictions_log = torch.log_softmax(predictions, dim=-1)
        encoding = torch.full_like(predictions_log, self.smoothing / (class_number - 1))
        encoding = torch.scatter(encoding, 1, targets.unsqueeze(1), self.confidence)
        loss = torch.sum(-encoding * predictions_log, dim=-1)

        return self.reduction(loss)
