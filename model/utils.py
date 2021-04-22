import abc
from typing import Tuple, Any, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        x_reshape = x.contiguous().view(-1, x.size(-1))

        y = self.module(x_reshape)
        y = y.view(-1, x.size(1), y.size(-1))

        return y


class PackedRNN(nn.Module):
    def __init__(self, rnn_module: nn.RNNBase):
        super().__init__()
        self.rnn_module = rnn_module

    def forward(self, sequence: torch.Tensor, sequence_lengths: torch.Tensor,
                *rnn_args: Any) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        lengths = sequence_lengths.squeeze().tolist()
        sequence_packed = pack_padded_sequence(sequence, lengths, enforce_sorted=False)
        output, hidden = self.rnn_module(sequence_packed, *rnn_args)
        output_padded, _ = pad_packed_sequence(output, total_length=int(sequence_lengths.max()))

        return output_padded, hidden


class View(nn.Module):
    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        return x_in.view(self.shape)


class Permute(nn.Module):
    def __init__(self, *dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        return x_in.permute(self.dimensions)


class Squeeze(nn.Module):
    def __init__(self, dimension: int = None):
        super().__init__()
        self.dimension = dimension

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        return x_in.squeeze(dim=self.dimension)


class Unsqueeze(nn.Module):
    def __init__(self, dimension: int = None):
        super().__init__()
        self.dimension = dimension

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        return x_in.unsqueeze(dim=self.dimension)


class Normalize(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        factor = torch.sum(x_in, dim=self.dimension, keepdim=True)
        return x_in / factor


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
    def forward(self, *args: Any) -> torch.Tensor:
        pass


class SummarizationLoss(LossWithReduction):
    def __init__(self, epsilon: float = 1e-12, reduction: str = 'mean'):
        super().__init__(reduction)
        self.epsilon = epsilon

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        padding_mask = torch.clip(targets, min=0, max=1)
        gathered_probabilities = torch.gather(predictions, 2, targets.unsqueeze(2)).squeeze()
        loss = -torch.log(gathered_probabilities + self.epsilon) * padding_mask
        loss = torch.sum(loss, dim=0) / target_lengths

        return self.reduction(loss)


class CoverageLoss(LossWithReduction):
    def __init__(self, weight: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction)
        self.weight = weight

    def forward(self, attention: torch.Tensor, coverage: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        padding_mask = torch.clip(targets, min=0, max=1)
        loss = self.weight * torch.sum(torch.min(attention, coverage), dim=1) * padding_mask
        if self.reduction is not None:
            return self.reduction(loss)
        else:
            return loss
