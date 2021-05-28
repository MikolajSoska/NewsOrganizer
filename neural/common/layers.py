from typing import Tuple, Any, Union

import torch
import torch.nn as nn
from torch import Tensor
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

    def forward(self, sequence: Tensor, sequence_lengths: Tensor,
                *rnn_args: Any) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, ...]]]:
        lengths = sequence_lengths.squeeze().tolist()
        if sequence_lengths.shape[0] == 1:
            lengths = [lengths]

        sequence_packed = pack_padded_sequence(sequence, lengths, enforce_sorted=False)
        output, hidden = self.rnn_module(sequence_packed, *rnn_args)
        output_padded, _ = pad_packed_sequence(output, total_length=int(sequence_lengths.max()))

        return output_padded, hidden


class View(nn.Module):
    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def forward(self, x_in: Tensor) -> Tensor:
        return x_in.view(self.shape)


class Permute(nn.Module):
    def __init__(self, *dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    def forward(self, x_in: Tensor) -> Tensor:
        return x_in.permute(self.dimensions)


class Squeeze(nn.Module):
    def __init__(self, dimension: int = None):
        super().__init__()
        self.dimension = dimension

    def forward(self, x_in: Tensor) -> Tensor:
        return x_in.squeeze(dim=self.dimension)


class Unsqueeze(nn.Module):
    def __init__(self, dimension: int = None):
        super().__init__()
        self.dimension = dimension

    def forward(self, x_in: Tensor) -> Tensor:
        return x_in.unsqueeze(dim=self.dimension)


class Normalize(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, x_in: Tensor) -> Tensor:
        factor = torch.sum(x_in, dim=self.dimension, keepdim=True)
        return x_in / factor
