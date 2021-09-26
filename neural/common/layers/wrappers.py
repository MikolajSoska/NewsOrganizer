from typing import Tuple, Any, Union

import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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


class Residual(nn.Module):
    def __init__(self, module: nn.Module, input_position: int = 0):
        super().__init__()
        self.module = module
        self.input_position = input_position

    def forward(self, *inputs: Tensor) -> Tensor:
        residual = inputs[self.input_position]
        output = self.module(*inputs)

        return output + residual


class SequentialMultiInput(nn.Sequential):
    def forward(self, *inputs: Any) -> Any:
        for module in self:
            if not isinstance(inputs, tuple):
                inputs = (inputs,)
            inputs = module(*inputs)

        return inputs
