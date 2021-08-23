from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

import neural.common.layers as layers


class Encoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = layers.PackedRNN(nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                                             bidirectional=True))
        self.features = nn.Sequential(
            nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False),
            layers.Permute(1, 2, 0)
        )

    def forward(self, inputs: Tensor, inputs_lengths: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        out, (hidden, cell) = self.lstm(inputs, inputs_lengths)
        out = self.features(out)
        hidden = hidden.view(-1, 2 * self.hidden_size)
        cell = cell.view(-1, 2 * self.hidden_size)

        return out, (hidden, cell)


class IntraTemporalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = layers.SequentialMultiInput(
            layers.MatrixProduct(),
            layers.Exponential()
        )

        self.normalize = layers.Normalize(-1)
        self.context = layers.SequentialMultiInput(
            layers.MatrixProduct(),
            layers.Transpose(0, 1)
        )

    def forward(self, outputs_hidden: Tensor, encoder_out: Tensor,
                temporal_scores_sum: Tensor = None) -> Tuple[Tensor, Tensor]:
        outputs_hidden = torch.transpose(outputs_hidden, 0, 1)
        temporal_attention = self.attention(outputs_hidden, encoder_out)
        if temporal_scores_sum is not None:
            attention = temporal_attention / temporal_scores_sum
            temporal_scores_sum = temporal_attention + temporal_scores_sum
        else:
            attention = temporal_attention
            temporal_scores_sum = temporal_attention

        attention = self.normalize(attention)
        encoder_out = torch.transpose(encoder_out, 1, 2)
        context = self.context(attention, encoder_out)

        return context, temporal_scores_sum


class IntraDecoderAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention_first = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            layers.Transpose(1, 2)
        )
        self.attention_second = layers.SequentialMultiInput(
            layers.MatrixProduct(),
            nn.Softmax(dim=-1)
        )
        self.context = layers.SequentialMultiInput(
            layers.MatrixProduct(),
            layers.Transpose(0, 1)
        )

    def forward(self, decoder_hidden: Tensor, previous_hidden: Tensor = None) -> Tuple[Tensor, Tensor]:
        if previous_hidden is None:
            context = torch.zeros_like(decoder_hidden)
            return context, decoder_hidden

        decoder_hidden = torch.transpose(decoder_hidden, 0, 1)
        attention = self.attention_first(previous_hidden)
        attention = self.attention_second(decoder_hidden, attention)
        context = self.context(attention, previous_hidden)
        previous_hidden = torch.cat((previous_hidden, decoder_hidden), dim=1)

        return context, previous_hidden
