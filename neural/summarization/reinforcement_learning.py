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

        self.reduce_hidden = nn.Sequential(
            layers.View(1, -1, hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
        )
        self.reduce_cell = nn.Sequential(
            layers.View(1, -1, hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
        )

    def forward(self, inputs: Tensor, inputs_lengths: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        out, (hidden, cell) = self.lstm(inputs, inputs_lengths)
        out = self.features(out)
        hidden = self.reduce_hidden(hidden)
        cell = self.reduce_cell(cell)

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

    def forward(self, outputs: Tensor, encoder_out: Tensor, temporal_scores: Tensor = None) -> Tuple[Tensor, Tensor]:
        temporal_attention = self.attention(outputs, encoder_out)
        if temporal_scores is not None:
            attention = temporal_attention / temporal_scores
        else:
            attention = temporal_attention

        attention = self.normalize(attention)
        encoder_out = torch.transpose(encoder_out, 1, 2)
        context = self.context(attention, encoder_out)

        return context, temporal_attention
