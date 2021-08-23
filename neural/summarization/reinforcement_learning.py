from typing import Tuple

import torch.nn as nn
from torch import Tensor

import neural.common.layers as layers


class Encoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.lstm = layers.PackedRNN(nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=True))

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
        hidden = self.reduce_hidden(hidden)
        cell = self.reduce_cell(cell)

        return out, (hidden, cell)
