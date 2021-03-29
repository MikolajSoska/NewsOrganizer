from typing import Tuple

import torch
import torch.nn as nn

import model.utils as utils


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=True)
        self.linear = nn.Sequential(
            utils.View(-1, 2 * hidden_size),
            nn.Linear(hidden_size * 2, hidden_size * 2)
        )
        self.reduce_hidden = nn.Sequential(
            utils.View(-1, hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            utils.View(1, -1, hidden_size)
        )
        self.reduce_cell = nn.Sequential(
            utils.View(-1, hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            utils.View(1, -1, hidden_size)
        )

    def forward(self, x_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.embedding(x_in)
        x, (hidden, cell) = self.lstm(x)
        x = x.contiguous()
        x_out = self.linear(x)

        hidden = hidden.transpose(0, 1).contiguous()
        hidden = self.reduce_hidden(hidden)
        cell = cell.transpose(0, 1).contiguous()
        cell = self.reduce_cell(cell)

        return x, x_out, (hidden, cell)
