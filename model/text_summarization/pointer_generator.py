from typing import Tuple

import torch
import torch.nn as nn

import model.utils as utils


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_size: int = 256):
        super().__init__()
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
            utils.Unsqueeze(0)
        )
        self.reduce_cell = nn.Sequential(
            utils.View(-1, hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            utils.Unsqueeze(0)
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


class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.features = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            utils.Unsqueeze(1),
        )
        self.coverage = nn.Sequential(
            utils.View(-1, 1),
            nn.Linear(1, hidden_size * 2)
        )
        self.attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_size * 2, 1),
            utils.Squeeze(2),
            nn.Softmax(dim=1),
            utils.Unsqueeze(1)
        )

    def forward(self, hidden: torch.Tensor, encoder_out: torch.Tensor, encoder_features: torch.Tensor,
                coverage: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence_len, hidden_size = encoder_out.shape

        decoder_features = self.features(hidden)
        decoder_features = decoder_features.expand(batch, sequence_len, hidden_size).contiguous().view(-1, hidden_size)
        coverage_features = self.coverage(coverage)

        attention = encoder_features + decoder_features + coverage_features
        attention = self.attention(attention)

        context = torch.bmm(attention, encoder_out).view(-1, self.hidden_size)
        attention = attention.view(-1, sequence_len)
        coverage = coverage.view(-1, sequence_len) + attention

        return context, attention, coverage
