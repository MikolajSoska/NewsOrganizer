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
        x = x.permute(1, 0, 2)

        hidden = hidden.transpose(0, 1).contiguous()
        hidden = self.reduce_hidden(hidden)
        cell = cell.transpose(0, 1).contiguous()
        cell = self.reduce_cell(cell)

        return x, x_out, (hidden, cell)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, batch_size: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.features = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            utils.Unsqueeze(0),
        )
        self.coverage = nn.Sequential(
            utils.View(-1, 1),
            nn.Linear(1, hidden_size * 2)
        )
        self.attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_size * 2, 1),
            utils.View(-1, batch_size),
            nn.Softmax(dim=1),
            utils.Permute(1, 0),
            utils.Unsqueeze(1)
        )

    def forward(self, hidden: torch.Tensor, encoder_out: torch.Tensor, encoder_features: torch.Tensor,
                coverage: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence_len, hidden_size = encoder_out.shape

        decoder_features = self.features(hidden)
        decoder_features = decoder_features.expand(sequence_len, batch, hidden_size).contiguous().view(-1, hidden_size)
        coverage_features = self.coverage(coverage)

        attention = encoder_features + decoder_features + coverage_features
        attention = self.attention(attention)

        context = torch.bmm(attention, encoder_out).view(-1, 2 * self.hidden_size)
        attention = attention.view(-1, sequence_len).permute(1, 0)
        coverage = coverage + attention

        return context, attention, coverage


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = Attention(hidden_size)
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size)
        self.context = nn.Sequential(
            nn.Linear(hidden_size * 2 + embedding_dim, embedding_dim),
            utils.Unsqueeze(0)
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.Linear(hidden_size, vocab_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x_in: torch.Tensor, hidden_in: Tuple[torch.Tensor, torch.Tensor], encoder_out: torch.Tensor,
                encoder_features: torch.Tensor, context: torch.Tensor,
                coverage: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.embedding(x_in)
        x = torch.cat((context, x), dim=1)
        x = self.context(x)
        x, (hidden, cell) = self.lstm(x, hidden_in)

        hidden = hidden.view(-1, self.hidden_size)
        cell = cell.view(-1, self.hidden_size)
        hidden_out = torch.cat((hidden, cell), dim=1)
        context, attention, coverage_next = self.attention(hidden_out, encoder_out, encoder_features, coverage)

        x = x.view(-1, self.hidden_size)
        x = torch.cat((x, context), dim=1)
        x = self.out(x)

        return x, hidden_out, context, attention, coverage_next
