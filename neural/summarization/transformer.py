import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

import neural.common.layers as layers


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.encoding_features = self.__get_encoding_features(embedding_dim)

    @staticmethod
    def __get_encoding_features(embedding_dim: int) -> nn.Parameter:
        exponent = 2 * torch.arange(embedding_dim) / embedding_dim
        encoding_features = torch.full_like(exponent, 1. / 10000)
        encoding_features = torch.pow(encoding_features, exponent)
        encoding_features = torch.unsqueeze(encoding_features, dim=0)

        # Constant variable (nn.Parameter to track module device change, etc.)
        return nn.Parameter(encoding_features, requires_grad=False)

    def forward(self, inputs: Tensor) -> Tensor:
        encoding_positions = torch.arange(inputs.shape[0], device=inputs.device, dtype=torch.float)
        encoding_positions = torch.unsqueeze(encoding_positions, dim=1)

        encoding = torch.matmul(encoding_positions, self.encoding_features)
        encoding[:, ::2] = torch.sin(encoding[:, ::2])
        encoding[:, 1::2] = torch.cos(encoding[:, 1::2])
        encoding = torch.unsqueeze(encoding, dim=1)

        return inputs + encoding


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, key_and_query_dim: int, value_dim: int, heads_number: int):
        super().__init__()
        self.scale = math.sqrt(key_and_query_dim)

        self.query = nn.Sequential(
            nn.Linear(embedding_dim, heads_number * key_and_query_dim, bias=False),
            nn.Unflatten(-1, (heads_number, key_and_query_dim)),
            layers.Permute(1, 2, 0, 3)
        )
        self.key = nn.Sequential(
            nn.Linear(embedding_dim, heads_number * key_and_query_dim, bias=False),
            nn.Unflatten(-1, (heads_number, key_and_query_dim)),
            layers.Permute(1, 2, 3, 0)
        )
        self.value = nn.Sequential(
            nn.Linear(embedding_dim, heads_number * value_dim, bias=False),
            nn.Unflatten(-1, (heads_number, value_dim)),
            layers.Permute(1, 2, 0, 3)
        )
        self.out = nn.Sequential(
            layers.Transpose(1, 2),
            nn.Flatten(start_dim=2),
            nn.Linear(heads_number * value_dim, embedding_dim, bias=False),
            layers.Transpose(0, 1)
        )

    def forward(self, inputs_query: Tensor, inputs_key: Tensor, inputs_value: Tensor, mask: Tensor = None) -> Tensor:
        query = self.query(inputs_query)
        key = self.key(inputs_key)
        value = self.value(inputs_value)

        attention = torch.matmul(query, key) / self.scale
        if mask is not None:
            attention = attention.masked_fill(mask, -math.inf)

        attention = torch.softmax(attention, dim=-1)
        attention = torch.matmul(attention, value)

        return self.out(attention)


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, feed_forward_size: int):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_size),
            nn.ReLU(),
            nn.Linear(feed_forward_size, embedding_dim)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.feed_forward(inputs)


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, key_and_query_dim: int, value_dim: int, heads_number: int,
                 feed_forward_size: int):
        super().__init__()
        self.network = layers.SequentialMultiInput(
            layers.Residual(
                SelfAttention(embedding_dim, key_and_query_dim, value_dim, heads_number)
            ),
            nn.LayerNorm(embedding_dim),
            layers.Residual(
                FeedForward(embedding_dim, feed_forward_size)
            ),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, inputs: Tensor, inputs_mask: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.network(inputs, inputs, inputs, inputs_mask)
        return out, inputs_mask


class Encoder(nn.Module):
    def __init__(self, encoder_layers: int, vocab_size: int, embedding_dim: int, key_and_query_dim: int, value_dim: int,
                 heads_number: int, feed_forward_size: int):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim, padding_idx=0),
            PositionalEncoding(embedding_dim)
        )
        self.encoders = layers.SequentialMultiInput(
            *[EncoderLayer(embedding_dim, key_and_query_dim, value_dim, heads_number, feed_forward_size)
              for _ in range(encoder_layers)]
        )

    def forward(self, inputs: Tensor, inputs_mask: Tensor) -> Tensor:
        embedded = self.embedding(inputs)
        out, _ = self.encoders(embedded, inputs_mask)
        return out
