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
            nn.Dropout(0.1),
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
            nn.Linear(feed_forward_size, embedding_dim),
            nn.Dropout(0.1)
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
    def __init__(self, encoder_layers: int, embedding_dim: int, key_and_query_dim: int, value_dim: int,
                 heads_number: int, feed_forward_size: int):
        super().__init__()
        self.encoders = layers.SequentialMultiInput(
            *[EncoderLayer(embedding_dim, key_and_query_dim, value_dim, heads_number, feed_forward_size)
              for _ in range(encoder_layers)]
        )

    def forward(self, inputs: Tensor, inputs_mask: Tensor) -> Tensor:
        out, _ = self.encoders(inputs, inputs_mask)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, key_and_query_dim: int, value_dim: int, heads_number: int,
                 feed_forward_size: int):
        super().__init__()
        self.self_attention = layers.SequentialMultiInput(
            layers.Residual(
                SelfAttention(embedding_dim, key_and_query_dim, value_dim, heads_number)
            ),
            nn.LayerNorm(embedding_dim)
        )
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

    def forward(self, outputs: Tensor, outputs_mask: Tensor, encoder_out: Tensor, encoder_mask: Tensor) -> \
            Tuple[Tensor, Tensor, Tensor, Tensor]:
        self_attention = self.self_attention(outputs, outputs, outputs, outputs_mask)
        out = self.network(self_attention, encoder_out, encoder_out, encoder_mask)

        return out, outputs_mask, encoder_out, encoder_mask


class Decoder(nn.Module):
    def __init__(self, decoder_layers: int, embedding_dim: int, key_and_query_dim: int, value_dim: int,
                 heads_number: int, feed_forward_size: int):
        super().__init__()
        self.decoders = layers.SequentialMultiInput(
            *[DecoderLayer(embedding_dim, key_and_query_dim, value_dim, heads_number, feed_forward_size)
              for _ in range(decoder_layers)]
        )

    @staticmethod
    def __get_decoder_mask(sequence):
        sequence_length = sequence.shape[0]
        mask = torch.triu(torch.ones(sequence_length, sequence_length, device=sequence.device), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add dimensions to be broadcastable to batch x heads x seq x seq
        return mask

    def forward(self, outputs: Tensor, outputs_mask: Tensor, encoder_out: Tensor, encoder_mask: Tensor) -> Tensor:
        outputs_mask = outputs_mask | self.__get_decoder_mask(outputs)
        out, _, _, _ = self.decoders(outputs, outputs_mask, encoder_out, encoder_mask)

        return out


class Transformer(nn.Module):
    def __init__(self, encoder_layers: int, decoder_layers: int, vocab_size: int, embedding_dim: int,
                 key_and_query_dim: int, value_dim: int, heads_number: int, feed_forward_size: int,
                 max_summary_length: int, bos_index: int, padding_idx: int = 0):
        super().__init__()
        self.max_summary_length = max_summary_length
        self.bos_index = bos_index
        self.padding_idx = padding_idx
        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx),
            layers.Multiply(value=math.sqrt(embedding_dim)),
            PositionalEncoding(embedding_dim),
            nn.Dropout(0.1)
        )
        self.encoder = Encoder(encoder_layers, embedding_dim, key_and_query_dim, value_dim, heads_number,
                               feed_forward_size)
        self.decoder = Decoder(decoder_layers, embedding_dim, key_and_query_dim, value_dim, heads_number,
                               feed_forward_size)
        self.out = nn.Sequential(
            nn.Linear(embedding_dim, vocab_size, bias=False),
            nn.Softmax(dim=-1)
        )
        self.out[0].weight = self.embedding[0].weight  # Share weights

    def __get_padding_mask(self, sequence):
        mask = sequence == self.padding_idx
        mask = mask.transpose(0, 1)
        mask = mask.unsqueeze(1).unsqueeze(1)  # Add dimensions to be broadcastable to batch x heads x seq x seq
        return mask

    def forward(self, inputs: Tensor, outputs: Tensor = None) -> Tensor:
        if self.training:
            if outputs is None:
                raise AttributeError('During training reference outputs must be provided.')
        else:  # In validation phase never use passed outputs
            outputs = torch.full((self.max_summary_length + 1, inputs.shape[1]), self.bos_index, dtype=torch.long,
                                 device=inputs.device)

        inputs_mask = self.__get_padding_mask(inputs)
        inputs_embedded = self.embedding(inputs)

        encoder_out = self.encoder(inputs_embedded, inputs_mask)
        if self.training:
            out = self.__decoder_step(outputs, encoder_out, inputs_mask)
        else:
            out = None
            for i in range(self.max_summary_length):
                decoder_input = outputs[:i + 1, :]
                out_step = self.__decoder_step(decoder_input, encoder_out, inputs_mask)
                outputs[1:i + 2, :] = torch.argmax(out_step, dim=-1)
                out = out_step

        return out

    def __decoder_step(self, outputs: Tensor, encoder_out: Tensor, inputs_mask: Tensor) -> Tensor:
        outputs_mask = self.__get_padding_mask(outputs)
        outputs_embedded = self.embedding(outputs)
        decoder_out = self.decoder(outputs_embedded, outputs_mask, encoder_out, inputs_mask)
        return self.out(decoder_out)
