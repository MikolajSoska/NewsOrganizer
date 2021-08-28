import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

import neural.common.layers as layers
from neural.common.layers.decode import BeamSearchDecoder


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.encoding_features = self.__get_encoding_features(embedding_dim)

    @staticmethod
    def __get_encoding_features(embedding_dim: int) -> nn.Parameter:
        exponent = 2 * (torch.arange(embedding_dim) // 2) / embedding_dim
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
    def __init__(self, embedding_dim: int, feed_forward_size: int, dropout_rate: float):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_size),
            nn.ReLU(),
            nn.Linear(feed_forward_size, embedding_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.feed_forward(inputs)


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, key_and_query_dim: int, value_dim: int, heads_number: int,
                 feed_forward_size: int, dropout_rate: float):
        super().__init__()
        self.network = layers.SequentialMultiInput(
            layers.Residual(
                SelfAttention(embedding_dim, key_and_query_dim, value_dim, heads_number)
            ),
            nn.LayerNorm(embedding_dim),
            layers.Residual(
                FeedForward(embedding_dim, feed_forward_size, dropout_rate)
            ),
            nn.LayerNorm(embedding_dim, eps=1e-6)
        )

    def forward(self, inputs: Tensor, inputs_mask: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.network(inputs, inputs, inputs, inputs_mask)
        return out, inputs_mask


class Encoder(nn.Module):
    def __init__(self, encoder_layers: int, embedding_dim: int, key_and_query_dim: int, value_dim: int,
                 heads_number: int, feed_forward_size: int, dropout_rate: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.encoders = layers.SequentialMultiInput(
            *[EncoderLayer(embedding_dim, key_and_query_dim, value_dim, heads_number, feed_forward_size, dropout_rate)
              for _ in range(encoder_layers)]
        )

    def forward(self, inputs: Tensor, inputs_mask: Tensor) -> Tensor:
        inputs = self.layer_norm(inputs)
        out, _ = self.encoders(inputs, inputs_mask)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, key_and_query_dim: int, value_dim: int, heads_number: int,
                 feed_forward_size: int, dropout_rate: float):
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
                FeedForward(embedding_dim, feed_forward_size, dropout_rate)
            ),
            nn.LayerNorm(embedding_dim, eps=1e-6)
        )

    def forward(self, outputs: Tensor, outputs_mask: Tensor, encoder_out: Tensor, encoder_mask: Tensor) -> \
            Tuple[Tensor, Tensor, Tensor, Tensor]:
        self_attention = self.self_attention(outputs, outputs, outputs, outputs_mask)
        out = self.network(self_attention, encoder_out, encoder_out, encoder_mask)

        return out, outputs_mask, encoder_out, encoder_mask


class Decoder(BeamSearchDecoder):
    def __init__(self, vocab_size: int, decoder_layers: int, embedding_dim: int, key_and_query_dim: int, value_dim: int,
                 heads_number: int, feed_forward_size: int, dropout_rate: float, bos_index: int, eos_index: int,
                 padding_index: int, max_output_length: int, embedding_weight: Tensor):
        super().__init__(bos_index, eos_index, max_output_length)
        self.padding_index = padding_index
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.decoders = layers.SequentialMultiInput(
            *[DecoderLayer(embedding_dim, key_and_query_dim, value_dim, heads_number, feed_forward_size, dropout_rate)
              for _ in range(decoder_layers)]
        )
        self.out = nn.Sequential(
            nn.Linear(embedding_dim, vocab_size, bias=False),
            layers.Multiply(value=embedding_dim ** -0.5),
        )
        self.out[0].weight = embedding_weight  # Share weights

    @staticmethod
    def __get_decoder_mask(sequence):
        sequence_length = sequence.shape[0]
        mask = torch.triu(torch.ones(sequence_length, sequence_length, device=sequence.device), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add dimensions to be broadcastable to batch x heads x seq x seq
        return mask

    def decoder_step(self, decoder_input: Tensor, cyclic_inputs: Tuple[()],
                     constant_inputs: Tuple[Tensor, Tensor, nn.Embedding]) -> Tuple[Tensor, Tuple[()], Tuple[()]]:
        encoder_out, encoder_mask, embedding = constant_inputs

        outputs_mask = Transformer.get_padding_mask(decoder_input, self.padding_index)
        outputs_mask = outputs_mask | self.__get_decoder_mask(decoder_input)
        outputs_embedded = embedding(decoder_input)
        outputs_embedded = self.layer_norm(outputs_embedded)
        decoder_out, _, _, _ = self.decoders(outputs_embedded, outputs_mask, encoder_out, encoder_mask)

        return self.out(decoder_out), (), ()

    def forward(self, outputs: Optional[Tensor], embedding: nn.Embedding, teacher_forcing_ratio: float, batch_size: int,
                device: str, cyclic_inputs: Tuple[()],
                constant_inputs: Tuple[Tensor, Tensor, nn.Embedding]) -> Tuple[Tensor, Tensor, List[None]]:
        outputs, teacher_forcing_ratio = self._validate_outputs(outputs, teacher_forcing_ratio, batch_size, device)

        # Custom forward for this model depending on phase
        if self.training and teacher_forcing_ratio == 1.0:
            predictions, _, _ = self.decoder_step(outputs, cyclic_inputs, constant_inputs)
            tokens = self._get_predicted_tokens(predictions)
        else:
            predictions = None
            for i in range(self.max_output_length):
                decoder_input = outputs[:i + 1, :]
                predictions_step, _, _ = self.decoder_step(decoder_input, cyclic_inputs, constant_inputs)
                predictions = predictions_step
                if i + 1 < self.max_output_length:
                    outputs[1:i + 2, :] = self._get_predicted_tokens(predictions_step)
            tokens = self._get_predicted_tokens(predictions)

        return predictions, tokens, []


class Transformer(nn.Module):
    def __init__(self, encoder_layers: int, decoder_layers: int, vocab_size: int, embedding_dim: int,
                 key_and_query_dim: int, value_dim: int, heads_number: int, feed_forward_size: int,
                 dropout_rate: float, max_summary_length: int, bos_index: int, eos_index: int, padding_index: int = 0):
        super().__init__()
        self.max_summary_length = max_summary_length
        self.bos_index = bos_index
        self.padding_index = padding_index
        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_index),
            PositionalEncoding(embedding_dim),
            nn.Dropout(dropout_rate)
        )
        self.encoder = Encoder(encoder_layers, embedding_dim, key_and_query_dim, value_dim, heads_number,
                               feed_forward_size, dropout_rate)
        self.decoder = Decoder(vocab_size, decoder_layers, embedding_dim, key_and_query_dim, value_dim, heads_number,
                               feed_forward_size, dropout_rate, bos_index, eos_index, padding_index, max_summary_length,
                               self.embedding[0].weight)

    @staticmethod
    def get_padding_mask(sequence, padding_index):
        mask = sequence == padding_index
        mask = torch.as_tensor(mask, dtype=torch.bool, device=sequence.device)
        mask = mask.transpose(0, 1)
        mask = mask.unsqueeze(1).unsqueeze(1)  # Add dimensions to be broadcastable to batch x heads x seq x seq
        return mask

    def forward(self, inputs: Tensor, outputs: Tensor = None) -> Tuple[Tensor, Tensor]:
        device = inputs.device
        batch_size = inputs.shape[1]
        teacher_forcing_ratio = 1.  # In this model teacher forcing is always used

        inputs_mask = self.get_padding_mask(inputs, self.padding_index)
        inputs_embedded = self.embedding(inputs)

        encoder_out = self.encoder(inputs_embedded, inputs_mask)
        out, tokens, _ = self.decoder(outputs, self.embedding, teacher_forcing_ratio, batch_size, device, (),
                                      constant_inputs=(encoder_out, inputs_mask, self.embedding))
        return out, tokens
