from typing import Tuple, Optional

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
                temporal_scores_sum: Tensor = None) -> Tuple[Tensor, Tensor, Tensor]:
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

        return context, attention, temporal_scores_sum


class IntraDecoderAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention_first = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            layers.Permute(1, 2, 0)
        )
        self.attention_second = layers.SequentialMultiInput(
            layers.MatrixProduct(),
            nn.Softmax(dim=-1)
        )
        self.context = layers.SequentialMultiInput(
            layers.MatrixProduct(),
            layers.Transpose(0, 1)
        )
        self.previous_hidden = layers.SequentialMultiInput(
            layers.Concatenate(1),
            layers.Transpose(0, 1)
        )

    def forward(self, decoder_hidden: Tensor, previous_hidden: Tensor = None) -> Tuple[Tensor, Tensor]:
        if previous_hidden is None:
            context = torch.zeros_like(decoder_hidden)
            return context, decoder_hidden

        decoder_hidden = torch.transpose(decoder_hidden, 0, 1)
        attention = self.attention_first(previous_hidden)
        previous_hidden = torch.transpose(previous_hidden, 0, 1)
        attention = self.attention_second(decoder_hidden, attention)
        context = self.context(attention, previous_hidden)
        previous_hidden = self.previous_hidden(previous_hidden, decoder_hidden)

        return context, previous_hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTMCell(input_size=embedding_dim, hidden_size=hidden_size)
        self.encoder_attention = IntraTemporalAttention()
        self.decoder_attention = IntraDecoderAttention(hidden_size)
        self.vocab_distribution = layers.SequentialMultiInput(
            layers.Concatenate(-1),
            layers.Squeeze(0),
            nn.Linear(3 * hidden_size, vocab_size),
            nn.Softmax(dim=1)
        )
        self.pointer_probability = layers.SequentialMultiInput(
            layers.Concatenate(-1),
            layers.Squeeze(0),
            nn.Linear(3 * hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, outputs: Tensor, encoder_out: Tensor, encoder_hidden: Tuple[Tensor],
                temporal_scores_sum: Optional[Tensor], previous_hidden: Optional[Tensor], texts_extended: Tensor,
                oov_size: int) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor, Tensor]:
        hidden, cell = self.lstm(outputs, encoder_hidden)
        decoder_hidden = torch.unsqueeze(hidden, 0)
        encoder_attention, temporal_attention, temporal_scores_sum = \
            self.encoder_attention(decoder_hidden, encoder_out, temporal_scores_sum)
        decoder_attention, previous_hidden = self.decoder_attention(decoder_hidden, previous_hidden)

        vocab_distribution = self.vocab_distribution(decoder_hidden, encoder_attention, decoder_attention)
        pointer_probability = self.pointer_probability(decoder_hidden, encoder_attention, decoder_attention)
        vocab_distribution = (1 - pointer_probability) * vocab_distribution
        attention = pointer_probability * temporal_attention.squeeze()

        if oov_size > 0:  # Add distribution for OOV words (with 0 value) to match dims
            batch_size = vocab_distribution.shape[0]
            device = vocab_distribution.device
            vocab_distribution = torch.cat((vocab_distribution, torch.zeros((batch_size, oov_size),
                                                                            device=device)), dim=1)

        final_distribution = torch.scatter_add(vocab_distribution, 1, texts_extended.permute(1, 0), attention)

        return final_distribution, (hidden, cell), temporal_attention, previous_hidden
