from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

import neural.common.layers as layers


class Encoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.lstm = layers.PackedRNN(nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size // 2,
                                             bidirectional=True))
        self.features = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            layers.Permute(1, 2, 0)
        )
        self.transform_state = layers.View(-1, hidden_size)

    def forward(self, inputs: Tensor, inputs_lengths: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        out, (hidden, cell) = self.lstm(inputs, inputs_lengths)
        out = self.features(out)
        hidden = self.transform_state(hidden)
        cell = self.transform_state(cell)

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


class ReinforcementSummarization(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_summary_length: int, bos_index: int, unk_index: int,
                 embedding_dim: int = None, embeddings: Tensor = None):
        if embeddings is None and embedding_dim is None:
            raise ValueError('Either embeddings vector or embedding dim must be passed.')

        super().__init__()
        self.vocab_size = vocab_size
        self.max_summary_length = max_summary_length
        self.bos_index = bos_index
        self.unk_index = unk_index

        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=0)
            embedding_dim = self.embedding.embedding_dim
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.encoder = Encoder(embedding_dim, hidden_size)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_size)

    def forward(self, inputs: Tensor, inputs_length: Tensor, inputs_extended: Tensor,
                oov_size: int, outputs: Tensor = None) -> Tensor:
        if self.training:
            if outputs is None:
                raise AttributeError('During training reference summaries must be provided.')
        else:  # In validation phase never use passed summaries
            outputs = torch.full((self.max_summary_length, inputs.shape[1]), self.bos_index, dtype=torch.long,
                                 device=inputs.device)

        inputs_embedded = self.embedding(inputs)
        encoder_out, encoder_hidden = self.encoder(inputs_embedded, inputs_length)

        temporal_attention = None
        previous_hidden = None
        predictions = []
        for i in range(outputs.shape[0]):
            decoder_input = outputs[i, :]
            if not self.training:  # Remove OOV tokens in validation phase
                decoder_input[decoder_input >= self.vocab_size] = self.unk_index

            decoder_input = self.embedding(decoder_input)
            prediction, encoder_hidden, temporal_attention, previous_hidden = \
                self.decoder(decoder_input, encoder_out, encoder_hidden, temporal_attention, previous_hidden,
                             inputs_extended, oov_size)
            predictions.append(prediction)

            if not self.training and i + 1 < outputs.shape[0]:
                outputs[i + 1, :] = torch.argmax(prediction, dim=1)

        return torch.stack(predictions)
