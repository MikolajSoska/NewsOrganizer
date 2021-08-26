from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

import neural.common.layers as layers


class Encoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.lstm = layers.PackedRNN(nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=True))
        self.features = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)
        )
        self.reduce_hidden = nn.Sequential(
            layers.Transpose(0, 1),
            layers.Contiguous(),
            layers.View(-1, hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )
        self.reduce_cell = nn.Sequential(
            layers.Transpose(0, 1),
            layers.Contiguous(),
            layers.View(-1, hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )

    def forward(self, inputs: Tensor, inputs_lengths: Tensor) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
        out, (hidden, cell) = self.lstm(inputs, inputs_lengths)
        features = self.features(out)
        out = out.transpose(0, 1)

        hidden = self.reduce_hidden(hidden)
        cell = self.reduce_cell(cell)

        return out, features, (hidden, cell)


class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            layers.Unsqueeze(0),
        )
        self.coverage = nn.Sequential(
            layers.Unsqueeze(-1),
            nn.Linear(1, 2 * hidden_size, bias=False)
        )
        self.attention_first = nn.Sequential(
            nn.Tanh(),
            nn.Linear(2 * hidden_size, 1, bias=False),
            layers.Squeeze(),
            nn.Softmax(dim=0)
        )
        self.attention_second = nn.Sequential(
            # If encoder_mask sets some weights to zero, than they don't sum to one, so re-normalization is needed
            layers.Normalize(0),
            layers.Permute(1, 0),
            layers.Unsqueeze(1)
        )
        self.attention_out = nn.Sequential(
            layers.Squeeze(),
            layers.Transpose(0, 1)
        )
        self.context = layers.SequentialMultiInput(
            layers.MatrixProduct(),
            layers.View(-1, 2 * hidden_size)
        )

    def forward(self, decoder_hidden: Tensor, encoder_out: Tensor, encoder_features: Tensor, encoder_mask: Tensor,
                coverage: Optional[Tensor]) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        decoder_features = self.features(decoder_hidden)
        attention = encoder_features + decoder_features

        if coverage is not None:  # If training with coverage
            coverage_features = self.coverage(coverage)
            attention = attention + coverage_features

        attention = self.attention_first(attention)
        attention = attention * encoder_mask
        attention = self.attention_second(attention)

        context = self.context(attention, encoder_out)
        attention = self.attention_out(attention)

        if coverage is not None:
            coverage = coverage + attention

        return context, attention, coverage


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTMCell(input_size=embedding_dim, hidden_size=hidden_size)
        self.context = nn.Sequential(
            nn.Linear(hidden_size * 2 + embedding_dim, embedding_dim)
        )
        self.pointer_generator = layers.SequentialMultiInput(
            layers.Concatenate(1),
            nn.Linear(hidden_size * 4 + embedding_dim, 1),
            nn.Sigmoid()
        )
        self.out = layers.SequentialMultiInput(
            layers.Concatenate(1),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Linear(hidden_size, vocab_size),
            nn.Softmax(dim=1)
        )

    def forward(self, outputs: Tensor, encoder_hidden: Tuple[Tensor, Tensor], encoder_out: Tensor,
                encoder_features: Tensor, encoder_mask: Tensor, previous_context: Tensor, coverage: Optional[Tensor],
                texts_extended: Tensor, oov_size: int) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor, Tensor,
                                                                Optional[Tensor]]:
        outputs = torch.cat((previous_context, outputs), dim=1)
        outputs = self.context(outputs)
        hidden, cell = self.lstm(outputs, encoder_hidden)

        decoder_hidden = torch.cat((hidden, cell), dim=1)
        context, attention, coverage = self.attention(decoder_hidden, encoder_out, encoder_features, encoder_mask,
                                                      coverage)

        p_gen = self.pointer_generator(context, decoder_hidden, outputs)
        out = self.out(decoder_hidden, context)
        vocab_distribution = p_gen * out
        oov_attention = (1 - p_gen).permute(1, 0) * attention
        if oov_size > 0:
            batch_size = vocab_distribution.shape[0]
            oov_zeros = torch.zeros((batch_size, oov_size), device=vocab_distribution.device)
            vocab_distribution = torch.cat((vocab_distribution, oov_zeros), dim=1)

        final = torch.scatter_add(vocab_distribution, 1, texts_extended.permute(1, 0), oov_attention.permute(1, 0))

        return final, (hidden, cell), context, attention, coverage


class PointerGeneratorNetwork(nn.Module):
    def __init__(self, vocab_size: int, bos_index: int, unk_index: int, embedding_dim: int, hidden_size: int,
                 max_summary_length: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_summary_length = max_summary_length
        self.bos_index = bos_index
        self.unk_index = unk_index
        self.with_coverage = False  # Coverage is active only during last phase of training
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = Encoder(embedding_dim, hidden_size)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_size)

    def activate_coverage(self):
        self.with_coverage = True

    def forward(self, inputs: Tensor, inputs_lengths: Tensor, inputs_extended: Tensor,
                oov_size: int, outputs: Tensor = None) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        device = inputs.device
        batch_size = inputs.shape[1]

        if self.training:
            if outputs is None:
                raise AttributeError('During training reference outputs must be provided.')
        else:  # In validation phase never use passed summaries
            outputs = torch.full((self.max_summary_length, batch_size), self.bos_index, dtype=torch.long, device=device)

        inputs_embedded = self.embedding(inputs)
        encoder_out, encoder_features, encoder_hidden = self.encoder(inputs_embedded, inputs_lengths)
        encoder_mask = torch.clip(inputs, min=0, max=1)

        context = torch.zeros((batch_size, 2 * self.hidden_size), device=device)
        if self.with_coverage:
            coverage = torch.zeros_like(inputs, device=device, dtype=torch.float)
        else:
            coverage = None

        outputs_list = []
        attention_list = []
        coverage_list = []

        for i in range(self.max_summary_length):
            decoder_input = outputs[i, :]
            if not self.training:  # Remove OOV tokens in validation phase
                decoder_input[decoder_input >= self.vocab_size] = self.unk_index
            outputs_embedded = self.embedding(decoder_input)
            decoder_out, encoder_hidden, context, attention, coverage = self.decoder(outputs_embedded, encoder_hidden,
                                                                                     encoder_out, encoder_features,
                                                                                     encoder_mask, context, coverage,
                                                                                     inputs_extended, oov_size)
            outputs_list.append(decoder_out)
            attention_list.append(attention)
            coverage_list.append(coverage)

            if not self.training and i + 1 < self.max_summary_length:
                outputs[i + 1, :] = torch.argmax(decoder_out, dim=1)

        outputs = torch.stack(outputs_list)
        attentions = torch.stack(attention_list)
        if self.with_coverage:
            coverages = torch.stack(coverage_list)
        else:
            coverages = None

        return outputs, attentions, coverages
