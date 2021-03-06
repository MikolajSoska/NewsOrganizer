from __future__ import annotations

from typing import Tuple, Dict, Optional, Any

import torch
import torch.nn as nn
from torch import Tensor

import neural.common.layers as layers
from neural.common.data.vocab import SpecialTokens
from neural.common.layers.decode import BeamSearchDecoder
from neural.common.model import BaseModel


class Encoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.lstm = layers.PackedRNN(nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=True))
        self.features = nn.Sequential(
            nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
        )
        self.reduce_hidden = nn.Sequential(
            layers.Transpose(0, 1),
            layers.Contiguous(),
            layers.View(-1, 2 * hidden_size),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU()
        )
        self.reduce_cell = nn.Sequential(
            layers.Transpose(0, 1),
            layers.Contiguous(),
            layers.View(-1, 2 * hidden_size),
            nn.Linear(2 * hidden_size, hidden_size),
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
            layers.Transpose(0, 1),
            layers.Unsqueeze(-1),
            nn.Linear(1, 2 * hidden_size, bias=False)
        )
        self.attention_first = nn.Sequential(
            nn.Tanh(),
            nn.Linear(2 * hidden_size, 1, bias=False),
            layers.Squeeze(-1),
            nn.Softmax(dim=0)
        )
        self.attention_second = nn.Sequential(
            # If encoder_mask sets some weights to zero, than they don't sum to one, so re-normalization is needed
            layers.Normalize(0),
            layers.Transpose(0, 1),
            layers.Unsqueeze(1)
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
        attention = torch.squeeze(attention)

        if coverage is not None:
            coverage = coverage + attention

        return context, attention, coverage


class Decoder(BeamSearchDecoder):
    def __init__(self, embedding_dim: int, vocab_size: int, hidden_size: int, max_summary_length, bos_index: int,
                 eos_index: int, unk_index: int, beam_size: int):
        super().__init__(bos_index, eos_index, max_summary_length, beam_size)
        self.unk_index = unk_index
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTMCell(input_size=embedding_dim, hidden_size=hidden_size)
        self.context = nn.Sequential(
            nn.Linear(2 * hidden_size + embedding_dim, embedding_dim)
        )
        self.pointer_generator = layers.SequentialMultiInput(
            layers.Concatenate(1),
            nn.Linear(4 * hidden_size + embedding_dim, 1),
            nn.Sigmoid()
        )
        self.out = layers.SequentialMultiInput(
            layers.Concatenate(1),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Linear(hidden_size, vocab_size),
            nn.Softmax(dim=1)
        )

    def decoder_step(self, decoder_input: Tensor, cyclic_inputs: Tuple[Tuple[Tensor, Tensor], Tensor, Tensor],
                     constant_inputs: Tuple[Tensor, Tensor, Tensor, Tensor, int]) -> Tuple[Tensor, Tuple[Any, ...],
                                                                                           Tuple[Any, ...]]:
        encoder_hidden, previous_context, coverage = cyclic_inputs
        encoder_out, encoder_features, encoder_mask, inputs_extended, oov_size = constant_inputs

        outputs = torch.cat((previous_context, decoder_input), dim=1)
        outputs = self.context(outputs)
        hidden, cell = self.lstm(outputs, encoder_hidden)

        decoder_hidden = torch.cat((hidden, cell), dim=1)
        context, attention, coverage = self.attention(decoder_hidden, encoder_out, encoder_features, encoder_mask,
                                                      coverage)

        p_gen = self.pointer_generator(context, decoder_hidden, outputs)
        out = self.out(decoder_hidden, context)
        vocab_distribution = p_gen * out
        oov_attention = (1 - p_gen) * attention
        if oov_size > 0:
            batch_size = vocab_distribution.shape[0]
            oov_zeros = torch.zeros((batch_size, oov_size), device=vocab_distribution.device)
            vocab_distribution = torch.cat((vocab_distribution, oov_zeros), dim=1)

        final = torch.scatter_add(vocab_distribution, 1, inputs_extended.transpose(0, 1), oov_attention)

        return final, ((hidden, cell), context, coverage), (attention, coverage)

    def _preprocess_decoder_inputs(self, decoder_inputs: Tensor) -> Tensor:
        if not self.training:  # Remove OOV tokens in validation phase
            decoder_inputs[decoder_inputs >= self.vocab_size] = self.unk_index

        return decoder_inputs


class PointerGeneratorNetwork(BaseModel):
    def __init__(self, vocab_size: int, bos_index: int, eos_index: int, unk_index: int, embedding_dim: int,
                 hidden_size: int, max_summary_length: int, beam_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.with_coverage = False  # Coverage is active only during last phase of training
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = Encoder(embedding_dim, hidden_size)
        self.decoder = Decoder(embedding_dim, vocab_size, hidden_size, max_summary_length, bos_index, eos_index,
                               unk_index, beam_size)

    @classmethod
    def create_from_args(cls, args: Dict[str, Any], bos_index: int = None, eos_index: int = None,
                         unk_index: int = None) -> PointerGeneratorNetwork:
        assert bos_index is not None, 'BOS index can\'t be None'
        assert eos_index is not None, 'EOS index can\'t be None'
        assert unk_index is not None, 'UNK index can\'t be None'

        return cls(
            vocab_size=args['vocab_size'] + len(SpecialTokens),
            bos_index=bos_index,
            eos_index=eos_index,
            unk_index=unk_index,
            embedding_dim=args['embedding_size'],
            hidden_size=args['hidden_size'],
            max_summary_length=args['max_summary_length'],
            beam_size=args['beam_size']
        )

    def predict(self, *inputs: Any) -> Tensor:
        self.activate_coverage()
        texts, texts_lengths, texts_extended, oov_list = inputs
        oov_size = len(max(oov_list, key=lambda x: len(x)))
        _, tokens, _, _ = self(texts, texts_lengths, texts_extended, oov_size)

        return tokens

    def activate_coverage(self):
        self.with_coverage = True

    def forward(self, inputs: Tensor, inputs_lengths: Tensor, inputs_extended: Tensor,
                oov_size: int, outputs: Tensor = None) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        device = inputs.device
        batch_size = inputs.shape[1]
        teacher_forcing_ratio = 1.  # In this model teacher forcing is used in every step

        inputs_embedded = self.embedding(inputs)
        encoder_out, encoder_features, encoder_hidden = self.encoder(inputs_embedded, inputs_lengths)
        encoder_mask = torch.clip(inputs, min=0, max=1)

        context = torch.zeros((batch_size, 2 * self.hidden_size), device=device)
        if self.with_coverage:
            coverage = torch.zeros_like(inputs, device=device, dtype=torch.float)
            coverage = torch.transpose(coverage, 0, 1)
        else:
            coverage = None

        outputs, tokens, decoder_outputs = self.decoder(outputs, self.embedding, teacher_forcing_ratio, batch_size,
                                                        device, cyclic_inputs=(encoder_hidden, context, coverage),
                                                        constant_inputs=(encoder_out, encoder_features, encoder_mask,
                                                                         inputs_extended, oov_size))
        attention_list, coverage_list = zip(*decoder_outputs)
        attentions = torch.stack(attention_list)
        if self.with_coverage:
            coverages = torch.stack(coverage_list)
        else:
            coverages = None

        return outputs, tokens, attentions, coverages
