from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

import neural.model.utils as utils


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = utils.PackedRNN(nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=True))
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

    def forward(self, texts_in: Tensor, texts_lengths: Tensor) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
        texts = self.embedding(texts_in)
        texts, (hidden, cell) = self.lstm(texts, texts_lengths)
        texts = texts.contiguous()
        texts_out = self.linear(texts)
        texts = texts.permute(1, 0, 2)

        hidden = hidden.transpose(0, 1).contiguous()
        hidden = self.reduce_hidden(hidden)
        cell = cell.transpose(0, 1).contiguous()
        cell = self.reduce_cell(cell)

        return texts, texts_out, (hidden, cell)


class Attention(nn.Module):
    def __init__(self, hidden_size: int):
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
        self.attention_first = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_size * 2, 1),
        )
        self.softmax = nn.Softmax(dim=0)
        self.attention_second = nn.Sequential(
            # If encoder_mask sets some weights to zero, than they don't sum to one, so re-normalization is needed
            utils.Normalize(0),
            utils.Permute(1, 0),
            utils.Unsqueeze(1)
        )

    def forward(self, hidden: Tensor, encoder_out: Tensor, encoder_features: Tensor, encoder_mask: Tensor,
                coverage: Optional[Tensor]) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        batch, sequence_len, hidden_size = encoder_out.shape

        decoder_features = self.features(hidden)
        decoder_features = decoder_features.expand(sequence_len, batch, hidden_size).contiguous().view(-1, hidden_size)
        attention = encoder_features + decoder_features

        if coverage is not None:  # If training with coverage
            coverage_features = self.coverage(coverage)
            attention = attention + coverage_features

        attention = self.attention_first(attention).view(sequence_len, batch)
        attention = self.softmax(attention) * encoder_mask
        attention = self.attention_second(attention)

        context = torch.bmm(attention, encoder_out).view(-1, 2 * self.hidden_size)
        attention = attention.view(-1, sequence_len).permute(1, 0)

        if coverage is not None:
            coverage = coverage + attention

        return context, attention, coverage


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = Attention(hidden_size)
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        # No need for PackedRNN because sequence length is always one
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size)
        self.context = nn.Sequential(
            nn.Linear(hidden_size * 2 + embedding_dim, embedding_dim),
            utils.Unsqueeze(0)
        )
        self.pointer_generator = nn.Sequential(
            nn.Linear(hidden_size * 4 + embedding_dim, 1),
            nn.Sigmoid()
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.Linear(hidden_size, vocab_size),
            nn.Softmax(dim=1)
        )

    def forward(self, summaries_in: Tensor, hidden_in: Tuple[Tensor, Tensor], encoder_out: Tensor,
                encoder_features: Tensor, encoder_mask: Tensor, context: Tensor,
                coverage: Optional[Tensor], texts_extended: Tensor, oov_size: int) -> Tuple[Tensor, Tensor, Tensor,
                                                                                            Tensor, Optional[Tensor]]:
        x = self.embedding(summaries_in)
        x = torch.cat((context, x), dim=1)
        context_out = self.context(x)
        x, (hidden, cell) = self.lstm(context_out, hidden_in)

        hidden = hidden.view(-1, self.hidden_size)
        cell = cell.view(-1, self.hidden_size)
        hidden_out = torch.cat((hidden, cell), dim=1)
        context, attention, coverage_next = self.attention(hidden_out, encoder_out, encoder_features, encoder_mask,
                                                           coverage)

        generator_in = torch.cat((context, hidden_out, context_out.squeeze(0)), dim=1)
        p_gen = self.pointer_generator(generator_in)

        x = x.view(-1, self.hidden_size)
        x = torch.cat((x, context), dim=1)
        x = self.out(x)

        vocab_dist = p_gen * x
        attention = (1 - p_gen).permute(1, 0) * attention
        if oov_size > 0:
            batch_size = vocab_dist.shape[0]
            device = vocab_dist.device
            vocab_dist = torch.cat((vocab_dist, torch.zeros((batch_size, oov_size), device=device)), dim=1)

        vocab_dist = vocab_dist.scatter_add_(1, texts_extended.permute(1, 0), attention.permute(1, 0))

        return vocab_dist, hidden_out, context, attention, coverage_next


class PointerGeneratorNetwork(nn.Module):
    def __init__(self, vocab_size: int, bos_index: int, embedding_dim: int = 128, hidden_size: int = 256,
                 max_summary_length: int = 100):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_summary_length = max_summary_length
        self.bos_index = bos_index
        self.with_coverage = False  # Coverage is active only during last phase of training
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_size)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_size)

    def activate_coverage(self):
        self.with_coverage = True

    def forward(self, texts: Tensor, texts_lengths: Tensor, texts_extended: Tensor,
                oov_size: int, summaries: Tensor = None) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        if summaries is None:
            if self.training:
                raise AttributeError('During training reference summaries must be provided.')
            else:
                summaries = torch.full((self.max_summary_length, texts.shape[1]), self.bos_index, dtype=torch.long,
                                       device=texts.device)

        encoder_out, encoder_features, hidden = self.encoder(texts, texts_lengths)
        encoder_mask = torch.clip(texts, min=0, max=1)
        device = texts.device
        batch_size = texts.shape[1]

        context = torch.zeros((batch_size, 2 * self.hidden_size), device=device)
        if self.with_coverage:
            coverage = torch.zeros_like(texts, device=device, dtype=torch.float)
        else:
            coverage = None

        outputs = []
        attention_list = []
        coverage_list = []

        for i in range(summaries.shape[0]):
            decoder_input = summaries[i, :]
            decoder_out, decoder_hidden, context, attention, coverage = self.decoder(decoder_input, hidden, encoder_out,
                                                                                     encoder_features, encoder_mask,
                                                                                     context, coverage, texts_extended,
                                                                                     oov_size)
            outputs.append(decoder_out)
            attention_list.append(attention)
            coverage_list.append(coverage)

            if not self.training and i + 1 < summaries.shape[0]:
                summaries[i + 1, :] = torch.argmax(decoder_out, dim=1)

        outputs = torch.stack(outputs)
        attentions = torch.stack(attention_list)
        if self.with_coverage:
            coverages = torch.stack(coverage_list)
        else:
            coverages = None

        return outputs, attentions, coverages
