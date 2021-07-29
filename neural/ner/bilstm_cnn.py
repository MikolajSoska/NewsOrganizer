from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

import neural.common.layers as layers


class BiLSTMConv(nn.Module):
    def __init__(self, output_size: int, conv_width: int, conv_output_size: int, hidden_size: int, lstm_layers: int,
                 dropout_rate: float, char_vocab_size: int, char_embedding_dim: int, word_vocab_size: int = None,
                 word_embedding_dim: int = None, embeddings: Tensor = None, use_word_features: bool = False,
                 use_char_features: bool = False):
        if embeddings is None and (word_embedding_dim is None or word_vocab_size is None):
            raise ValueError('Either embeddings vector or embedding dim and vocab size must be passed.')

        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.conv_width = conv_width

        self.char_embedding = nn.Sequential(
            nn.Flatten(start_dim=0, end_dim=1),
            nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=0),
        )

        if use_char_features:
            self.char_features = nn.Sequential(
                nn.Flatten(start_dim=0, end_dim=1),
                nn.Embedding(num_embeddings=5, embedding_dim=4, padding_idx=0)
            )
            char_embedding_dim += 4  # Add size for additional char features
        else:
            self.char_features = None

        self.char_nn = nn.Sequential(
            layers.Permute(0, 2, 1),
            nn.Conv1d(char_embedding_dim, conv_output_size, kernel_size=conv_width),
            layers.Max(dimension=-1)
        )

        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(word_vocab_size, word_embedding_dim, padding_idx=0)

        if use_word_features:
            self.word_features = nn.Embedding(num_embeddings=6, embedding_dim=5, padding_idx=0)
            conv_output_size += 5  # Add size for additional word features
        else:
            self.word_features = None

        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim + conv_output_size, hidden_size=hidden_size,
                            num_layers=lstm_layers, bidirectional=True, dropout=dropout_rate if lstm_layers > 1 else 0)

        self.forward_out = layers.TimeDistributed(nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.hidden_size, output_size),
            nn.LogSoftmax(dim=1)
        ))

        self.backward_out = layers.TimeDistributed(nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.hidden_size, output_size),
            nn.LogSoftmax(dim=1)
        ))

    def __init_hidden(self, batch_size: int, device: str) -> Tuple[Tensor, Tensor]:
        hidden_shape = (2 * self.lstm_layers, batch_size, self.hidden_size)
        return torch.zeros(hidden_shape, device=device), torch.zeros(hidden_shape, device=device)

    def forward(self, sentences_in: Tensor, chars_in: Tensor, word_features_in: Tensor,
                char_features_in: Tensor) -> Tensor:
        sequence_length, batch_size = sentences_in.shape

        char_features = self.char_embedding(chars_in)
        if self.char_features is not None:
            additional_features = self.char_features(char_features_in)
            char_features = torch.cat((char_features, additional_features), dim=-1)

        char_features = self.char_nn(char_features)
        char_features = char_features.view((sequence_length, batch_size, -1))

        sentences = self.embedding(sentences_in)
        if self.word_features is not None:
            additional_features = self.word_features(word_features_in)
            sentences = torch.cat((sentences, additional_features, char_features), dim=-1)
        else:
            sentences = torch.cat((sentences, char_features), dim=-1)

        hidden, cell = self.__init_hidden(batch_size, sentences_in.device)
        sentences, _ = self.lstm(sentences, (hidden, cell))
        forward = sentences[:, :, :self.hidden_size]
        backward = sentences[:, :, self.hidden_size:]

        forward = self.forward_out(forward)
        backward = self.backward_out(backward)

        return forward + backward
