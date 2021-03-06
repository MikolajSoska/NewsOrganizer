from __future__ import annotations

from typing import Tuple, Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

import neural.common.layers as layers
from neural.common.layers.decode import CRF
from neural.common.model import BaseModel


class BiLSTMCRF(BaseModel):
    def __init__(self, output_size: int, char_hidden_size: int, word_hidden_size: int, char_vocab_size: int,
                 char_embedding_dim: int, dropout_rate: float, word_vocab_size: int = None,
                 word_embedding_dim: int = None, embeddings: Tensor = None):
        if embeddings is None and (word_embedding_dim is None or word_vocab_size is None):
            raise ValueError('Either embeddings vector or embedding dim and vocab size must be passed.')

        super().__init__()
        self.char_hidden_size = char_hidden_size // 2
        self.word_hidden_size = word_hidden_size // 2

        self.char_embedding = nn.Sequential(
            nn.Flatten(start_dim=0, end_dim=1),
            layers.Transpose(0, 1),
            nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=0),
        )

        if embeddings is not None:
            self.word_embedding = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=0)
            word_embedding_dim = self.word_embedding.embedding_dim
        else:
            self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim, padding_idx=0)

        self.final_embedding = layers.SequentialMultiInput(
            layers.Concatenate(dimension=-1),
            nn.Dropout(dropout_rate)
        )
        self.char_lstm = nn.LSTM(input_size=char_embedding_dim, hidden_size=char_hidden_size // 2, bidirectional=True)
        self.word_lstm = nn.LSTM(input_size=word_embedding_dim + char_hidden_size, hidden_size=word_hidden_size // 2,
                                 bidirectional=True)
        self.out = nn.Linear(word_hidden_size, output_size)
        self.crf = CRF(output_size)

    @classmethod
    def create_from_args(cls, args: Dict[str, Any], tags_count: int = None, word_vocab_size: int = None,
                         char_vocab_size: int = None, embeddings: Tensor = None) -> BiLSTMCRF:
        assert tags_count is not None, 'Tags count can\'t be None'
        assert word_vocab_size is not None, 'Words vocab size can\'t be None'
        assert char_vocab_size is not None, 'Chars vocab size can\'t be None'

        return cls(
            output_size=tags_count,
            char_hidden_size=args['char_lstm_hidden'],
            word_hidden_size=args['word_lstm_hidden'],
            char_vocab_size=char_vocab_size,
            char_embedding_dim=args['char_embedding_size'],
            dropout_rate=args['dropout'],
            word_vocab_size=word_vocab_size,
            word_embedding_dim=args['word_embedding_size'],
            embeddings=embeddings
        )

    def predict(self, *inputs: Any) -> Tensor:
        words, chars, word_features, char_features = inputs
        mask = (words > 0).float()

        _, predictions = self(words, chars, mask)
        return predictions

    @staticmethod
    def __init_hidden(hidden_dim: int, batch_size: int, device: str) -> Tuple[Tensor, Tensor]:
        hidden_shape = (2, batch_size, hidden_dim)
        return torch.randn(hidden_shape, device=device), torch.randn(hidden_shape, device=device)

    def forward(self, sentences_in: Tensor, chars_in: Tensor, mask: Tensor,
                tags: Tensor = None) -> Tuple[Optional[Tensor], Tensor]:
        if self.training and tags is None:
            raise ValueError('Target tags must be provided during training.')

        sequence_length, batch_size = sentences_in.shape
        device = sentences_in.device

        char_hidden = self.__init_hidden(self.char_hidden_size, sequence_length * batch_size, device)
        word_hidden = self.__init_hidden(self.word_hidden_size, batch_size, device)

        char_features = self.char_embedding(chars_in)
        char_features, _ = self.char_lstm(char_features, char_hidden)
        char_features = char_features[-1, :, :]  # Take only last sequences as char features
        char_features = char_features.view((sequence_length, batch_size, -1))

        word_features = self.word_embedding(sentences_in)
        word_features = self.final_embedding(word_features, char_features)
        word_features, _ = self.word_lstm(word_features, word_hidden)
        predictions = self.out(word_features)

        if self.training:
            score = self.crf(predictions, tags, mask)
        else:
            score = None
        predictions = self.crf.decode(predictions, mask)

        return score, predictions
