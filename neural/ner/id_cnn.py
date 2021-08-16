from typing import List

import torch
import torch.nn as nn
from torch import Tensor

import neural.common.layers as layers


class IteratedDilatedCNN(nn.Module):
    def __init__(self, output_size: int, conv_width: int, conv_filters: int, dilation_sizes: List[int],
                 block_repeats: int, input_dropout: float, block_dropout: float, vocab_size: int = None,
                 embedding_dim: int = None, embeddings: Tensor = None, use_word_features: bool = False):
        if embeddings is None and (embedding_dim is None or vocab_size is None):
            raise ValueError('Either embeddings vector or embedding dim and vocab size must be passed.')

        super().__init__()
        self.block_repeats = block_repeats
        if embeddings is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding.from_pretrained(embeddings, padding_idx=0, freeze=False)
            embedding_dim = self.embedding.embedding_dim

        if use_word_features:
            self.word_features = nn.Embedding(num_embeddings=6, embedding_dim=5, padding_idx=0)
            embedding_dim += 5
        else:
            self.word_features = None

        conv_padding = int(conv_width / 2)
        self.first_conv = nn.Sequential(
            nn.Dropout(input_dropout),
            layers.Permute(1, 2, 0),
            nn.Conv1d(embedding_dim, conv_filters, kernel_size=conv_width, padding=conv_padding),
            nn.ReLU()
        )

        conv_block = []
        for dilation in dilation_sizes:
            conv_block.append(nn.Conv1d(conv_filters, conv_filters, conv_width, padding=conv_padding * dilation,
                                        dilation=dilation))
            conv_block.append(nn.ReLU())
            conv_block.append(nn.Dropout(block_dropout))

        self.conv_block = nn.Sequential(*conv_block)
        self.out = nn.Sequential(
            layers.Permute(2, 0, 1),
            nn.Linear(conv_filters, output_size)
        )

    def forward(self, sentences_in: Tensor, word_features_in: Tensor) -> List[Tensor]:
        embeddings = self.embedding(sentences_in)
        if self.word_features is not None:
            word_features = self.word_features(word_features_in)
            embeddings = torch.cat((embeddings, word_features), dim=-1)

        features = self.first_conv(embeddings)
        outputs = []
        for _ in range(self.block_repeats):
            features = self.conv_block(features)
            outputs.append(self.out(features))

        return outputs
