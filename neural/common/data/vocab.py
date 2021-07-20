import enum
import os
from collections import Counter
from typing import List

import torch
from torchtext.vocab import Vocab

from neural.common.data.datasets import DatasetGenerator


class SpecialTokens(enum.Enum):
    PAD = '<pad>'
    UNK = '<unk>'
    BOS = '<bos>'
    EOS = '<eos>'

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.__str__()

    @classmethod
    def get_tokens(cls) -> List[str]:
        return [token.value for token in SpecialTokens]


def build_vocab(dataset_name: str, vocab_name: str, vocab_size: int = None, vocab_dir: str = '../data/vocabs') -> Vocab:
    vocab_size_str = vocab_size or 'all'
    vocab_path = f'{vocab_dir}/vocab-{vocab_name}-{vocab_size_str}-{dataset_name}.pt'
    if os.path.exists(vocab_path):
        return torch.load(vocab_path)

    generator = DatasetGenerator(dataset_name, 'train')  # Generate vocab from train split
    counter = Counter()
    for text_tokens, summary_tokens in generator.generate_dataset():
        counter.update(token.lower() for token in text_tokens + summary_tokens)

    vocab = Vocab(counter, max_size=vocab_size, specials=SpecialTokens.get_tokens())

    torch.save(vocab, vocab_path)
    return vocab
