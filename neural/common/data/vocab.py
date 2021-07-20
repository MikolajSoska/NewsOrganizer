import enum
import os
from collections import Counter
from typing import List, Tuple, Iterator

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


class VocabWithChars(Vocab):
    def __init__(self, words_counter: Counter, chars_counter: Counter, max_size: int = None):
        super().__init__(words_counter, max_size)
        self.chars = Vocab(chars_counter, specials=[SpecialTokens.PAD.value, SpecialTokens.UNK.value])


class VocabBuilder:
    @classmethod
    def build_vocab(cls, dataset_name: str, vocab_name: str, vocab_type: str = 'base', vocab_size: int = None,
                    vocab_dir: str = '../data/vocabs'):
        if vocab_type == 'base':
            builder = cls.__build_base_vocab
        elif vocab_type == 'char':
            builder = cls.__build_vocab_with_chars
        else:
            raise ValueError(f'Unrecognized vocab type: "{vocab_type}".')

        vocab_size_str = vocab_size or 'all'
        vocab_path = f'{vocab_dir}/vocab-{vocab_name}-{vocab_size_str}-{dataset_name}.pt'
        if os.path.exists(vocab_path):
            return torch.load(vocab_path)

        # Generate vocab from train split
        generator = DatasetGenerator.generate_dataset(dataset_name, 'train', for_vocab=True)
        vocab = builder(generator, vocab_size)
        torch.save(vocab, vocab_path)
        return vocab

    @classmethod
    def __build_base_vocab(cls, dataset_generator: Iterator[Tuple[List[str], ...]], vocab_size: int) -> Vocab:
        counter = Counter()
        for tokens, in dataset_generator:
            counter.update(token.lower() for token in tokens)

        return Vocab(counter, max_size=vocab_size, specials=SpecialTokens.get_tokens())

    @classmethod
    def __build_vocab_with_chars(cls, dataset_generator: Iterator[Tuple[List[str], ...]],
                                 vocab_size: int) -> VocabWithChars:
        words_counter = Counter()
        chars_counter = Counter()
        for tokens, in dataset_generator:
            words_counter.update(token.lower() for token in tokens)
            for token in tokens:
                chars_counter.update(list(token))

        return VocabWithChars(words_counter, chars_counter, vocab_size)
