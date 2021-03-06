import enum
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Iterator, Union

import torch
from torchtext.vocab import Vocab

from neural.common.data.datasets import DatasetGenerator
from utils.general import preprocess_token


class SpecialTokens(enum.Enum):
    PAD = '<pad>'
    UNK = '<unk>'
    BOS = '<bos>'
    EOS = '<eos>'

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def get_tokens() -> List[str]:
        return [token.value for token in SpecialTokens]


class VocabWithChars(Vocab):
    def __init__(self, words_counter: Counter, chars_counter: Counter, specials: List[str], max_size: int = None):
        super().__init__(words_counter, max_size, specials=specials)
        self.chars = Vocab(chars_counter, specials=[SpecialTokens.PAD.value, SpecialTokens.UNK.value])


class VocabBuilder:
    @classmethod
    def build_vocab(cls, dataset_name: str, vocab_name: str, vocab_type: str = 'base', vocab_size: int = None,
                    lowercase: bool = True, digits_to_zero: bool = False,
                    vocab_dir: Union[Path, str] = '../data/saved/vocabs'):
        if isinstance(vocab_dir, str):
            vocab_dir = Path(vocab_dir)

        if vocab_type == 'base':
            builder = cls.__build_base_vocab
        elif vocab_type == 'char':
            builder = cls.__build_vocab_with_chars
        else:
            raise ValueError(f'Unrecognized vocab type: "{vocab_type}".')

        vocab_size_str = vocab_size or 'all'
        vocab_dir.mkdir(parents=True, exist_ok=True)
        vocab_path = vocab_dir / f'vocab-{vocab_name}-{vocab_type}-{vocab_size_str}-{dataset_name}.pt'
        if vocab_path.exists():
            return torch.load(vocab_path)

        # Generate vocab from train split
        generator = DatasetGenerator.generate_dataset(dataset_name, 'train', for_vocab=True)
        vocab = builder(generator, vocab_size, lowercase, digits_to_zero)
        torch.save(vocab, vocab_path)
        return vocab

    @staticmethod
    def __build_base_vocab(dataset_generator: Iterator[Tuple[List[str], ...]], vocab_size: int,
                           lowercase: bool, digits_to_zero: bool) -> Vocab:
        counter = Counter()
        for tokens, in dataset_generator:
            counter.update(preprocess_token(token, lowercase, digits_to_zero) for token in tokens)

        return Vocab(counter, max_size=vocab_size, specials=SpecialTokens.get_tokens())

    @staticmethod
    def __build_vocab_with_chars(dataset_generator: Iterator[Tuple[List[str], ...]], vocab_size: int,
                                 lowercase: bool, digits_to_zero: bool) -> VocabWithChars:
        words_counter = Counter()
        chars_counter = Counter()
        for tokens, in dataset_generator:
            words_counter.update(preprocess_token(token, lowercase, digits_to_zero) for token in tokens)
            for token in tokens:
                chars_counter.update(list(token))

        return VocabWithChars(words_counter, chars_counter, specials=SpecialTokens.get_tokens(), max_size=vocab_size)
