import itertools
import string
from pathlib import Path
from typing import List, Tuple, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, T_co

from neural.common.data.datasets import DatasetGenerator
from neural.common.data.vocab import SpecialTokens, VocabWithChars


class NERDataset(Dataset):
    def __init__(self, dataset_name: str, split: str, vocab: VocabWithChars,
                 data_dir: Union[Path, str] = '../data/saved/datasets'):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        self.__vocab = vocab
        self.__dataset = self.__build_dataset(dataset_name, split, data_dir)

    def __build_dataset(self, dataset_name: str, split: str, data_dir: Path) -> List[Tuple[Tensor, Tensor, List[Tensor],
                                                                                           Tensor, List[Tensor]]]:
        data_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = data_dir / f'dataset-{split}-ner-{dataset_name}-vocab-' \
                                  f'{len(self.__vocab) - len(SpecialTokens.get_tokens())}.pt'
        if dataset_path.exists():
            return torch.load(dataset_path)

        dataset = []
        generator = DatasetGenerator.generate_dataset(dataset_name, split)
        for tokens, tags in generator:
            words_tensor, word_types_tensor, char_list, char_types = self.process_tokens(tokens, self.__vocab)
            tags_tensor = torch.tensor(tags, dtype=torch.long)
            dataset.append((words_tensor, tags_tensor, char_list, word_types_tensor, char_types))
        torch.save(dataset, dataset_path)

        return dataset

    @classmethod
    def process_tokens(cls, tokens: List[str], vocab: VocabWithChars) -> Tuple[Tensor, Tensor, List[Tensor],
                                                                               List[Tensor]]:
        word_indexes = []
        word_types = []
        char_list = []
        char_types = []
        for word in tokens:
            word_indexes.append(vocab.stoi[word.lower()])
            word_types.append(cls.__get_word_type(word))
            char_tensor, types = cls.__process_word_chars(word, vocab)
            char_list.append(char_tensor)
            char_types.append(types)

        words_tensor = torch.tensor(word_indexes, dtype=torch.long)
        word_types_tensor = torch.tensor(word_types, dtype=torch.long)

        return words_tensor, word_types_tensor, char_list, char_types

    @classmethod
    def __process_word_chars(cls, word: str, vocab: VocabWithChars) -> Tuple[Tensor, Tensor]:
        indexes = []
        types = []
        for char in word:
            indexes.append(vocab.chars.stoi[char])
            types.append(cls.__get_char_type(char))

        return torch.tensor(indexes, dtype=torch.long), torch.tensor(types, dtype=torch.long)

    @staticmethod
    def __get_word_type(word: str) -> int:
        if word.isupper():
            return 1  # All caps
        elif word.istitle():
            return 2  # Upper initial
        elif word.lower():
            return 3  # All lower
        elif any(char.isupper() for char in word):
            return 4  # Mixed case
        else:
            return 5  # No info

    @staticmethod
    def __get_char_type(char: str) -> int:
        if char.isupper():
            return 1  # Upper case
        elif char.islower():
            return 2  # Lower case
        elif char in string.punctuation:
            return 3  # Punctuation
        else:
            return 4  # Other

    def __getitem__(self, index: int) -> T_co:
        return self.__dataset[index]

    def __len__(self) -> int:
        return len(self.__dataset)


class NERDataLoader(DataLoader):
    def __init__(self, dataset: NERDataset, batch_size: int, conv_kernel_size: int):
        super().__init__(dataset, batch_size, shuffle=True, collate_fn=self.__generate_batch)
        self.conv_kernel_size = conv_kernel_size

    def __generate_batch(self, batch: List[Tuple[Tensor, Tensor, List[Tensor], Tensor,
                                                 List[Tensor]]]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        words, tags, chars, word_features, char_features = zip(*batch)
        words_padded = pad_sequence(words)
        tags_padded = pad_sequence(tags)
        word_features_padded = pad_sequence(word_features)
        chars_padded = self.pad_char_sequence(chars, self.conv_kernel_size)
        char_features_padded = self.pad_char_sequence(char_features, self.conv_kernel_size)

        return words_padded, tags_padded, chars_padded, word_features_padded, char_features_padded

    @staticmethod
    def pad_char_sequence(char_sequence: Tuple[List[Tensor]], conv_kernel_size: int) -> Tensor:
        padded_sequences = []
        max_word_length = len(max(itertools.chain(*char_sequence), key=lambda item: len(item)))
        max_sequence_length = len(max(char_sequence, key=lambda item: len(item)))
        for chars in char_sequence:
            padded_chars = pad_sequence(chars)
            word_length, sequence_length = padded_chars.shape

            # Pad words to longest one + additional padding for longest word due to kernel size
            pad_length = max_word_length - word_length + 2 * (conv_kernel_size - 1)
            additional_padding = torch.zeros((pad_length, sequence_length), dtype=torch.long)
            padded_chars = torch.cat((padded_chars, additional_padding), dim=0)

            if conv_kernel_size > 1:  # Two-sided padding is only required when kernel_size is larger than 1
                for i, word in enumerate(chars):  # Center chars to make two-sided padding
                    roll_size = (word_length + pad_length - len(word)) // 2
                    padded_chars[:, i] = torch.roll(padded_chars[:, i], roll_size)

            # Pad sequences to longest one
            if max_sequence_length > sequence_length:
                padding_shape = (pad_length + word_length, max_sequence_length - sequence_length)
                sequence_padding = torch.zeros(padding_shape, dtype=torch.long)
                padded_chars = torch.cat((padded_chars, sequence_padding), dim=1)
            padded_sequences.append(padded_chars)

        return torch.stack(padded_sequences).permute(2, 0, 1)  # Sequence x Batch x Chars
