import itertools
import string
from collections import Counter
from pathlib import Path
from typing import List, Set, Tuple, Union

import pandas as pd
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, T_co
from torchtext.vocab import Vocab

from neural.common.data.datasets import DatasetGenerator
from neural.common.data.vocab import SpecialTokens, VocabWithChars


class NERDatasetNew(Dataset):
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


class NERDataLoaderNew(DataLoader):
    def __init__(self, dataset: NERDatasetNew, batch_size: int, conv_kernel_size: int):
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


class NERDataset(Dataset):
    def __init__(self, path_to_data: str, embedding: str):
        self.vocab = None
        self.sentences = None
        self.labels = None
        self.chars = None
        self.label_to_index = None
        self.labels_count = 0
        self.char_to_index = None
        self.char_count = 0
        self.max_word_length = 0

        data = pd.read_csv(path_to_data, encoding='unicode_escape')
        self.__create_vocab(data['Word'].to_list(), embedding)
        self.__initialize_dictionaries(data['Tag'].to_list(), data['Word'].to_list())
        self.__initialize_data(data)

    def get_label_name(self, label_index: int) -> str:  # TODO zamiast ten metody słownik idx2tag
        for key, value in self.label_to_index.items():
            if value == label_index:
                return key

    def __len__(self):
        return len(self.sentences)

    # TODO: dodać Error kiedy index nie będzie int, napisać ze dataset działa tylko dla batch równego 1
    def __getitem__(self, index):
        if torch.is_tensor(index):  # TODO czy tu przypadkiem to nie jest zawsze int?
            index = index.tolist()

        sentences = self.sentences[index]
        labels = self.labels[index]
        chars = self.chars[index]

        return sentences, labels, chars

    def __create_vocab(self, words: List[str], embedding: str) -> None:
        counter = Counter(set(words))
        self.vocab = Vocab(counter, vectors=embedding, specials=['<pad>', '<unk>'], vectors_cache='../.vector_cache')

    def __initialize_dictionaries(self, labels: List[str], words: Set[str]) -> None:
        self.label_to_index = {'<pad>': self.labels_count}
        self.labels_count += 1
        for label in labels:  # TODO to można w jednej linii zrobić przez enumerate, trzeba będzie to poprawić
            if label not in self.label_to_index:
                self.label_to_index[label] = self.labels_count
                self.labels_count += 1

        self.char_to_index = {'<pad>': 0, '<unk>': 1}
        self.char_count += 2

        for word in words:
            for char in word:
                if char not in self.char_to_index:
                    self.char_to_index[char] = self.char_count
                    self.char_count += 1
        self.max_word_length = len(max(words, key=lambda word: len(word)))

    def __initialize_data(self, data: pd.DataFrame) -> None:
        data['word_index'] = data['Word'].map(self.vocab.stoi)
        data['tag_index'] = data['Tag'].map(self.label_to_index)
        data['Char'] = data['Word'].map(lambda word: [char for char in word])
        data['char_index'] = data['Char'].map(lambda chars: [self.char_to_index[char] for char in chars])
        data = data.fillna(method='ffill', axis=0)
        data = data.groupby(['Sentence #'], sort=False).agg(list)

        data['sentence_length'] = data['Word'].map(lambda words: len(words))
        data = data.sort_values(by='sentence_length')
        data['word_index'] = data['word_index'].apply(torch.tensor)
        data['tag_index'] = data['tag_index'].apply(torch.tensor)

        self.sentences = data['word_index'].to_numpy()
        self.labels = data['tag_index'].to_numpy()
        self.chars = data['char_index'].to_numpy()


class NERDataLoader(DataLoader):
    def __init__(self, dataset: NERDataset, batch_size: int):
        super().__init__(dataset, batch_size=batch_size, shuffle=True, collate_fn=self.__generate_batch, drop_last=True)
        self.max_word_length = dataset.max_word_length

    def __generate_batch(self, batch):
        sentences, labels, chars = zip(*batch)
        sentences = pad_sequence(sentences)
        labels = pad_sequence(labels)
        chars = self.pad_chars_sequence(chars, sentences.shape[0])

        return sentences, labels, chars

    def pad_chars_sequence(self, chars_sequence, sentence_max_length: int):
        padded_sequences = []
        for sequence in chars_sequence:
            padded = []
            for chars in sequence:
                padded.append(self.pad_single_sequence(chars))
            for _ in range(sentence_max_length - len(padded)):
                padded.append(torch.zeros(self.max_word_length, dtype=int))
            padded_sequences.append(torch.stack(padded, dim=1))

        return torch.stack(padded_sequences, dim=0).permute(2, 0, 1)

    def pad_single_sequence(self, chars):
        forward_pad_length = (self.max_word_length - len(chars)) // 2
        backward_pad_length = forward_pad_length + (self.max_word_length - len(chars)) % 2

        return torch.cat([torch.zeros(forward_pad_length, dtype=int), torch.tensor(chars),
                          torch.zeros(backward_pad_length, dtype=int)], dim=0)


# TODO: cyfry chyba trzeba będzie preprocesować (chociaż to chyba zależy od datasetu i tagów jakie tam są)
class Vocabulary:
    def __init__(self, words_set: Set[str]):
        self.word_to_index = {}
        self.label_to_index = {}
        self.char_to_index = {}
        self.vocab_size = 0
        self.chars_size = 0

        self.__initialize_vocabulary()
        self.__create_vocabulary(words_set)

    def __initialize_vocabulary(self) -> None:
        self.word_to_index['PADDING'] = 0
        self.word_to_index['UNKNOWN'] = 1

        self.char_to_index['PADDING'] = 0
        self.char_to_index['UNKNOWN'] = 1

        self.vocab_size += 2
        self.chars_size += 2

    def __create_vocabulary(self, words_set: Set[str]) -> None:
        for word in words_set:
            self.__add_character_info(word)
            word = word.lower()
            if word not in self.word_to_index:
                self.word_to_index[word] = self.vocab_size
                self.vocab_size += 1

    def __add_character_info(self, word: str) -> None:  # TODO nie jestem pewny czy wszystkie znaki są ok
        for char in word:
            if char not in self.char_to_index:
                self.char_to_index[char] = self.chars_size
                self.chars_size += 1

    @staticmethod
    def get_char_type(char: str):
        if char.isupper():
            return 0
        elif char.islower():
            return 1
        elif char in string.punctuation:
            return 2
        else:
            return 3
