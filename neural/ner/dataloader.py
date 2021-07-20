import os
import string
from collections import Counter
from typing import List, Set, Tuple

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

    def __init__(self, dataset_name: str, split: str, vocab: VocabWithChars, data_dir: str = '../data/datasets'):
        self.__vocab = vocab
        self.__dataset = self.__build_dataset(dataset_name, split, data_dir)

    def __build_dataset(self, dataset_name: str, split: str, data_dir: str) -> List[Tuple[Tensor, Tensor, List[Tensor],
                                                                                          List[int], List[List[int]]]]:
        dataset_path = f'{data_dir}/dataset-{split}-ner-{dataset_name}-vocab-' \
                       f'{len(self.__vocab) - len(SpecialTokens.get_tokens())}.pt'
        if os.path.exists(dataset_path):
            return torch.load(dataset_path)

        dataset = []
        generator = DatasetGenerator.generate_dataset(dataset_name, split)
        for tokens, tags in generator:
            word_indexes = []
            word_types = []
            char_list = []
            char_types = []
            for word in tokens:
                word_indexes.append(self.__vocab.stoi[word.lower()])
                word_types.append(self.__get_word_type(word))
                char_tensor, types = self.__process_word_chars(word)
                char_list.append(char_tensor)
                char_types.append(types)

            words_tensor = torch.tensor(word_indexes, dtype=torch.long)
            tags_tensor = torch.tensor(tags, dtype=torch.long)
            dataset.append((words_tensor, tags_tensor, char_list, word_types, char_types))

        torch.save(dataset, dataset_path)
        return dataset

    def __process_word_chars(self, word: str) -> Tuple[Tensor, List[int]]:
        indexes = []
        types = []
        for char in word:
            indexes.append(self.__vocab.chars.stoi[char])
            types.append(self.__get_char_type(char))

        return torch.tensor(indexes, dtype=torch.long), types

    @staticmethod
    def __get_word_type(word: str) -> int:
        if word.isupper():
            return 0  # All caps
        elif word.istitle():
            return 1  # Upper initial
        elif word.lower():
            return 2  # All lower
        elif any(char.isupper() for char in word):
            return 3  # Mixed case
        else:
            return 4  # No info

    @staticmethod
    def __get_char_type(char: str) -> int:
        if char.isupper():
            return 0  # Upper case
        elif char.islower():
            return 1  # Lower case
        elif char in string.punctuation:
            return 2  # Punctuation
        else:
            return 3  # Other

    def __getitem__(self, index: int) -> T_co:
        return self.__dataset[index]

    def __len__(self) -> int:
        return len(self.__dataset)


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
