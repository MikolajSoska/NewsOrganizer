import enum
import os
from collections import Counter
from typing import List, Tuple, Iterator

import datasets
import torch
import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, T_co
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab


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


class SummarizationDataset(Dataset):
    def __init__(self, dataset_name: str, vocab_dir: str = 'data/vocabs', data_dir: str = 'data/datasets'):
        self.__vocab = self.__build_vocab(dataset_name, vocab_dir)
        self.__dataset = self.__build_dataset(dataset_name, data_dir)

    def __build_vocab(self, dataset_name: str, vocab_dir: str) -> Vocab:
        vocab_path = f'{vocab_dir}/vocab-summarization-{dataset_name}.pt'
        if os.path.exists(vocab_path):
            return torch.load(vocab_path)

        counter = Counter()
        for text_tokens, summary_tokens in self.__dataset_tokens_generator(dataset_name):
            counter.update(token.lower() for token in text_tokens + summary_tokens)

        vocab = Vocab(counter, max_size=150000, specials=SpecialTokens.get_tokens())

        torch.save(vocab, vocab_path)
        return vocab

    def __build_dataset(self, dataset_name: str, data_dir: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        dataset_path = f'{data_dir}/dataset-summarization-{dataset_name}.pt'
        if os.path.exists(dataset_path):
            return torch.load(dataset_path)

        dataset = []
        for text_tokens, summary_tokens in self.__dataset_tokens_generator(dataset_name):
            text_tokens = [SpecialTokens.BOS.value] + text_tokens + [SpecialTokens.EOS.value]
            summary_tokens = [SpecialTokens.BOS.value] + summary_tokens + [SpecialTokens.EOS.value]
            text_tensor = torch.tensor([self.__vocab.stoi[token] for token in text_tokens], dtype=torch.int)
            summary_tensor = torch.tensor([self.__vocab.stoi[token] for token in summary_tokens], dtype=torch.int)
            dataset.append((text_tensor, summary_tensor))

        torch.save(dataset, dataset_path)
        return dataset

    def __dataset_tokens_generator(self, dataset_name: str) -> Iterator[Tuple[List[str], List[str]]]:
        if dataset_name == 'cnn_dailymail':
            texts, summaries = self.__get_cnn_dailymail()
        else:
            raise ValueError(f'Unsupported dataset: {dataset_name}.')

        tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        for text, summary in tqdm.tqdm(zip(texts, summaries), total=len(texts)):
            text_tokens = tokenizer(text)
            summary_tokens = tokenizer(summary)
            yield text_tokens, summary_tokens

    @staticmethod
    def __get_cnn_dailymail() -> Tuple[List[str], List[str]]:
        dataset = datasets.load_dataset('cnn_dailymail', '3.0.0', split='train')
        dataset = dataset.to_dict()
        return dataset['article'], dataset['highlights']

    def __getitem__(self, index: int) -> T_co:
        return self.__dataset[index]

    def __len__(self) -> int:
        return len(self.__dataset)


class SummarizationDataLoader(DataLoader):
    def __init__(self, dataset: Dataset[T_co], batch_size: int):
        super().__init__(dataset, batch_size, shuffle=True, drop_last=True, collate_fn=self.__generate_batch)

    @staticmethod
    def __generate_batch(batch: List) -> Tuple[torch.Tensor, torch.Tensor]:
        texts, summaries = zip(*batch)
        texts_padded = pad_sequence(texts)
        summaries_padded = pad_sequence(summaries)

        return texts_padded, summaries_padded
