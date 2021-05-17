import enum
import os
from collections import Counter
from typing import List, Tuple, Iterator, Union

import datasets
import torch
import tqdm
from torch import Tensor
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
    def __init__(self, dataset_name: str, max_article_length: int, max_summary_length: int, vocab_size: int,
                 get_oov: bool = False, vocab_dir: str = 'data/vocabs', data_dir: str = 'data/datasets'):
        self.get_oov = get_oov
        self.vocab = self.__build_vocab(dataset_name, vocab_dir, vocab_size)
        self.__dataset = self.__build_dataset(dataset_name, data_dir)
        self.__max_article_length = max_article_length
        self.__max_summary_length = max_summary_length

    def __build_vocab(self, dataset_name: str, vocab_dir: str, vocab_size: int) -> Vocab:
        vocab_path = f'{vocab_dir}/vocab-summarization-{vocab_size}-{dataset_name}.pt'
        if os.path.exists(vocab_path):
            return torch.load(vocab_path)

        counter = Counter()
        for text_tokens, summary_tokens in self.__dataset_tokens_generator(dataset_name):
            counter.update(token.lower() for token in text_tokens + summary_tokens)

        vocab = Vocab(counter, max_size=vocab_size, specials=SpecialTokens.get_tokens())

        torch.save(vocab, vocab_path)
        return vocab

    def __build_dataset(self, dataset_name: str, data_dir: str) -> List[Tuple[Tensor, Tensor, List[str]]]:
        dataset_path = f'{data_dir}/dataset-summarization-{dataset_name}-vocab-' \
                       f'{len(self.vocab) - len(SpecialTokens.get_tokens())}.pt'
        if os.path.exists(dataset_path):
            return torch.load(dataset_path)

        dataset = []
        for text_tokens, summary_tokens in self.__dataset_tokens_generator(dataset_name):
            text_tokens = [SpecialTokens.BOS.value] + text_tokens + [SpecialTokens.EOS.value]
            summary_tokens = [SpecialTokens.BOS.value] + summary_tokens + [SpecialTokens.EOS.value]
            text_tensor, oov_list = self.__get_tokens_tensor(text_tokens)
            summary_tensor, _ = self.__get_tokens_tensor(summary_tokens, oov_list, update_oov=False)
            dataset.append((text_tensor, summary_tensor, oov_list))

        torch.save(dataset, dataset_path)
        return dataset

    def __get_tokens_tensor(self, tokens: List[str], oov_list: List[str] = None,
                            update_oov: bool = True) -> Tuple[Tensor, List[str]]:
        token_indexes = []
        oov_list = oov_list or []
        for token in tokens:
            token_index = self.vocab.stoi[token.lower()]
            if token_index == self.vocab.unk_index:
                if token.lower() not in oov_list:
                    if update_oov:
                        oov_list.append(token.lower())
                        token_index = len(self.vocab) + oov_list.index(token.lower())
                else:
                    token_index = len(self.vocab) + oov_list.index(token.lower())
            token_indexes.append(token_index)

        return torch.tensor(token_indexes, dtype=torch.long), oov_list

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
        text, summary, oov_list = self.__dataset[index]
        text = text[:self.__max_article_length]
        target = summary[1:]  # Without BOS token
        if len(summary) > self.__max_summary_length:
            summary = summary[:self.__max_summary_length]
            target = target[:self.__max_summary_length]
        else:
            summary = summary[:-1]  # Remove EOS token

        if self.get_oov:
            return text, summary, target, oov_list
        else:  # Replace OOV tokens with UNK
            text[text >= len(self.vocab)] = self.vocab.unk_index
            summary[summary >= len(self.vocab)] = self.vocab.unk_index
            target[target >= len(self.vocab)] = self.vocab.unk_index
            return text, summary, target

    def __len__(self) -> int:
        return len(self.__dataset)


class SummarizationDataLoader(DataLoader):
    def __init__(self, dataset: SummarizationDataset, batch_size: int):
        super().__init__(dataset, batch_size, shuffle=True, drop_last=True, collate_fn=self.__generate_batch)
        self.get_oov = dataset.get_oov

    def __generate_batch(self, batch: List) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
                                                     Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tuple[List[str]]]]:
        oov_list = None
        if self.get_oov:
            texts, summaries, targets, oov_list = zip(*batch)
        else:
            texts, summaries, targets = zip(*batch)
        texts_lengths = torch.tensor([len(text) for text in texts])
        summaries_lengths = torch.tensor([len(summary) for summary in summaries])

        texts_padded = pad_sequence(texts)
        summaries_padded = pad_sequence(summaries)
        targets_padded = pad_sequence(targets)

        if self.get_oov:
            return texts_padded, texts_lengths, summaries_padded, summaries_lengths, targets_padded, oov_list
        else:
            return texts_padded, texts_lengths, summaries_padded, summaries_lengths, targets_padded
