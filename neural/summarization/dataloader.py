from pathlib import Path
from typing import List, Tuple, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, T_co
from torchtext.vocab import Vocab

from neural.common.data.datasets import DatasetGenerator
from neural.common.data.vocab import SpecialTokens


class SummarizationDataset(Dataset):
    def __init__(self, dataset_name: str, split: str, max_article_length: int, max_summary_length: int, vocab: Vocab,
                 get_oov: bool = False, data_dir: Union[Path, str] = '../data/saved/datasets'):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        self.get_oov = get_oov
        self.max_article_length = max_article_length
        self.max_summary_length = max_summary_length
        self.__vocab = vocab
        self.__dataset = self.__build_dataset(dataset_name, split, data_dir)

    def __build_dataset(self, dataset_name: str, split: str, data_dir: Path) -> List[Tuple[Tensor, Tensor, List[str]]]:
        dataset_path = data_dir / f'dataset-{split}-summarization-{dataset_name}-vocab-' \
                                  f'{len(self.__vocab) - len(SpecialTokens.get_tokens())}.pt'
        if dataset_path.exists():
            return torch.load(dataset_path)

        dataset = []
        generator = DatasetGenerator.generate_dataset(dataset_name, split)
        for text_tokens, summary_tokens in generator:
            text_tokens = [SpecialTokens.BOS.value] + text_tokens + [SpecialTokens.EOS.value]
            summary_tokens = [SpecialTokens.BOS.value] + summary_tokens + [SpecialTokens.EOS.value]
            text_tensor, oov_list = self.get_tokens_tensor(text_tokens, self.__vocab)
            summary_tensor, _ = self.get_tokens_tensor(summary_tokens, self.__vocab, oov_list, update_oov=False)
            dataset.append((text_tensor, summary_tensor, oov_list))

        torch.save(dataset, dataset_path)
        return dataset

    @staticmethod
    def get_tokens_tensor(tokens: List[str], vocab: Vocab, oov_list: List[str] = None,
                          update_oov: bool = True) -> Tuple[Tensor, List[str]]:
        token_indexes = []
        oov_list = oov_list or []
        for token in tokens:
            token_index = vocab.stoi[token.lower()]
            if token_index == vocab.unk_index:
                if token.lower() not in oov_list:
                    if update_oov:
                        oov_list.append(token.lower())
                        token_index = len(vocab) + oov_list.index(token.lower())
                else:
                    token_index = len(vocab) + oov_list.index(token.lower())
            token_indexes.append(token_index)

        return torch.tensor(token_indexes, dtype=torch.long), oov_list

    @staticmethod
    def remove_oov_words(text: Tensor, vocab: Vocab) -> Tensor:
        text_without_oov = torch.clone(text)
        text_without_oov[text >= len(vocab)] = vocab.unk_index
        return text_without_oov

    def __getitem__(self, index: int) -> T_co:
        text, summary, oov_list = self.__dataset[index]
        text = text[:self.max_article_length]
        target = summary[1:]  # Without BOS token
        if len(summary) > self.max_summary_length:
            summary = summary[:self.max_summary_length]
            target = target[:self.max_summary_length]
        else:
            summary = summary[:-1]  # Remove EOS token

        if self.get_oov:
            text_without_oov = self.remove_oov_words(text, self.__vocab)
            summary_without_oov = torch.clone(summary)
            summary_without_oov[summary >= len(self.__vocab)] = self.__vocab.unk_index
            return text_without_oov, summary_without_oov, text, summary, target, oov_list
        else:  # Replace OOV tokens with UNK
            text[text >= len(self.__vocab)] = self.__vocab.unk_index
            summary[summary >= len(self.__vocab)] = self.__vocab.unk_index
            target[target >= len(self.__vocab)] = self.__vocab.unk_index
            return text, summary, target

    def __len__(self) -> int:
        return len(self.__dataset)


class SummarizationDataLoader(DataLoader):
    def __init__(self, dataset: SummarizationDataset, batch_size: int, pad_to_max: bool = True):
        super().__init__(dataset, batch_size, shuffle=True, drop_last=False, collate_fn=self.__generate_batch)
        self.__get_oov = dataset.get_oov
        self.__max_summary_length = dataset.max_summary_length
        self.__pad_to_max = pad_to_max

    def __generate_batch(self, batch: List) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
                                                     Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                                                           Tuple[List[str]]]]:
        oov_list = None
        texts_extended_padded = None
        if self.__get_oov:
            texts, summaries, texts_extended, summaries_extended, targets, oov_list = zip(*batch)
            texts_extended_padded = pad_sequence(texts_extended)
        else:
            texts, summaries, targets = zip(*batch)
        texts_lengths = torch.tensor([len(text) for text in texts])
        summaries_lengths = torch.tensor([len(summary) for summary in summaries])

        texts_padded = pad_sequence(texts)
        summaries_padded = pad_sequence(summaries)
        targets_padded = pad_sequence(targets)
        extra_padding = self.__max_summary_length - targets_padded.shape[0]
        # Summaries are always padded to max length to prevent errors with different prediction and target lengths
        if extra_padding > 0 and self.__pad_to_max:
            padding = torch.zeros(extra_padding, targets_padded.shape[1], dtype=torch.long)
            summaries_padded = torch.cat((summaries_padded, padding), dim=0)
            targets_padded = torch.cat((targets_padded, padding), dim=0)

        if self.__get_oov:
            return texts_padded, texts_lengths, summaries_padded, summaries_lengths, texts_extended_padded, \
                   targets_padded, oov_list
        else:
            return texts_padded, texts_lengths, summaries_padded, summaries_lengths, targets_padded
