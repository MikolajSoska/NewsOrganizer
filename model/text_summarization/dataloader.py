import os
from collections import Counter
from typing import List, Tuple

import datasets
import torch
import tqdm
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab


class SummarizationDataset:
    def __init__(self, dataset_name: str, vocab_dir: str = 'data/weights'):
        self.__vocab = self.__build_vocab(dataset_name, vocab_dir)

    def __build_vocab(self, dataset_name: str, vocab_dir: str) -> Vocab:
        vocab_path = f'{vocab_dir}/vocab-summarization-{dataset_name}.pt'
        if os.path.exists(vocab_path):
            return torch.load(vocab_path)

        if dataset_name == 'cnn_dailymail':
            texts, summaries = self.__get_cnn_dailymail()
        else:
            raise ValueError(f'Unsupported dataset: {dataset_name}.')

        tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        counter = Counter()
        for text, summary in tqdm.tqdm(zip(texts, summaries), total=len(texts)):
            tokens = tokenizer(text) + tokenizer(summary)
            counter.update(token.lower() for token in tokens)

        vocab = Vocab(counter, max_size=150000, specials=['<pad>', '<unk>', '<bos>', '<eos>'])
        torch.save(vocab, vocab_path)
        return vocab

    @staticmethod
    def __get_cnn_dailymail() -> Tuple[List[str], List[str]]:
        dataset = datasets.load_dataset('cnn_dailymail', '3.0.0', split='train')
        dataset = dataset.to_dict()
        return dataset['article'], dataset['highlights']
