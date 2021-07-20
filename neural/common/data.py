import sys
from typing import List, Tuple, Any, Iterator

import datasets
import tqdm
from torchtext.data.utils import get_tokenizer


class DatasetGenerator:
    def __init__(self, dataset_name: str, split: str):
        self.__dataset_name = dataset_name
        self.__split = split

    def generate_dataset(self) -> Iterator[Tuple[Any, ...]]:
        if self.__dataset_name == 'cnn_dailymail':
            return self.__generate_cnn_dailymail()
        elif self.__dataset_name == 'conll2003':
            return self.__generate_conll2003()
        else:
            raise ValueError(f'Dataset "{self.__dataset_name}" is not supported.')

    def __generate_cnn_dailymail(self) -> Iterator[Tuple[List[str], List[str]]]:
        dataset = datasets.load_dataset('cnn_dailymail', '3.0.0', split=self.__split)
        dataset = dataset.to_dict()
        texts = dataset['article']
        summaries = dataset['highlights']

        tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        for text, summary in tqdm.tqdm(zip(texts, summaries), total=len(texts), file=sys.stdout):
            text_tokens = tokenizer(text)
            summary_tokens = tokenizer(summary)
            yield text_tokens, summary_tokens

    def __generate_conll2003(self) -> Iterator[Tuple[List[str], List[str]]]:
        dataset = datasets.load_dataset('conll2003', split=self.__split)
        dataset = dataset.to_dict()
        tokens_list = dataset['tokens']
        tags_list = dataset['ner_tags']

        for tokens, tags in tqdm.tqdm(zip(tokens_list, tags_list), total=len(tokens_list), file=sys.stdout):
            yield tokens, tags
