import sys
from typing import List, Tuple, Any, Iterator

import datasets
import tqdm
from torchtext.data.utils import get_tokenizer


class DatasetGenerator:
    @classmethod
    def generate_dataset(cls, dataset_name: str, split: str, for_vocab: bool = False) -> Iterator[Tuple[Any, ...]]:
        if dataset_name == 'cnn_dailymail':
            return cls.__generate_cnn_dailymail(split, for_vocab)
        elif dataset_name == 'xsum':
            return cls.__generate_xsum(split, for_vocab)
        elif dataset_name == 'conll2003':
            return cls.__generate_conll2003(split, for_vocab)
        else:
            raise ValueError(f'Dataset "{dataset_name}" is not supported.')

    @classmethod
    def __generate_cnn_dailymail(cls, split: str, for_vocab: bool) -> Iterator[Tuple[List[str], ...]]:
        dataset = datasets.load_dataset('cnn_dailymail', '3.0.0', split=split)
        dataset = dataset.to_dict()
        texts = dataset['article']
        summaries = dataset['highlights']

        return cls.__generate_summarization_dataset(texts, summaries, for_vocab)

    @classmethod
    def __generate_xsum(cls, split: str, for_vocab: bool) -> Iterator[Tuple[List[str], ...]]:
        dataset = datasets.load_dataset('xsum', split=split)
        dataset = dataset.to_dict()
        texts = dataset['document']
        summaries = dataset['summary']

        return cls.__generate_summarization_dataset(texts, summaries, for_vocab)

    @staticmethod
    def __generate_summarization_dataset(texts: List[str], summaries: List[str],
                                         for_vocab: bool) -> Iterator[Tuple[List[str], ...]]:
        tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        for text, summary in tqdm.tqdm(zip(texts, summaries), total=len(texts), file=sys.stdout):
            text_tokens = tokenizer(text)
            summary_tokens = tokenizer(summary)
            if for_vocab:
                yield (text_tokens + summary_tokens),
            else:
                yield text_tokens, summary_tokens

    @staticmethod
    def __generate_conll2003(split: str, for_vocab: bool) -> Iterator[Tuple[List[str], ...]]:
        dataset = datasets.load_dataset('conll2003', split=split)
        dataset = dataset.to_dict()
        tokens_list = dataset['tokens']
        tags_list = dataset['ner_tags']

        for tokens, tags in tqdm.tqdm(zip(tokens_list, tags_list), total=len(tokens_list), file=sys.stdout):
            if for_vocab:
                yield tokens,
            else:
                yield tokens, tags
