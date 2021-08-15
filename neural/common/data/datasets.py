import os
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterator

import datasets
import tqdm
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from torchtext.utils import download_from_url


class DatasetGenerator:
    @classmethod
    def generate_dataset(cls, dataset_name: str, split: str, for_vocab: bool = False) -> Iterator[Tuple[Any, ...]]:
        if dataset_name == 'cnn_dailymail':
            return cls.__generate_cnn_dailymail(split, for_vocab)
        elif dataset_name == 'xsum':
            return cls.__generate_xsum(split, for_vocab)
        elif dataset_name == 'conll2003':
            return cls.__generate_conll2003(split, for_vocab)
        elif dataset_name == 'gmb':
            return cls.__generate_gmb(split, for_vocab)
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

    @classmethod
    def __generate_gmb(cls, split: str, for_vocab: bool) -> Iterator[Tuple[List[str], ...]]:
        download_dir = Path.home() / '.cache' / 'datasets'
        download_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = download_dir / 'gmb_dataset'
        if not dataset_dir.exists():
            cls.__download_and_parse_gmb_dataset(download_dir, dataset_dir)
        else:
            print(f'Reusing downloaded dataset from {dataset_dir}.')

        split_dir = dataset_dir / split
        files = os.listdir(split_dir)
        for filename in tqdm.tqdm(files, total=len(files), file=sys.stdout):
            tokens = []
            tags = []
            with open(split_dir / filename, 'r', encoding='utf-8') as text_file:
                for line in text_file.readlines():
                    token, tag = line.strip().split('\t')
                    tokens.append(token)
                    tags.append(int(tag))

            if for_vocab:
                yield tokens,
            else:
                yield tokens, tags

    @classmethod
    def __download_and_parse_gmb_dataset(cls, download_dir: Path, dataset_dir: Path) -> None:
        print('Downloading GMB dataset...')
        zip_file = download_from_url('https://gmb.let.rug.nl/releases/gmb-2.2.0.zip', root=download_dir)
        extracted_dir = download_dir / 'gmb'
        if not extracted_dir.exists():
            with zipfile.ZipFile(zip_file, 'r') as dataset_zip:
                print('Extracting dataset archive...')
                dataset_zip.extractall(extracted_dir)
                print('Archive extracted.')

        text_indexes = []
        for i, metadata_file in tqdm.tqdm(enumerate(extracted_dir.rglob('*/en.met')), desc='Processing metadata',
                                          file=sys.stdout):
            with open(metadata_file, 'r', encoding='utf-8') as metadata:
                genre = metadata.readlines()[3].split('genre:')[-1].strip()  # Extract text genre from metadata

            if 'newspaper' not in genre:  # Get only texts from news articles:
                continue

            text_indexes.append(i)

        # Train, test, validation spit
        train_indexes, test_indexes = train_test_split(text_indexes, test_size=0.15, random_state=0)
        train_indexes, val_indexes = train_test_split(train_indexes, test_size=0.15, random_state=0)

        dataset_dir.mkdir(parents=True, exist_ok=False)  # If this directory exists, raise an error
        (dataset_dir / 'train').mkdir(parents=False, exist_ok=False)
        (dataset_dir / 'validation').mkdir(parents=False, exist_ok=False)
        (dataset_dir / 'test').mkdir(parents=False, exist_ok=False)

        tag_to_index = defaultdict(lambda: len(tag_to_index))
        tag_to_index['O'] = 0  # Add no-entity tag as first one
        for i, text_file in tqdm.tqdm(enumerate(extracted_dir.rglob('*/en.tags')), desc='Processing texts',
                                      file=sys.stdout):
            if i not in text_indexes:
                continue

            if i in train_indexes:
                split = 'train'
            elif i in val_indexes:
                split = 'validation'
            elif i in test_indexes:
                split = 'test'
            else:
                raise ValueError(f'Article with index {i} is not on the splits lists.')

            output_file = dataset_dir / split / f'article_{i}.txt'
            cls.__process_gmb_named_entities(text_file, output_file, tag_to_index)

        print('Cleaning directory...')
        shutil.rmtree(extracted_dir)
        Path(zip_file).unlink()
        print('Dataset ready.')

    @staticmethod
    def __process_gmb_named_entities(text_file: Path, output_file: Path, tag_to_index: Dict[str, int]) -> None:
        tag_started = None
        with open(text_file, 'r', encoding='utf-8') as text_data, open(output_file, 'w', encoding='utf-8') as out:
            for line in text_data.readlines():
                if not line.strip():
                    continue

                token, _, _, tag, *_ = line.split('\t')
                tag = tag.upper().split('-')[0]  # Store tags in upper case + get only first main category of tag
                if tag == 'O':
                    tag_started = None
                else:
                    if tag == tag_started:
                        position_prefix = 'I'
                    else:
                        position_prefix = 'B'
                    tag_started = tag
                    tag = f'{position_prefix}-{tag}'

                tag_index = tag_to_index[tag]
                out.write(f'{token}\t{tag_index}\n')
