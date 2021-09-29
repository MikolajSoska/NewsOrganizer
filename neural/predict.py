import importlib
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Union, Type

import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from tqdm import trange

import neural.common.utils as utils
from neural.common.data.vocab import SpecialTokens, VocabBuilder, VocabWithChars
from neural.common.model import BaseModel
from neural.ner.dataloader import NERDataset, NERDataLoader
from neural.summarization.dataloader import SummarizationDataset
from news.article import NewsArticle, NewsModel, NamedEntity
from utils.database import DatabaseConnector
from utils.general import tokenize_text_content


class NewsPredictor:
    def __init__(self, summarization_vocab_size: int, use_cuda: bool, cuda_index: int = 0,
                 path_to_models: Union[Path, str] = '../data/saved/models',
                 path_to_vocabs: Union[Path, str] = '../data/saved/vocabs'):
        connector = DatabaseConnector()
        if isinstance(path_to_models, str):
            path_to_models = Path(path_to_models)
        if isinstance(path_to_vocabs, str):
            path_to_vocabs = Path(path_to_vocabs)

        self.__path_to_models = path_to_models
        self.__device = utils.get_device(use_cuda, cuda_index)
        self.__tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

        self.__ner_datasets = connector.get_datasets_identifiers('Named Entity Recognition')
        self.__ner_models = self.__get_models(self.__ner_datasets)
        self.__ner_vocabs = self.__create_ner_vocabs(path_to_vocabs)
        self.__ner_tags_dict = self.__get_ner_tag_dicts()

        self.__summarization_datasets = connector.get_datasets_identifiers('Abstractive Summarization')
        self.__summarization_models = self.__get_models(self.__summarization_datasets)
        self.__summarization_vocabs = self.__create_summarization_vocabs(path_to_vocabs, summarization_vocab_size)

    @staticmethod
    def __get_models(datasets: List[str]) -> Dict[str, List[NewsModel]]:
        connector = DatabaseConnector()
        return {dataset: connector.get_models_by_dataset(dataset) for dataset in datasets}

    def __create_ner_vocabs(self, path_to_vocabs: Path) -> Dict[str, VocabWithChars]:
        vocabs = {}
        for dataset in self.__ner_datasets:
            vocab = VocabBuilder.build_vocab(dataset, 'ner', vocab_type='char', digits_to_zero=True,
                                             vocab_dir=path_to_vocabs)
            vocabs[dataset] = vocab

        return vocabs

    def __get_ner_tag_dicts(self) -> Dict[str, Dict[int, Tuple[str, str]]]:
        connector = DatabaseConnector()
        return {dataset: connector.get_tags_dict(dataset) for dataset in self.__ner_datasets}

    def __create_summarization_vocabs(self, path_to_vocabs: Path, vocab_size: int) -> Dict[str, Vocab]:
        vocabs = {}
        for dataset in self.__summarization_datasets:
            vocab = VocabBuilder.build_vocab(dataset, 'summarization', vocab_size=vocab_size, vocab_dir=path_to_vocabs)
            vocabs[dataset] = vocab

        return vocabs

    def process_articles(self, articles: List[NewsArticle], model_type: str) -> None:
        if model_type == 'NER':
            models = self.__ner_models
            datasets = self.__ner_datasets
            create_data = self.__get_named_entities
            loop_description = 'Extracting named entities'
        elif model_type == 'summarization':
            models = self.__summarization_models
            datasets = self.__summarization_datasets
            create_data = self.__create_summaries
            loop_description = 'Creating articles summaries'
        else:
            raise ValueError(f'Unknown model type: {model_type}.')

        for dataset in datasets:
            print(f'Starting {model_type} phase for {dataset} dataset...')
            for news_model in models[dataset]:
                print(f'Initializing {news_model.fullname} model...')
                model = self.__load_pretrained_model(news_model, model_type=model_type.lower())
                for i in trange(0, len(articles), news_model.batch_size, desc=loop_description, file=sys.stdout):
                    create_data(model, news_model, articles[i:i + news_model.batch_size])
                del model
                torch.cuda.empty_cache()

    def __load_pretrained_model(self, news_model: NewsModel, model_type: str) -> BaseModel:
        model_directory = f'{news_model.name_identifier}-{news_model.dataset_name}'

        args = utils.load_args_from_file(self.__path_to_models / model_directory)
        model_class: Type[BaseModel] = getattr(importlib.import_module(f'neural.{model_type}'), news_model.class_name)
        model = model_class.create_from_args(args, **news_model.constructor_args)

        weights_path = self.__path_to_models / model_directory / f'{news_model.name_identifier}.pt'
        weights = torch.load(weights_path, map_location=self.__device)

        for key in weights.keys():
            if 'state_dict' in key and 'optimizer' not in key:
                weights_key = key
                break
        else:  # Shouldn't ever happen
            raise ValueError(f'Can\'t find model state dict in weights file from {weights_path}.')

        model.load_state_dict(weights[weights_key])
        model = model.to(self.__device)
        model.eval()
        del weights

        return model

    def __get_named_entities(self, model: BaseModel, news_model: NewsModel, articles: List[NewsArticle]) -> None:
        tokens_list = []
        words_tensor = []
        chars_tensor = []
        word_types_tensor = []
        chars_types_tensor = []
        vocab = self.__ner_vocabs[news_model.dataset_name]
        conv_width = news_model.dataset_args['conv_width'] or 1

        for article in articles:
            tokens = tokenize_text_content(article.content, word_tokenizer=self.__tokenizer)
            words, word_types, chars, char_types = NERDataset.process_tokens(tokens, vocab, lowercase=True,
                                                                             digits_to_zero=True)
            words_tensor.append(words)
            chars_tensor.append(chars)
            word_types_tensor.append(word_types)
            chars_types_tensor.append(char_types)
            tokens_list.append(tokens)

        words_tensor = pad_sequence(words_tensor).to(self.__device)
        word_types_tensor = pad_sequence(word_types_tensor).to(self.__device)
        chars_tensor = NERDataLoader.pad_char_sequence(tuple(chars_tensor), conv_width).to(self.__device)
        chars_types_tensor = NERDataLoader.pad_char_sequence(tuple(chars_types_tensor), conv_width).to(self.__device)

        with torch.no_grad():
            tags = model.predict(words_tensor, chars_tensor, word_types_tensor, chars_types_tensor)

        for i in range(len(articles)):
            entities = self.__convert_article_entities(tags[:, i], tokens_list[i], news_model.dataset_name)
            articles[i].named_entities[news_model.model_id] = entities

        del words_tensor
        del word_types_tensor
        del chars_tensor
        del chars_types_tensor
        del tags

    def __convert_article_entities(self, tags: torch.Tensor, tokens: List[str], dataset: str) -> List[NamedEntity]:
        named_entities = []
        entity_begun = False
        for i, (tag, word) in enumerate(zip(tags, tokens)):
            tag = tag.item()
            if tag == 0:
                entity_begun = False
                continue

            tag_name, category_name = self.__ner_tags_dict[dataset][tag]
            position, tag_name = tag_name.split('-')

            if position == 'B':
                entity_begun = True
                named_entities.append(NamedEntity(category_name, tag_name, i, 1, word))  # Initial tag length is 1
            else:
                if entity_begun:
                    if category_name == named_entities[-1].full_name:
                        # Increase entity length by one and add another word
                        named_entities[-1].length += 1
                        named_entities[-1].words += f' {word}'
                    else:
                        named_entities.append(NamedEntity(category_name, tag_name, i, 1, word))
                        entity_begun = False
                else:
                    named_entities.append(NamedEntity(category_name, tag_name, i, 1, word))

        return named_entities

    def __create_summaries(self, model: BaseModel, news_model: NewsModel, articles: List[NewsArticle]) -> None:
        articles_tensor = []
        articles_lengths = []
        extended_articles_tensor = []
        oov_lists = []
        vocab = self.__summarization_vocabs[news_model.dataset_name]
        max_summary_length = news_model.dataset_args['max_article_length']
        for article in articles:
            tokens = tokenize_text_content(article.content, word_tokenizer=self.__tokenizer)
            tokens = [SpecialTokens.BOS.value] + tokens + [SpecialTokens.EOS.value]
            tokens = tokens[:max_summary_length]
            oov_article_tensor, oov_list = SummarizationDataset.get_tokens_tensor(tokens, vocab)
            article_tensor = SummarizationDataset.remove_oov_words(oov_article_tensor, vocab)

            articles_tensor.append(article_tensor)
            articles_lengths.append(len(tokens))
            extended_articles_tensor.append(oov_article_tensor)
            oov_lists.append(oov_list)

        articles_tensor = pad_sequence(articles_tensor).to(self.__device)
        articles_lengths = torch.tensor(articles_lengths, device=self.__device)
        extended_articles_tensor = pad_sequence(extended_articles_tensor).to(self.__device)

        with torch.no_grad():
            summary_tensor = model.predict(articles_tensor, articles_lengths, extended_articles_tensor, oov_lists)
        summary_tensor = utils.clean_predicted_tokens(summary_tensor, vocab.stoi[SpecialTokens.EOS.value])

        for i in range(len(articles)):
            summary = self.__get_article_summary(summary_tensor[:, i], vocab, oov_lists[i])
            articles[i].summaries[news_model.model_id] = summary

    @staticmethod
    def __get_article_summary(summary_tensor: torch.Tensor, vocab: Vocab, oov_list: List[str]) -> str:
        summary_tensor = utils.remove_unnecessary_padding(summary_tensor)
        utils.add_words_to_vocab(vocab, oov_list)
        summary = utils.tensor_to_string(vocab, summary_tensor, detokenize=True)
        utils.remove_words_from_vocab(vocab, oov_list)

        return summary
