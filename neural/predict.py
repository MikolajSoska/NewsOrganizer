from argparse import Namespace
from pathlib import Path
from typing import List, Tuple, Union, Any, Callable

import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer

import neural.common.utils as utils
import neural.train_bilstm_cnn as ner
import neural.train_pointer_generator as summ
from neural.common.data.vocab import SpecialTokens, VocabBuilder
from neural.ner.dataloader import NERDataset, NERDataLoader
from neural.summarization.dataloader import SummarizationDataset
from news.article import NewsArticle
from utils.database import DatabaseConnector
from utils.general import tokenize_text_content


class NewsPredictor:
    def __init__(self, use_cuda: bool, path_to_models: Union[Path, str] = '../data/saved/models', seed: int = 0):
        utils.set_random_seed(seed)
        connector = DatabaseConnector()
        if isinstance(path_to_models, str):
            path_to_models = Path(path_to_models)

        self.__device = utils.get_device(use_cuda)
        self.__tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        self.__tags_dict = connector.get_tags_dict('conll2003')

        self.__ner_vocab = VocabBuilder.build_vocab('conll2003', 'ner', vocab_type='char', digits_to_zero=True)
        tags_count = connector.get_tag_count('conll2003') + 1
        self.__ner_model = self.__load_pretrained_model(path_to_models, ner.create_model_from_args, 'bilstm_cnn',
                                                        tags_count=tags_count, vocab=self.__ner_vocab)

        self.__summarization_vocab = VocabBuilder.build_vocab('cnn_dailymail', 'summarization', vocab_size=50000)
        bos_index = self.__summarization_vocab.stoi[SpecialTokens.BOS.value]
        unk_index = self.__summarization_vocab.unk_index
        self.__summarization_model = self.__load_pretrained_model(path_to_models, summ.create_model_from_args,
                                                                  'pointer_generator', bos_index=bos_index,
                                                                  unk_index=unk_index)
        self.__summarization_model.activate_coverage()

    def process_article(self, article: NewsArticle) -> NewsArticle:
        article.summary = self.__create_summarization(article.content)
        article.named_entities = self.__get_named_entities(article.content)

        return article

    def __load_pretrained_model(self, path_to_model: Path, model_builder: Callable[[Namespace, Any], nn.Module],
                                model_name: str, **additional_args: Any) -> nn.Module:
        args = utils.load_args_from_file(path_to_model / model_name)
        model = model_builder(args, **additional_args)
        weights_path = path_to_model / model_name / f'{model_name}.pt'
        weights = torch.load(weights_path)

        model.load_state_dict(weights[f'{model_name}_state_dict'])
        model.to(self.__device)
        model.eval()
        del weights

        return model

    def __create_summarization(self, article_content: str) -> str:
        tokens = tokenize_text_content(article_content, word_tokenizer=self.__tokenizer)
        tokens = [SpecialTokens.BOS.value] + tokens + [SpecialTokens.EOS.value]
        oov_article_tensor, oov_list = SummarizationDataset.get_tokens_tensor(tokens, self.__summarization_vocab)
        oov_article_tensor = oov_article_tensor.to(self.__device)
        oov_article_tensor = oov_article_tensor.unsqueeze(1)
        article_tensor = SummarizationDataset.remove_oov_words(oov_article_tensor, self.__summarization_vocab)
        article_length = torch.tensor(len(tokens), device=self.__device).unsqueeze(0)

        summary, _, _ = self.__summarization_model(article_tensor, article_length, oov_article_tensor, len(oov_list))
        summary_tokens = []
        vocab_size = len(self.__summarization_vocab)
        for output in summary:
            token = torch.argmax(output).item()
            if token == self.__summarization_vocab.stoi[SpecialTokens.EOS.value]:
                break  # Discarding tokens after first EOS token

            if token < vocab_size:
                summary_tokens.append(self.__summarization_vocab.itos[token])
            else:
                summary_tokens.append(oov_list[token - vocab_size])

        return ' '.join(summary_tokens)

    def __get_named_entities(self, article_content: str) -> List[Tuple[str, int, int, str]]:
        tokens = tokenize_text_content(article_content, word_tokenizer=self.__tokenizer)
        words_tensor, word_types_tensor, char_list, char_types = NERDataset.process_tokens(tokens, self.__ner_vocab,
                                                                                           lowercase=True,
                                                                                           digits_to_zero=True)
        chars_tensor = NERDataLoader.pad_char_sequence((char_list,), self.__ner_model.conv_width)
        chars_types_tensor = NERDataLoader.pad_char_sequence((char_types,), self.__ner_model.conv_width)

        words_tensor = words_tensor.unsqueeze(1).to(self.__device)
        word_types_tensor = word_types_tensor.unsqueeze(1).to(self.__device)
        chars_tensor = chars_tensor.to(self.__device)
        chars_types_tensor = chars_types_tensor.to(self.__device)

        tags = self.__ner_model(words_tensor, chars_tensor, word_types_tensor, chars_types_tensor)
        tags = torch.argmax(tags, dim=-1).squeeze()

        named_entities = []
        entity_begun = False
        for i, (tag, word) in enumerate(zip(tags, tokens)):
            tag = tag.item()
            if tag == 0:
                entity_begun = False
                continue

            tag_name, category_name = self.__tags_dict[tag]
            position, tag_name = tag_name.split('-')

            if position == 'B':
                entity_begun = True
                named_entities.append((category_name, i, 1, word))  # Initial tag length is 1
            else:
                if entity_begun:
                    if category_name == named_entities[-1][0]:
                        category_name, index, length, tag_word = named_entities[-1]
                        # Increase entity length by one and add another word
                        named_entities[-1] = (category_name, index, length + 1, f'{tag_word} {word}')
                    else:
                        named_entities.append((category_name, i, 1, word))
                        entity_begun = False
                else:
                    named_entities.append((category_name, i, 1, word))

        return named_entities
