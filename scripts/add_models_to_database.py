from typing import Dict, Type, Any

from neural.common.data.vocab import VocabBuilder
from neural.common.model import BaseModel
from neural.ner import BiLSTMCRF, BiLSTMConv, IteratedDilatedCNN
from neural.summarization import PointerGeneratorNetwork, ReinforcementSummarization, Transformer
from news.article import NewsModel
from utils.database import DatabaseConnector


def add_model(fullname: str, name_identifier: str, class_name: Type[BaseModel], dataset: str,
              constructor_args: Dict[str, Any], dataset_args: Dict[str, Any], batch_size: int) -> None:
    class_name = class_name.__name__
    model = NewsModel(0, fullname, name_identifier, class_name, constructor_args, dataset_args, batch_size, dataset)
    DatabaseConnector().add_new_model(model)
    print(f'Added model {fullname} (dataset: {dataset}) to database.')


def main():
    vocab_ner_base = VocabBuilder.build_vocab('conll2003', 'ner', vocab_type='char', vocab_dir='../data/saved/vocabs',
                                              digits_to_zero=True)
    vocab_ner_additional = VocabBuilder.build_vocab('gmb', 'ner', vocab_type='char', vocab_dir='../data/saved/vocabs',
                                                    digits_to_zero=True)

    add_model(
        fullname='BiLSTM-CNN',
        name_identifier='bilstm_cnn',
        class_name=BiLSTMConv,
        dataset='conll2003',
        constructor_args={
            'tags_count': DatabaseConnector().get_tag_count('conll2003') + 1,
            'word_vocab_size': len(vocab_ner_base),
            'char_vocab_size': len(vocab_ner_base.chars)
        },
        dataset_args={
            'conv_width': 3
        },
        batch_size=9
    )

    add_model(
        fullname='BiLSTM-CNN',
        name_identifier='bilstm_cnn',
        class_name=BiLSTMConv,
        dataset='gmb',
        constructor_args={
            'tags_count': DatabaseConnector().get_tag_count('gmb') + 1,
            'word_vocab_size': len(vocab_ner_additional),
            'char_vocab_size': len(vocab_ner_additional.chars)
        },
        dataset_args={
            'conv_width': 3
        },
        batch_size=9
    )

    add_model(
        fullname='BiLSTM-CRF',
        name_identifier='bilstm_crf',
        class_name=BiLSTMCRF,
        dataset='conll2003',
        constructor_args={
            'tags_count': DatabaseConnector().get_tag_count('conll2003') + 1,
            'word_vocab_size': len(vocab_ner_base),
            'char_vocab_size': len(vocab_ner_base.chars)
        },
        dataset_args={
            'conv_width': None
        },
        batch_size=128
    )

    add_model(
        fullname='BiLSTM-CRF',
        name_identifier='bilstm_crf',
        class_name=BiLSTMCRF,
        dataset='gmb',
        constructor_args={
            'tags_count': DatabaseConnector().get_tag_count('gmb') + 1,
            'word_vocab_size': len(vocab_ner_additional),
            'char_vocab_size': len(vocab_ner_additional.chars)
        },
        dataset_args={
            'conv_width': None
        },
        batch_size=32
    )

    add_model(
        fullname='ID-CNN',
        name_identifier='id_cnn',
        class_name=IteratedDilatedCNN,
        dataset='conll2003',
        constructor_args={
            'tags_count': DatabaseConnector().get_tag_count('conll2003') + 1,
            'vocab_size': len(vocab_ner_base),
        },
        dataset_args={
            'conv_width': 3
        },
        batch_size=128
    )

    add_model(
        fullname='ID-CNN',
        name_identifier='id_cnn',
        class_name=IteratedDilatedCNN,
        dataset='gmb',
        constructor_args={
            'tags_count': DatabaseConnector().get_tag_count('gmb') + 1,
            'vocab_size': len(vocab_ner_additional),
        },
        dataset_args={
            'conv_width': 3
        },
        batch_size=128
    )

    add_model(
        fullname='Pointer-generator',
        name_identifier='pointer_generator',
        class_name=PointerGeneratorNetwork,
        dataset='cnn_dailymail',
        constructor_args={
            'bos_index': 2,
            'eos_index': 3,
            'unk_index': 1
        },
        dataset_args={
            'max_article_length': 400
        },
        batch_size=8
    )

    add_model(
        fullname='Pointer-generator',
        name_identifier='pointer_generator',
        class_name=PointerGeneratorNetwork,
        dataset='xsum',
        constructor_args={
            'bos_index': 2,
            'eos_index': 3,
            'unk_index': 1
        },
        dataset_args={
            'max_article_length': 400
        },
        batch_size=8
    )

    add_model(
        fullname='RL+ML',
        name_identifier='reinforcement_learning',
        class_name=ReinforcementSummarization,
        dataset='cnn_dailymail',
        constructor_args={
            'bos_index': 2,
            'eos_index': 3,
            'unk_index': 1
        },
        dataset_args={
            'max_article_length': 800
        },
        batch_size=8
    )

    add_model(
        fullname='RL+ML',
        name_identifier='reinforcement_learning',
        class_name=ReinforcementSummarization,
        dataset='xsum',
        constructor_args={
            'bos_index': 2,
            'eos_index': 3,
            'unk_index': 1
        },
        dataset_args={
            'max_article_length': 800
        },
        batch_size=8
    )

    add_model(
        fullname='Transformer',
        name_identifier='transformer',
        class_name=Transformer,
        dataset='cnn_dailymail',
        constructor_args={
            'bos_index': 2,
            'eos_index': 3,
        },
        dataset_args={
            'max_article_length': 400
        },
        batch_size=4
    )

    add_model(
        fullname='Transformer',
        name_identifier='transformer',
        class_name=Transformer,
        dataset='xsum',
        constructor_args={
            'bos_index': 2,
            'eos_index': 3,
        },
        dataset_args={
            'max_article_length': 400
        },
        batch_size=4
    )


if __name__ == '__main__':
    main()
