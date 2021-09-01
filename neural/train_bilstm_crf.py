import argparse
from functools import partial
from pathlib import Path
from typing import Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor
from torchtext.vocab import GloVe

from neural.common.data.embeddings import CollobertEmbeddings
from neural.common.data.vocab import VocabBuilder, VocabWithChars
from neural.common.scores import Precision, Recall, F1Score
from neural.common.scores import ScoreValue
from neural.common.train import Trainer, add_base_train_args
from neural.common.utils import set_random_seed, dump_args_to_file
from neural.ner.bilstm_crf import BiLSTMCRF
from neural.ner.dataloader import NERDataset, NERDataLoader
from utils.database import DatabaseConnector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Parameters for BiLSTM-CRF model training.')
    parser.add_argument('--dataset', choices=['conll2003', 'gmb'], default='conll2003', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--char-lstm-hidden', type=int, default=50, help='Char LSTM hidden layers size')
    parser.add_argument('--word-lstm-hidden', type=int, default=200, help='Word LSTM hidden layers size')
    parser.add_argument('--char-embedding-size', type=int, default=25, help='Size of chars embedding')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--pretrained-embeddings', choices=['no', 'collobert', 'glove'], default='collobert',
                        help='Which pretrained embeddings use')
    parser.add_argument('--word-embedding-size', type=int, default=100, help='Size of word embedding (if no pretrained')
    parser.add_argument('--max-gradient-norm', type=int, default=5, help='Max norm for gradient clipping')
    parser.add_argument('--embedding-path', type=Path, default='../data/saved/embeddings', help='Path to embeddings')
    parser = add_base_train_args(parser)

    return parser.parse_args()


def train_step(trainer: Trainer, inputs: Tuple[Any, ...]) -> Tuple[Tensor, ScoreValue]:
    words, tags, chars, _, _ = inputs
    model = trainer.model.bilstm_crf
    mask = (tags >= 0).float()

    loss, predictions = model(words, chars, tags, mask)
    predictions = torch.flatten(predictions)
    tags = torch.flatten(tags)

    # Remove padding from tags and output
    predictions = predictions[predictions >= 0]
    tags = tags[tags >= 0]

    score = trainer.score(predictions, tags)
    del predictions

    return loss, score


def create_model_from_args(args: argparse.Namespace, tags_count: int, vocab: VocabWithChars,
                           embeddings: Tensor = None) -> BiLSTMCRF:
    return BiLSTMCRF(
        output_size=tags_count,
        char_hidden_size=args.char_lstm_hidden,
        word_hidden_size=args.word_lstm_hidden,
        char_vocab_size=len(vocab.chars),
        char_embedding_dim=args.char_embedding_size,
        dropout_rate=args.dropout,
        word_vocab_size=len(vocab),
        word_embedding_dim=args.word_embedding_size,
        embeddings=embeddings
    )


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    model_name = 'bilstm_crf'
    dump_args_to_file(args, args.model_path / model_name)
    vocab = VocabBuilder.build_vocab(args.dataset, 'ner', vocab_type='char', vocab_dir=args.vocab_path,
                                     digits_to_zero=True)

    if args.pretrained_embeddings == 'collobert':
        vectors = CollobertEmbeddings(args.embedding_path)
        vocab.load_vectors(vectors)
        embeddings = vocab.vectors
    elif args.pretrained_embeddings == 'glove':
        vectors = GloVe(name='6B', dim=50, cache=args.embedding_path)
        vocab.load_vectors(vectors)
        embeddings = vocab.vectors
    else:
        embeddings = None

    dataset = partial(NERDataset, args.dataset, vocab=vocab, lowercase=True, digits_to_zero=True,
                      data_dir=args.data_path)
    dataloader = partial(NERDataLoader, batch_size=args.batch, two_sided_char_padding=False)

    train_dataset = dataset(split='train')
    validation_dataset = dataset(split='validation')
    test_dataset = dataset(split='test')

    train_loader = dataloader(train_dataset)
    validation_loader = dataloader(validation_dataset)
    test_loader = dataloader(test_dataset)
    model = create_model_from_args(args, DatabaseConnector().get_tag_count(args.dataset) + 1, vocab, embeddings)

    trainer = Trainer(
        train_step=train_step,
        epochs=args.epochs,
        max_gradient_norm=args.max_gradient_norm,
        model_save_path=args.model_path,
        log_save_path=args.logs_path,
        model_name=model_name,
        use_cuda=args.use_gpu,
        cuda_index=args.gpu_index,
        scores=[Precision(), Recall(), F1Score()]
    )
    trainer.set_models(
        bilstm_crf=model
    )
    trainer.set_criterion(
        loss=nn.Identity()  # Loss is calculated via CRF layer in model
    )
    trainer.set_optimizer(
        sgd=torch.optim.SGD(model.parameters(), lr=args.lr)
    )
    trainer.train(train_loader, validation_loader, verbosity=50, save_interval=50)
    trainer.eval(test_loader)


if __name__ == '__main__':
    main()
