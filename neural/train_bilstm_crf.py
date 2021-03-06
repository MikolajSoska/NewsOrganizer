import argparse
from functools import partial
from pathlib import Path
from typing import Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor
from torchtext.vocab import GloVe

from neural.common.data.embeddings import CollobertEmbeddings
from neural.common.data.vocab import VocabBuilder
from neural.common.scores import Precision, Recall, F1Score
from neural.common.scores import ScoreValue
from neural.common.train import Trainer, add_base_train_args
from neural.common.utils import set_random_seed, dump_args_to_file
from neural.ner.bilstm_crf import BiLSTMCRF
from neural.ner.dataloader import NERDataset, NERDataLoader
from utils.database import DatabaseConnector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Parameters for BiLSTM-CRF model training.')
    parser.add_argument('--experiment-name', type=str, default='bilstm_crf', help='Name of the experiment')
    parser.add_argument('--dataset', choices=['conll2003', 'gmb'], default='conll2003', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch', type=int, default=128, help='Batch size during training')
    parser.add_argument('--eval-batch', type=int, default=-1, help='Batch size in eval (equal to training batch if -1')
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

    if any(tags) != 0:  # Compute score only if targets contains named entities
        score = trainer.score(predictions, tags)
    else:
        score = ScoreValue()

    del predictions

    return loss, score


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    dump_args_to_file(args, args.model_path / args.experiment_name)
    if args.eval_batch < 0:
        args.eval_batch = args.batch

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
    dataloader = partial(NERDataLoader, two_sided_char_padding=False)

    train_dataset = dataset(split='train') if not args.eval_only else None
    validation_dataset = dataset(split='validation')
    test_dataset = dataset(split='test')

    train_loader = dataloader(train_dataset, batch_size=args.batch) if not args.eval_only else None
    validation_loader = dataloader(validation_dataset, batch_size=args.eval_batch)
    test_loader = dataloader(test_dataset, batch_size=args.eval_batch)
    tags_count = DatabaseConnector().get_tag_count(args.dataset) + 1
    model = BiLSTMCRF.create_from_args(vars(args), tags_count, len(vocab), len(vocab.chars), embeddings)
    labels = list(DatabaseConnector().get_tags_dict(args.dataset).keys())

    trainer = Trainer(
        train_step=train_step,
        epochs=args.epochs,
        max_gradient_norm=args.max_gradient_norm,
        model_save_path=args.model_path / args.experiment_name,
        log_save_path=args.logs_path / args.experiment_name,
        model_name='bilstm_crf',
        use_cuda=args.use_gpu,
        cuda_index=args.gpu_index,
        scores=[Precision(labels), Recall(labels), F1Score(labels)]
    )
    trainer.set_models(
        bilstm_crf=model
    )
    trainer.set_criterion(
        loss=nn.Identity()  # Loss is calculated via CRF layer in model
    )
    trainer.set_optimizer(
        adam=torch.optim.Adam(model.parameters(), lr=args.lr)
    )
    if not args.eval_only:
        trainer.train(train_loader, validation_loader)
    trainer.eval(test_loader, validation_loader, full_validation=args.full_validation)


if __name__ == '__main__':
    main()
