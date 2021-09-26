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
from neural.ner.dataloader import NERDataset, NERDataLoader
from neural.ner.id_cnn import IteratedDilatedCNN
from utils.database import DatabaseConnector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Parameters for ID-CNN model training.')
    parser.add_argument('--experiment-name', type=str, default='id_cnn', help='Name of the experiment')
    parser.add_argument('--dataset', choices=['conll2003', 'gmb'], default='conll2003', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch', type=int, default=128, help='Batch size during training')
    parser.add_argument('--eval-batch', type=int, default=-1, help='Batch size in eval (equal to training batch if -1')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--cnn-width', type=int, default=3, help='Convolution width')
    parser.add_argument('--cnn-filters', type=int, default=300, help='Convolution filters size')
    parser.add_argument('--dilation', type=int, default=[1, 2, 1], nargs='+', help='Dilation sizes in CNN block')
    parser.add_argument('--block-repeat', type=int, default=1, help='Convolution block repeats number')
    parser.add_argument('--input-dropout', type=float, default=0.65, help='Dropout rate for inputs')
    parser.add_argument('--block-dropout', type=float, default=0.85, help='Dropout rate after each CNN block')
    parser.add_argument('--pretrained-embeddings', choices=['no', 'collobert', 'glove'], default='collobert',
                        help='Which pretrained embeddings use')
    parser.add_argument('--word-embedding-size', type=int, default=100, help='Size of word embedding (if no pretrained')
    parser.add_argument('--word-features', action='store_true', help='Use additional word features')
    parser.add_argument('--max-gradient-norm', type=int, default=5, help='Max norm for gradient clipping')
    parser.add_argument('--adam-betas', type=float, default=[0.9, 0.9], nargs=2, help='Betas for Adam optimizer')
    parser.add_argument('--adam-eps', type=float, default=1e-6, help='Epsilon value for Adam optimizer')
    parser.add_argument('--embedding-path', type=Path, default='../data/saved/embeddings', help='Path to embeddings')
    parser = add_base_train_args(parser)

    return parser.parse_args()


def train_step(trainer: Trainer, inputs: Tuple[Any, ...]) -> Tuple[Tensor, ScoreValue]:
    words, tags, _, word_features, _ = inputs
    model = trainer.model.id_cnn

    outputs = model(words, word_features)
    tags = torch.flatten(tags)

    loss = torch.tensor(0., device=words.device)
    score = ScoreValue()
    tags_without_padding = tags[tags >= 0]
    for output in outputs:
        output = torch.flatten(output, end_dim=1)
        output = output[tags >= 0]
        loss += trainer.criterion.cross_entropy(output, tags_without_padding)
        score += trainer.score(output, tags_without_padding)

    loss = loss / len(outputs)
    score = score / len(outputs)
    if not any(tags) != 0:  # Compute score only if targets contains named entities
        score = ScoreValue()

    del outputs

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
    dataloader = partial(NERDataLoader, conv_kernel_size=args.cnn_width)

    train_dataset = dataset(split='train') if not args.eval_only else None
    validation_dataset = dataset(split='validation')
    test_dataset = dataset(split='test')

    train_loader = dataloader(train_dataset, batch_size=args.batch) if not args.eval_only else None
    validation_loader = dataloader(validation_dataset, batch_size=args.eval_batch)
    test_loader = dataloader(test_dataset, batch_size=args.eval_batch)
    tags_count = DatabaseConnector().get_tag_count(args.dataset) + 1
    model = IteratedDilatedCNN.create_from_args(vars(args), tags_count, len(vocab), embeddings)
    labels = list(DatabaseConnector().get_tags_dict(args.dataset).keys())

    trainer = Trainer(
        train_step=train_step,
        epochs=args.epochs,
        max_gradient_norm=args.max_gradient_norm,
        model_save_path=args.model_path / args.experiment_name,
        log_save_path=args.logs_path / args.experiment_name,
        model_name='id_cnn',
        use_cuda=args.use_gpu,
        cuda_index=args.gpu_index,
        scores=[Precision(labels), Recall(labels), F1Score(labels)]
    )
    trainer.set_models(
        id_cnn=model
    )
    trainer.set_criterion(
        cross_entropy=nn.CrossEntropyLoss()
    )
    trainer.set_optimizer(
        adam=torch.optim.Adam(model.parameters(), lr=args.lr, betas=tuple(args.adam_betas), eps=args.adam_eps)
    )
    if not args.eval_only:
        trainer.train(train_loader, validation_loader)
    trainer.eval(test_loader, validation_loader, full_validation=args.full_validation)


if __name__ == '__main__':
    main()
