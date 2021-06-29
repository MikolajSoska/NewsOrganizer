import argparse
from typing import Tuple, Any

import torch
from torch import Tensor

import neural.summarization.dataloader as dataloader
from neural.common.losses import SummarizationLoss, CoverageLoss
from neural.common.trainer import Trainer
from neural.summarization.pointer_generator import PointerGeneratorNetwork
from utils.general import set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Parameters for text summarization model training.')
    parser.add_argument('--dataset', choices=['cnn_dailymail'], default='cnn_dailymail', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=13, help='Training epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--vocab-size', type=int, default=50000, help='Vocabulary size')
    parser.add_argument('--max-article-length', type=int, default=400, help='Articles will be truncated to this value')
    parser.add_argument('--max-summary-length', type=int, default=100, help='Summaries will be truncated to this value')
    parser.add_argument('--coverage', type=int, default=48000, help='Number of iteration with coverage')
    parser.add_argument('--lr', type=float, default=0.15, help='Learning rate')
    parser.add_argument('--init_acc_value', type=float, default=0.1, help='Initial accumulator value for Adagrad')
    parser.add_argument('--max-gradient-norm', type=int, default=2, help='Max norm for gradient clipping')
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--load-checkpoint', action='store_true')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    return parser.parse_args()


def train_step(train: Trainer, inputs: Tuple[Any, ...]) -> Tensor:
    texts, texts_lengths, summaries, summaries_lengths, texts_extended, targets, oov_list = inputs
    model = train.model.pointer_generator

    if train.current_iteration >= train.params.iterations_without_coverage and not model.with_coverage:
        print(f'Iteration {train.current_iteration // train.batch_size}. Activated coverage mechanism.')
        model.activate_coverage()

    oov_size = len(max(oov_list, key=lambda x: len(x)))
    output, attention, coverage = model(texts, texts_lengths, texts_extended, oov_size, summaries)
    loss = train.criterion.summarization(output, targets, summaries_lengths)
    if coverage is not None:
        loss = loss + train.criterion.coverage(attention, coverage, targets)

    del output
    del attention
    del coverage

    return loss


def main():
    args = parse_args()
    set_random_seed(args.seed)

    vocab = dataloader.build_vocab(args.dataset, vocab_size=args.vocab_size)
    train_dataset = dataloader.SummarizationDataset(args.dataset, 'train', max_article_length=args.max_article_length,
                                                    max_summary_length=args.max_summary_length, vocab=vocab,
                                                    get_oov=True)
    validation_dataset = dataloader.SummarizationDataset(args.dataset, 'validation',
                                                         max_article_length=args.max_article_length,
                                                         max_summary_length=args.max_summary_length, vocab=vocab,
                                                         get_oov=True)
    test_dataset = dataloader.SummarizationDataset(args.dataset, 'test', max_article_length=args.max_article_length,
                                                   max_summary_length=args.max_summary_length, vocab=vocab,
                                                   get_oov=True)

    train_loader = dataloader.SummarizationDataLoader(train_dataset, batch_size=args.batch, get_oov=True)
    validation_loader = dataloader.SummarizationDataLoader(validation_dataset, batch_size=args.batch, get_oov=True)
    test_loader = dataloader.SummarizationDataLoader(test_dataset, batch_size=args.batch, get_oov=True)

    bos_index = vocab.stoi[dataloader.SpecialTokens.BOS]
    model = PointerGeneratorNetwork(args.vocab_size + len(dataloader.SpecialTokens), bos_index)
    iterations_without_coverage = len(train_dataset) - args.coverage

    trainer = Trainer(
        train_step=train_step,
        epochs=args.epochs,
        batch_size=args.batch,
        max_gradient_norm=args.max_gradient_norm,
        save_path='../data/weights',
        model_name='summarization-model',
        use_cuda=args.use_gpu,
        load_checkpoint=args.load_checkpoint,
        iterations_without_coverage=iterations_without_coverage
    )
    trainer.set_models(
        pointer_generator=model
    )
    trainer.set_criterion(
        summarization=SummarizationLoss(),
        coverage=CoverageLoss()
    )
    trainer.set_optimizer(
        adagrad=torch.optim.Adagrad(model.parameters(), lr=args.lr, initial_accumulator_value=args.init_acc_value)
    )
    trainer.train(train_loader, validation_loader)


if __name__ == '__main__':
    main()
