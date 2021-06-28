import argparse
from typing import Tuple, Any

import torch
from torch import Tensor

from neural.common.losses import SummarizationLoss, CoverageLoss
from neural.common.trainer import Trainer
from neural.summarization.dataloader import SummarizationDataset, SummarizationDataLoader, SpecialTokens
from neural.summarization.pointer_generator import PointerGeneratorNetwork
from utils.general import set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Parameters for text summarization model training.')
    parser.add_argument('--epochs', type=int, default=13, help='Training epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--vocab-size', type=int, default=50000, help='Vocabulary size')
    parser.add_argument('--max-article-length', type=int, default=400, help='Articles will be truncated to this value')
    parser.add_argument('--max-summary-length', type=int, default=100, help='Summaries will be truncated to this value')
    parser.add_argument('--coverage', type=int, default=48000, help='Number of iteration with coverage')
    parser.add_argument('--lr', type=float, default=0.15, help='Learning rate')
    parser.add_argument('--init_acc_value', type=float, default=0.1, help='Initial accumulator value for Adagrad')
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

    dataset = SummarizationDataset('cnn_dailymail', max_article_length=args.max_article_length,
                                   max_summary_length=args.max_summary_length, vocab_size=args.vocab_size, get_oov=True)
    loader = SummarizationDataLoader(dataset, batch_size=args.batch, get_oov=True)
    bos_index = dataset.token_to_index(SpecialTokens.BOS)
    model = PointerGeneratorNetwork(args.vocab_size + len(SpecialTokens), bos_index)
    iterations_without_coverage = len(dataset) - args.coverage

    trainer = Trainer(
        train_step=train_step,
        train_loader=loader,
        epochs=args.epochs,
        batch_size=args.batch,
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
    trainer.train()


if __name__ == '__main__':
    main()
