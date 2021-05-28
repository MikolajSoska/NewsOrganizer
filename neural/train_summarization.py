from typing import Tuple, Any

import torch
from torch import Tensor

import neural.model.utils as utils
from neural.model.summarization.dataloader import SummarizationDataset, SummarizationDataLoader, SpecialTokens
from neural.model.summarization.pointer_generator import PointerGeneratorNetwork
from neural.train import Trainer
from utils.general import set_random_seed


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
    set_random_seed(0)
    epochs = 13
    batch_size = 8
    coverage_iterations = 48000
    vocab_size = 50000

    dataset = SummarizationDataset('cnn_dailymail', max_article_length=400, max_summary_length=100,
                                   vocab_size=vocab_size, get_oov=True)
    loader = SummarizationDataLoader(dataset, batch_size=batch_size, get_oov=True)
    bos_index = dataset.token_to_index(SpecialTokens.BOS)
    model = PointerGeneratorNetwork(vocab_size + len(SpecialTokens), bos_index)
    iterations_without_coverage = len(dataset) - coverage_iterations

    trainer = Trainer(
        train_step=train_step,
        train_loader=loader,
        epochs=epochs,
        batch_size=batch_size,
        save_path='../data/weights',
        model_name='summarization-model',
        use_cuda=True,
        load_checkpoint=True,
        iterations_without_coverage=iterations_without_coverage
    )
    trainer.set_models(
        pointer_generator=model
    )
    trainer.set_criterion(
        summarization=utils.SummarizationLoss(),
        coverage=utils.CoverageLoss()
    )
    trainer.set_optimizer(
        adagrad=torch.optim.Adagrad(model.parameters(), lr=0.15, initial_accumulator_value=0.1)
    )
    trainer.train()


if __name__ == '__main__':
    main()
