import argparse
from functools import partial
from typing import Tuple, Any

import torch
from torch import Tensor

import neural.common.scores as scores
import neural.common.utils as utils
from neural.common.data.vocab import SpecialTokens, VocabBuilder
from neural.common.losses import SummarizationLoss, CoverageLoss
from neural.common.scores import ScoreValue
from neural.common.train import Trainer, add_base_train_args
from neural.summarization.dataloader import SummarizationDataset, SummarizationDataLoader
from neural.summarization.pointer_generator import PointerGeneratorNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Parameters for Pointer-Generator model training.')
    parser.add_argument('--dataset', choices=['cnn_dailymail', 'xsum'], default='cnn_dailymail', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=13, help='Training epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--vocab-size', type=int, default=50000, help='Vocabulary size')
    parser.add_argument('--embedding-size', type=int, default=128, help='Size of word embeddings')
    parser.add_argument('--hidden-size', type=int, default=256, help='Size of hidden dim in LSTM layers')
    parser.add_argument('--max-article-length', type=int, default=400, help='Articles will be truncated to this value')
    parser.add_argument('--max-summary-length', type=int, default=100, help='Summaries will be truncated to this value')
    parser.add_argument('--coverage-iter', type=int, default=48000, help='Number of iteration with coverage')
    parser.add_argument('--coverage-lambda', type=float, default=1, help='Weight value for coverage loss')
    parser.add_argument('--lr', type=float, default=0.15, help='Learning rate')
    parser.add_argument('--init_acc_value', type=float, default=0.1, help='Initial accumulator value for Adagrad')
    parser.add_argument('--max-gradient-norm', type=int, default=2, help='Max norm for gradient clipping')
    parser.add_argument('--beam-size', type=int, default=4, help='Beam size for beam search decoding')
    parser = add_base_train_args(parser)

    return parser.parse_args()


def train_step(trainer: Trainer, inputs: Tuple[Any, ...]) -> Tuple[Tensor, ScoreValue]:
    texts, texts_lengths, summaries, summaries_lengths, texts_extended, targets, oov_list = inputs
    model = trainer.model.pointer_generator

    if not model.with_coverage and (trainer.current_phase == 'test' or (trainer.current_phase == 'train' and
                                                                        trainer.current_iteration >=
                                                                        trainer.params.iterations_without_coverage)):
        batch_size = texts.shape[1]
        trainer.logger.info(f'Iteration {trainer.current_iteration // batch_size}.'
                            f' Activated coverage mechanism.')
        model.activate_coverage()

    oov_size = len(max(oov_list, key=lambda x: len(x)))
    output, tokens, attention, coverage = model(texts, texts_lengths, texts_extended, oov_size, summaries)
    loss = trainer.criterion.summarization(output, targets, summaries_lengths)
    if coverage is not None:
        loss = loss + trainer.criterion.coverage(attention, coverage, targets)

    batch_size = targets.shape[1]
    score = ScoreValue()
    for i in range(batch_size):  # Due to different OOV words for each sequence in a batch, it has to scored separately
        utils.add_words_to_vocab(trainer.params.vocab, oov_list[i])
        score_out = tokens[:, i].unsqueeze(dim=1)
        score_target = targets[:, i].unsqueeze(dim=1)
        score += trainer.score(score_out, score_target)
        utils.remove_words_from_vocab(trainer.params.vocab, oov_list[i])

    del output
    del attention
    del coverage

    return loss, score / batch_size


def create_model_from_args(args: argparse.Namespace, bos_index: int, eos_index: int,
                           unk_index: int) -> PointerGeneratorNetwork:
    return PointerGeneratorNetwork(
        vocab_size=args.vocab_size + len(SpecialTokens),
        bos_index=bos_index,
        eos_index=eos_index,
        unk_index=unk_index,
        embedding_dim=args.embedding_size,
        hidden_size=args.hidden_size,
        max_summary_length=args.max_summary_length,
        beam_size=args.beam_size
    )


def main() -> None:
    args = parse_args()
    utils.set_random_seed(args.seed)
    model_name = 'pointer_generator'
    utils.dump_args_to_file(args, args.model_path / model_name)

    vocab = VocabBuilder.build_vocab(args.dataset, 'summarization', vocab_size=args.vocab_size,
                                     vocab_dir=args.vocab_path)
    dataset = partial(SummarizationDataset, args.dataset, max_article_length=args.max_article_length,
                      max_summary_length=args.max_summary_length, vocab=vocab, get_oov=True,
                      data_dir=args.data_path)
    dataloader = partial(SummarizationDataLoader, batch_size=args.batch)

    train_dataset = dataset(split='train')
    validation_dataset = dataset(split='validation')
    test_dataset = dataset(split='test')
    train_loader = dataloader(train_dataset)
    validation_loader = dataloader(validation_dataset)
    test_loader = dataloader(test_dataset)

    bos_index = vocab.stoi[SpecialTokens.BOS.value]
    eos_index = vocab.stoi[SpecialTokens.EOS.value]
    model = create_model_from_args(args, bos_index=bos_index, eos_index=eos_index, unk_index=vocab.unk_index)
    iterations_without_coverage = len(train_dataset) * args.epochs - args.coverage_iter
    rouge = scores.ROUGE(vocab, 'rouge1', 'rouge2', 'rougeL')
    meteor = scores.METEOR(vocab)

    trainer = Trainer(
        train_step=train_step,
        epochs=args.epochs,
        max_gradient_norm=args.max_gradient_norm,
        model_save_path=args.model_path,
        log_save_path=args.logs_path,
        model_name=model_name,
        use_cuda=args.use_gpu,
        load_checkpoint=args.load_checkpoint,
        validation_scores=[rouge],
        test_scores=[rouge, meteor],
        iterations_without_coverage=iterations_without_coverage,
        vocab=vocab
    )
    trainer.set_models(
        pointer_generator=model
    )
    trainer.set_criterion(
        summarization=SummarizationLoss(),
        coverage=CoverageLoss(args.coverage_lambda)
    )
    trainer.set_optimizer(
        adagrad=torch.optim.Adagrad(model.parameters(), lr=args.lr, initial_accumulator_value=args.init_acc_value)
    )
    trainer.train(train_loader, validation_loader)
    trainer.eval(test_loader)


if __name__ == '__main__':
    main()
