import argparse
from functools import partial
from typing import Tuple, Any

from torch import Tensor

import neural.common.scores as scores
from neural.common.data.vocab import SpecialTokens, VocabBuilder
from neural.common.losses import LabelSmoothingCrossEntropy
from neural.common.optimizers import TransformerAdam
from neural.common.scores import ScoreValue
from neural.common.train import Trainer, add_base_train_args
from neural.common.utils import set_random_seed, dump_args_to_file
from neural.summarization.dataloader import SummarizationDataset, SummarizationDataLoader
from neural.summarization.transformer import Transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Parameters for Transformer model training.')
    parser.add_argument('--experiment-name', type=str, default='transformer', help='Name of the experiment')
    parser.add_argument('--dataset', choices=['cnn_dailymail', 'xsum', 'psc'], default='cnn_dailymail',
                        help='Dataset name')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch', type=int, default=4, help='Batch size during training')
    parser.add_argument('--eval-batch', type=int, default=-1, help='Batch size in eval (equal to training batch if -1')
    parser.add_argument('--vocab-size', type=int, default=50000, help='Vocabulary size')
    parser.add_argument('--max-article-length', type=int, default=400, help='Articles will be truncated to this value')
    parser.add_argument('--max-summary-length', type=int, default=100, help='Summaries will be truncated to this value')
    parser.add_argument('--encoder-layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--decoder-layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('--embedding-dim', type=int, default=512, help='Embedding dimension size')
    parser.add_argument('--key-and_query-dim', type=int, default=64, help='Key and query dimension size')
    parser.add_argument('--value-dim', type=int, default=64, help='Value dimension size')
    parser.add_argument('--heads-number', type=int, default=8, help='Number of heads in self-attention layers')
    parser.add_argument('--ffn-size', type=int, default=2048, help='Size of hidden dimension in feed forward layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--warmup-steps', type=int, default=256000, help='Number of warmup steps in LR scheduling')
    parser.add_argument('--adam-betas', type=float, default=[0.9, 0.98], nargs=2, help='Betas for Adam optimizer')
    parser.add_argument('--adam-eps', type=float, default=1e-9, help='Epsilon value for Adam optimizer')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing value')
    parser.add_argument('--beam-size', type=int, default=4, help='Beam size for beam search decoding')

    parser = add_base_train_args(parser)

    return parser.parse_args()


def train_step(trainer: Trainer, inputs: Tuple[Any, ...]) -> Tuple[Tensor, ScoreValue]:
    texts, _, summaries, _, targets = inputs
    model = trainer.model.transformer
    output, tokens = model(texts, summaries)

    loss = trainer.criterion.loss(output, targets)
    score = trainer.score(tokens, targets)
    del output

    return loss, score


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    dump_args_to_file(args, args.model_path / args.experiment_name)
    if args.eval_batch < 0:
        args.eval_batch = args.batch

    vocab = VocabBuilder.build_vocab(args.dataset, 'summarization', vocab_size=args.vocab_size,
                                     vocab_dir=args.vocab_path)
    dataset = partial(SummarizationDataset, args.dataset, max_article_length=args.max_article_length,
                      max_summary_length=args.max_summary_length, vocab=vocab, get_oov=False,
                      data_dir=args.data_path)

    train_dataset = dataset(split='train') if not args.eval_only else None
    validation_dataset = dataset(split='validation')
    test_dataset = dataset(split='test')

    # Predictions and targets always have the same size, so no need to extra padding
    train_loader = SummarizationDataLoader(train_dataset, batch_size=args.batch, pad_to_max=False) \
        if not args.eval_only else None
    validation_loader = SummarizationDataLoader(validation_dataset, batch_size=args.eval_batch, pad_to_max=True)
    test_loader = SummarizationDataLoader(test_dataset, batch_size=args.eval_batch, pad_to_max=True)

    bos_index = vocab.stoi[SpecialTokens.BOS.value]
    eos_index = vocab.stoi[SpecialTokens.EOS.value]
    model = Transformer.create_from_args(vars(args), bos_index, eos_index)
    rouge = scores.ROUGE(vocab, 'rouge1', 'rouge2', 'rougeL')
    meteor = scores.METEOR(vocab)

    trainer = Trainer(
        train_step=train_step,
        epochs=args.epochs,
        max_gradient_norm=None,
        model_save_path=args.model_path / args.experiment_name,
        log_save_path=args.logs_path / args.experiment_name,
        model_name='transformer',
        use_cuda=args.use_gpu,
        cuda_index=args.gpu_index,
        validation_scores=[rouge],
        test_scores=[rouge, meteor]
    )
    trainer.set_models(
        transformer=model
    )
    trainer.set_criterion(
        loss=LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    )
    trainer.set_optimizer(
        adam=TransformerAdam(model.parameters(), betas=tuple(args.adam_betas), eps=args.adam_eps,
                             model_dim=args.embedding_dim, warmup_steps=args.warmup_steps, batch_size=args.batch))
    if not args.eval_only:
        trainer.train(train_loader, validation_loader)
    trainer.eval(test_loader, validation_loader, full_validation=args.full_validation)


if __name__ == '__main__':
    main()
