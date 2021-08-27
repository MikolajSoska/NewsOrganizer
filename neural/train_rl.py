import argparse
from functools import partial
from pathlib import Path
from typing import Tuple, Any

import torch
from torch import Tensor
from torchtext.vocab import GloVe

import neural.common.scores as scores
import neural.common.utils as utils
from neural.common.data.embeddings import CollobertEmbeddings
from neural.common.data.vocab import SpecialTokens, VocabBuilder
from neural.common.losses import MLLoss, PolicyLearning, MixedRLLoss
from neural.common.scores import ScoreValue
from neural.common.train import Trainer, add_base_train_args
from neural.summarization.dataloader import SummarizationDataset, SummarizationDataLoader
from neural.summarization.reinforcement_learning import ReinforcementSummarization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Parameters for ML+RL model training.')
    parser.add_argument('--dataset', choices=['cnn_dailymail', 'xsum'], default='cnn_dailymail', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--vocab-size', type=int, default=50000, help='Vocabulary size')
    parser.add_argument('--hidden-size', type=int, default=400, help='Hidden dimension size')
    parser.add_argument('--max-article-length', type=int, default=800, help='Articles will be truncated to this value')
    parser.add_argument('--max-summary-length', type=int, default=100, help='Summaries will be truncated to this value')
    parser.add_argument('--pretrain-epochs', type=int, default=15, help='Pretraining (only ML train) epochs number')
    parser.add_argument('--pretrain-lr', type=float, default=0.001, help='Learning rate during pretraining phase')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--teacher-forcing', type=float, default=0.75, help='Teacher forcing ratio')
    parser.add_argument('--gamma', type=float, default=0.9984, help='Scaling factor to mixed objective loss')
    parser.add_argument('--train-ml', action='store_true', help='Train using maximum likelihood')
    parser.add_argument('--train-rl', action='store_true', help='Train using reinforcement learning')
    parser.add_argument('--intra-attention', action='store_true', help='Use decoder intra-attention in training')
    parser.add_argument('--pretrained-embeddings', choices=['no', 'collobert', 'glove'], default='glove',
                        help='Which pretrained embeddings use')
    parser.add_argument('--embedding-size', type=int, default=100, help='Size of embeddings (if no pretrained')
    parser.add_argument('--embedding-path', type=Path, default='../data/saved/embeddings', help='Path to embeddings')
    parser = add_base_train_args(parser)

    return parser.parse_args()


def train_step(trainer: Trainer, inputs: Tuple[Any, ...]) -> Tuple[Tensor, ScoreValue]:
    __update_training_phase(trainer)
    texts, texts_lengths, summaries, summaries_lengths, texts_extended, targets, oov_list = inputs
    model = trainer.model.rl_model
    teacher_forcing_ratio = trainer.params.teacher_forcing_ratio
    oov_size = len(max(oov_list, key=lambda x: len(x)))
    batch_size = targets.shape[1]
    score = ScoreValue()

    if trainer.params.pretrain:  # Pretraining is performed only with ML training
        train_ml = True
        train_rl = False
    else:  # During normal phase use config from path args
        train_ml = trainer.params.train_ml
        train_rl = trainer.params.train_rl

    if train_ml or trainer.current_phase != 'train':  # During validation only use this approach
        predictions, tokens, _ = model(texts, texts_lengths, texts_extended, oov_size, summaries, teacher_forcing_ratio)
        ml_loss = trainer.criterion.ml_loss(predictions, targets, summaries_lengths)

        # Scoring is performed only in ML approach
        # Due to different OOV words for each sequence in a batch, it has to scored separately
        for i in range(batch_size):
            utils.add_words_to_vocab(trainer.params.vocab, oov_list[i])
            score_out = tokens[:, i].unsqueeze(dim=1)
            score_target = targets[:, i].unsqueeze(dim=1)
            score += trainer.score(score_out, score_target)
            utils.remove_words_from_vocab(trainer.params.vocab, oov_list[i])

        del predictions
        del tokens
    else:
        ml_loss = 0

    if train_rl and trainer.current_phase == 'train':  # Use RL only in training phase
        log_probabilities, tokens, _ = model(texts, texts_lengths, texts_extended, oov_size, teacher_forcing_ratio=0.,
                                             train_rl=True)
        with torch.no_grad():
            _, baseline_tokens, _ = model(texts, texts_lengths, texts_extended, oov_size, teacher_forcing_ratio=0.)
        rl_loss = trainer.criterion.rl_loss(log_probabilities, tokens, baseline_tokens, targets, oov_list)
        del log_probabilities
        del baseline_tokens
        del tokens
    else:
        rl_loss = 0

    if train_ml and train_rl:
        loss = trainer.criterion.mixed_loss(ml_loss, rl_loss)
    else:
        loss = ml_loss + rl_loss

    return loss, score / batch_size


def __update_training_phase(trainer: Trainer) -> None:
    if trainer.params.pretrain and trainer.current_epoch >= trainer.params.pretrain_epochs:
        trainer.logger.info(f'Epoch {trainer.current_epoch}. Ending pretraining.')
        trainer.params.pretrain = False  # Start normal training
        optimizer = trainer.optimizer.adam
        for params in optimizer.param_groups:  # Update learning rate
            params['lr'] = trainer.params.normal_lr


def create_model_from_args(args: argparse.Namespace, bos_index: int, unk_index: int,
                           embeddings: Tensor = None) -> ReinforcementSummarization:
    return ReinforcementSummarization(args.vocab_size + len(SpecialTokens), args.hidden_size, args.max_summary_length,
                                      bos_index, unk_index, args.embedding_size, embeddings, args.intra_attention)


def main() -> None:
    args = parse_args()
    assert args.train_ml or args.train_rl, 'At least one training method need to be specified'
    utils.set_random_seed(args.seed)
    model_name = 'reinforcement_learning'
    utils.dump_args_to_file(args, args.model_path / model_name)

    vocab = VocabBuilder.build_vocab(args.dataset, 'summarization', vocab_size=args.vocab_size,
                                     vocab_dir=args.vocab_path)
    dataset = partial(SummarizationDataset, args.dataset, max_article_length=args.max_article_length,
                      max_summary_length=args.max_summary_length, vocab=vocab, get_oov=True,
                      data_dir=args.data_path)
    dataloader = partial(SummarizationDataLoader, batch_size=args.batch)

    if args.pretrained_embeddings == 'collobert':
        vectors = CollobertEmbeddings(args.embedding_path)
        vocab.load_vectors(vectors)
        embeddings = vocab.vectors
    elif args.pretrained_embeddings == 'glove':
        vectors = GloVe(name='6B', dim=100, cache=args.embedding_path)
        vocab.load_vectors(vectors)
        embeddings = vocab.vectors
    else:
        embeddings = None

    train_dataset = dataset(split='train')
    validation_dataset = dataset(split='validation')
    test_dataset = dataset(split='test')
    train_loader = dataloader(train_dataset)
    validation_loader = dataloader(validation_dataset)
    test_loader = dataloader(test_dataset)

    model = create_model_from_args(args, bos_index=vocab.stoi[SpecialTokens.BOS.value], unk_index=vocab.unk_index,
                                   embeddings=embeddings)
    rouge = scores.ROUGE(vocab, 'rouge1', 'rouge2', 'rougeL')
    meteor = scores.METEOR(vocab)

    trainer = Trainer(
        train_step=train_step,
        epochs=args.epochs,
        max_gradient_norm=None,
        model_save_path=args.model_path,
        log_save_path=args.logs_path,
        model_name=model_name,
        use_cuda=args.use_gpu,
        load_checkpoint=args.load_checkpoint,
        validation_scores=[rouge],
        test_scores=[rouge, meteor],
        vocab=vocab,
        teacher_forcing_ratio=args.teacher_forcing,
        train_ml=args.train_ml,
        train_rl=args.train_rl,
        normal_lr=args.lr,
        pretrain=True,
        pretrain_epochs=args.pretrain_epochs
    )
    trainer.set_models(
        rl_model=model
    )
    trainer.set_criterion(
        ml_loss=MLLoss(),
        rl_loss=PolicyLearning(vocab),
        mixed_loss=MixedRLLoss(args.gamma)
    )
    trainer.set_optimizer(
        adam=torch.optim.Adam(model.parameters(), lr=args.pretrain_lr)  # Start optimizer with pretrain LR
    )
    trainer.train(train_loader, validation_loader)
    trainer.eval(test_loader)


if __name__ == '__main__':
    main()
