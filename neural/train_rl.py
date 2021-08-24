import argparse
from functools import partial
from pathlib import Path
from typing import List, Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor
from torchtext.vocab import GloVe
from torchtext.vocab import Vocab

import neural.common.scores as scores
from neural.common.data.embeddings import CollobertEmbeddings
from neural.common.data.vocab import SpecialTokens, VocabBuilder
from neural.common.scores import ScoreValue
from neural.common.train import Trainer, add_base_train_args
from neural.common.utils import set_random_seed, dump_args_to_file
from neural.summarization.dataloader import SummarizationDataset, SummarizationDataLoader
from neural.summarization.reinforcement_learning import ReinforcementSummarization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Parameters for ML+RL model training.')
    parser.add_argument('--dataset', choices=['cnn_dailymail', 'xsum'], default='cnn_dailymail', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=13, help='Training epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--vocab-size', type=int, default=50000, help='Vocabulary size')
    parser.add_argument('--hidden-size', type=int, default=400, help='Hidden dimension size')
    parser.add_argument('--max-article-length', type=int, default=800, help='Articles will be truncated to this value')
    parser.add_argument('--max-summary-length', type=int, default=100, help='Summaries will be truncated to this value')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--pretrained-embeddings', choices=['no', 'collobert', 'glove'], default='glove',
                        help='Which pretrained embeddings use')
    parser.add_argument('--embedding-size', type=int, default=100, help='Size of embeddings (if no pretrained')
    parser.add_argument('--embedding-path', type=Path, default='../data/saved/embeddings', help='Path to embeddings')
    parser = add_base_train_args(parser)

    return parser.parse_args()


def train_step(trainer: Trainer, inputs: Tuple[Any, ...]) -> Tuple[Tensor, ScoreValue]:
    texts, texts_lengths, summaries, summaries_lengths, texts_extended, targets, oov_list = inputs
    model = trainer.model.rl_model

    oov_size = len(max(oov_list, key=lambda x: len(x)))
    predictions = model(texts, texts_lengths, texts_extended, oov_size, summaries)
    loss = trainer.criterion.nll(torch.flatten(predictions, end_dim=1), torch.flatten(targets))

    batch_size = targets.shape[1]
    score = ScoreValue()
    for i in range(batch_size):  # Due to different OOV words for each sequence in a batch, it has to scored separately
        add_words_to_vocab(trainer.params.vocab, oov_list[i])
        score_out = predictions[:, i, :].unsqueeze(dim=1)
        score_target = targets[:, i].unsqueeze(dim=1)
        score += trainer.score(score_out, score_target)
        remove_words_from_vocab(trainer.params.vocab, oov_list[i])

    del predictions

    return loss, score / batch_size


def add_words_to_vocab(vocab: Vocab, words: List[str]) -> None:
    for word in words:
        vocab.itos.append(word)
        vocab.stoi[word] = len(vocab.itos)


def remove_words_from_vocab(vocab: Vocab, words: List[str]) -> None:
    for word in words:
        del vocab.itos[-1]
        del vocab.stoi[word]


def create_model_from_args(args: argparse.Namespace, bos_index: int, unk_index: int,
                           embeddings: Tensor = None) -> ReinforcementSummarization:
    return ReinforcementSummarization(args.vocab_size + len(SpecialTokens), args.hidden_size, args.max_summary_length,
                                      bos_index, unk_index, args.embedding_size, embeddings)


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    model_name = 'reinforcement_learning'
    dump_args_to_file(args, args.model_path / model_name)

    vocab = VocabBuilder.build_vocab(args.dataset, 'summarization', vocab_size=args.vocab_size,
                                     vocab_dir=args.vocab_path)
    dataset = partial(SummarizationDataset, args.dataset, max_article_length=args.max_article_length,
                      max_summary_length=args.max_summary_length, vocab=vocab, get_oov=True,
                      data_dir=args.data_path)
    dataloader = partial(SummarizationDataLoader, batch_size=args.batch, get_oov=True)

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
        vocab=vocab
    )
    trainer.set_models(
        rl_model=model
    )
    trainer.set_criterion(
        nll=nn.NLLLoss(),
    )
    trainer.set_optimizer(
        adam=torch.optim.Adam(model.parameters(), lr=args.lr)
    )
    trainer.train(train_loader, validation_loader)
    trainer.eval(test_loader)


if __name__ == '__main__':
    main()
