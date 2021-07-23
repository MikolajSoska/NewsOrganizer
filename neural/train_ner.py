import argparse
from pathlib import Path
from typing import Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor

from neural.common.data.vocab import VocabBuilder
from neural.common.scores import Precision, Recall, F1Score
from neural.common.scores import ScoreValue
from neural.common.trainer import Trainer
from neural.ner.bilstm_cnn import BiLSTMConv
from neural.ner.dataloader import NERDatasetNew, NERDataLoaderNew
from utils.general import set_random_seed, dump_args_to_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Parameters for named entity recognition model training.')
    parser.add_argument('--dataset', choices=['conll2003'], default='conll2003', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=80, help='Training epochs')
    parser.add_argument('--batch', type=int, default=9, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0105, help='Learning rate')
    parser.add_argument('--cnn-width', type=int, default=3, help='Convolution width')
    parser.add_argument('--cnn-output', type=int, default=53, help='Convolution output size')
    parser.add_argument('--lstm-state', type=int, default=275, help='LSTM hidden layers size')
    parser.add_argument('--lstm-layers', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.68, help='Dropout rate')
    parser.add_argument('--char-embedding-size', type=int, default=25, help='Size of chars embedding')
    parser.add_argument('--pretrained-embeddings', choices=['no'], default='no', help='Which pretrained embeddings use')
    parser.add_argument('--word-embedding-size', type=int, default=50, help='Size of word embedding (if no pretrained')
    parser.add_argument('--word-features', action='store_true', help='Use additional word features')
    parser.add_argument('--char-features', action='store_true', help='Use additional char features')
    parser.add_argument('--use-gpu', action='store_true', help='Train with CUDA')
    parser.add_argument('--load-checkpoint', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--vocab-path', type=Path, default='../data/vocabs', help='Path to vocab files')
    parser.add_argument('--data-path', type=Path, default='../data/datasets', help='Path to dataset files')
    parser.add_argument('--model-path', type=Path, default='../data/models', help='Path to model files')
    parser.add_argument('--logs-path', type=Path, default='../data/logs', help='Path to logs files')

    return parser.parse_args()


def train_step(trainer: Trainer, inputs: Tuple[Any, ...]) -> Tuple[Tensor, ScoreValue]:
    words, tags, chars, word_features, char_features = inputs
    model = trainer.model.bilstm_cnn

    output = model(words, chars, word_features, char_features)
    output = torch.flatten(output, end_dim=1)
    tags = torch.flatten(tags)

    loss = trainer.criterion.cross_entropy(output, tags)
    score = trainer.score(output, tags)
    del output

    return loss, score


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    model_name = 'bilstm-cnn'
    dump_args_to_file(args, args.model_path / model_name)
    vocab = VocabBuilder.build_vocab(args.dataset, 'ner', vocab_type='char', vocab_dir=args.vocab_path)
    train_dataset = NERDatasetNew(args.dataset, split='train', vocab=vocab, data_dir=args.data_path)
    validation_dataset = NERDatasetNew(args.dataset, split='validation', vocab=vocab, data_dir=args.data_path)
    test_dataset = NERDatasetNew(args.dataset, split='test', vocab=vocab, data_dir=args.data_path)

    train_loader = NERDataLoaderNew(train_dataset, batch_size=args.batch, conv_kernel_size=args.cnn_width)
    validation_loader = NERDataLoaderNew(validation_dataset, batch_size=args.batch, conv_kernel_size=args.cnn_width)
    test_loader = NERDataLoaderNew(test_dataset, batch_size=args.batch, conv_kernel_size=args.cnn_width)

    if args.pretrained_embeddings == 'no':
        embeddings = None
    else:
        raise NotImplementedError('Pretrained embeddings aren\'t implemented yet')

    model = BiLSTMConv(
        output_size=train_dataset.get_tags_number(),
        conv_width=args.cnn_width,
        conv_output_size=args.cnn_output,
        hidden_size=args.lstm_state,
        lstm_layers=args.lstm_layers,
        dropout_rate=args.dropout,
        char_vocab_size=len(vocab.chars),
        char_embedding_dim=args.char_embedding_size,
        word_vocab_size=len(vocab),
        word_embedding_dim=args.word_embedding_size,
        embeddings=embeddings,
        use_word_features=args.word_features,
        use_char_features=args.char_features
    )

    trainer = Trainer(
        train_step=train_step,
        epochs=args.epochs,
        max_gradient_norm=None,
        model_save_path=args.model_path,
        log_save_path=args.logs_path,
        model_name=model_name,
        use_cuda=args.use_gpu,
        load_checkpoint=args.load_checkpoint,
        scores=[Precision(), Recall(), F1Score()]
    )
    trainer.set_models(
        bilstm_cnn=model
    )
    trainer.set_criterion(
        cross_entropy=nn.CrossEntropyLoss()
    )
    trainer.set_optimizer(
        sgd=torch.optim.SGD(model.parameters(), lr=args.lr)
    )
    trainer.train(train_loader, validation_loader, verbosity=500, save_interval=500)
    trainer.eval(test_loader)


if __name__ == '__main__':
    main()
