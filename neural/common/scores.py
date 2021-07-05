from typing import Any

import sklearn.metrics as metrics
import torch
import torch.nn as nn
from rouge_score import rouge_scorer
from torch import Tensor
from torchtext.vocab import Vocab


class Score:
    def __init__(self, **kwargs: Any):
        for name, value in kwargs.items():
            self.__dict__[name] = value

    def __str__(self) -> str:
        scores = sorted(self.__dict__)
        return ', '.join(f'{score}: {self.__dict__[score]}' for score in scores)

    def __repr__(self) -> str:
        return self.__str__()


class Accuracy(nn.Module):
    @staticmethod
    def forward(predictions: Tensor, target: Tensor) -> float:
        labels = torch.argmax(predictions, dim=-1)
        return metrics.accuracy_score(torch.flatten(target.cpu()), torch.flatten(labels.cpu()))


class F1Score(nn.Module):
    def __init__(self, average: str):
        super().__init__()
        self.average = average

    def forward(self, predictions: Tensor, target: Tensor) -> float:
        labels = torch.argmax(predictions, dim=-1)
        return metrics.f1_score(torch.flatten(target.cpu()), torch.flatten(labels.cpu()), average=self.average)


class RougeScore(nn.Module):
    def __init__(self, vocab: Vocab, *score_types: str):
        super().__init__()
        self.vocab = vocab
        self.score_types = score_types
        self.scorer = rouge_scorer.RougeScorer(score_types, use_stemmer=False)

    def forward(self, predictions: Tensor, target: Tensor) -> Score:
        labels = torch.argmax(predictions, dim=-1)
        scores = dict.fromkeys(self.score_types, 0)
        batch_size = labels.shape[1]
        for i in range(batch_size):
            hypothesis = ' '.join(self.vocab.itos[token] for token in labels[:, i] if token != 0)
            reference = ' '.join(self.vocab.itos[token] for token in target[:, i] if token != 0)
            rouge = self.scorer.score(hypothesis, reference)
            for name, value in rouge.items():
                scores[name] += value.fmeasure

        return Score(**{name: value / batch_size for name, value in scores.items()})
