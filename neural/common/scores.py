from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Union

import sklearn.metrics as metrics
import torch
from nltk.translate import meteor_score
from rouge_score import rouge_scorer
from torch import Tensor
from torchtext.vocab import Vocab


class ScoreValue:
    def __init__(self, **kwargs: Any):
        for name, value in kwargs.items():
            self.__dict__[name] = value

    def __str__(self) -> str:
        scores = sorted(self.__dict__)
        return ', '.join(f'{score}: {self.__dict__[score]}' for score in scores)

    def __repr__(self) -> str:
        return self.__str__()

    def __add__(self, score: ScoreValue) -> ScoreValue:
        if not isinstance(score, ScoreValue):
            return NotImplemented

        return ScoreValue(**(Counter(self.__dict__) + Counter(score.__dict__)))

    def __truediv__(self, number: Union[int, float]) -> ScoreValue:
        new_scores = {score: value / number for score, value in self.__dict__.items()}
        return ScoreValue(**new_scores)

    def __len__(self) -> int:
        return len(self.__dict__)


class Scorer(ABC):
    @abstractmethod
    def score(self, predictions: Tensor, target: Tensor) -> ScoreValue:
        pass


class Accuracy(Scorer):
    def score(self, predictions: Tensor, target: Tensor) -> ScoreValue:
        labels = torch.argmax(predictions, dim=-1)
        accuracy = metrics.accuracy_score(torch.flatten(target.cpu()), torch.flatten(labels.cpu()))
        return ScoreValue(Accuracy=accuracy)


class F1Score(Scorer):
    def __init__(self, average: str):
        super().__init__()
        self.average = average

    def score(self, predictions: Tensor, target: Tensor) -> ScoreValue:
        labels = torch.argmax(predictions, dim=-1)
        f1_score = metrics.f1_score(torch.flatten(target.cpu()), torch.flatten(labels.cpu()), average=self.average)
        return ScoreValue(F1=f1_score)


class ROUGE(Scorer):
    def __init__(self, vocab: Vocab, *score_types: str):
        super().__init__()
        self.vocab = vocab
        self.score_types = score_types
        self.scorer = rouge_scorer.RougeScorer(score_types, use_stemmer=False)

    def score(self, predictions: Tensor, target: Tensor) -> ScoreValue:
        labels = torch.argmax(predictions, dim=-1)
        scores = dict.fromkeys(self.score_types, 0)
        batch_size = labels.shape[1]
        for i in range(batch_size):
            hypothesis = ' '.join(self.vocab.itos[token] for token in labels[:, i] if token != 0)
            reference = ' '.join(self.vocab.itos[token] for token in target[:, i] if token != 0)
            rouge = self.scorer.score(hypothesis, reference)
            for name, value in rouge.items():
                scores[name] += value.fmeasure

        scores = {f"ROUGE-{name.replace('rouge', '')}": value / batch_size for name, value in scores.items()}
        return ScoreValue(**scores)


class METEOR(Scorer):  # TODO add exact match METEOR
    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    def score(self, predictions: Tensor, target: Tensor) -> ScoreValue:
        labels = torch.argmax(predictions, dim=-1)
        batch_size = labels.shape[1]
        meteor = 0
        for i in range(batch_size):
            hypothesis = ' '.join(self.vocab.itos[token] for token in labels[:, i] if token != 0)
            reference = ' '.join(self.vocab.itos[token] for token in target[:, i] if token != 0)
            meteor += meteor_score.single_meteor_score(reference, hypothesis)

        meteor = meteor / batch_size
        return ScoreValue(METEOR=meteor)