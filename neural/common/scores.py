from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Any, Union

import sklearn.metrics as metrics
import torch
from nltk.translate import meteor_score
from rouge_score import rouge_scorer
from torch import Tensor
from torchtext.vocab import Vocab

import neural.common.utils as utils
from neural.common.data.vocab import SpecialTokens


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

        scores_keys = list(self.__dict__.keys()) + list(score.__dict__.keys())
        first_scores = defaultdict(int, self.__dict__)
        second_scores = defaultdict(int, score.__dict__)

        scores_sum = {score: first_scores[score] + second_scores[score] for score in scores_keys}
        return ScoreValue(**scores_sum)

    def __truediv__(self, number: Union[int, float]) -> ScoreValue:
        new_scores = {score: value / number for score, value in self.__dict__.items()}
        return ScoreValue(**new_scores)

    def __len__(self) -> int:
        return len(self.__dict__)

    def __getitem__(self, item: str) -> float:
        return self.__dict__[item]


class Scorer(ABC):
    @abstractmethod
    def score(self, predictions: Tensor, target: Tensor) -> ScoreValue:
        pass

    @staticmethod
    def _get_labels(predictions: Tensor) -> Tensor:
        if isinstance(predictions.cpu(), torch.LongTensor):
            return predictions
        else:
            return torch.argmax(predictions, dim=-1)


class Precision(Scorer):
    def __init__(self, labels: List[int] = None, average: str = 'micro'):
        super().__init__()
        self.labels = labels
        self.average = average

    def score(self, predictions: Tensor, target: Tensor) -> ScoreValue:
        labels = self._get_labels(predictions)
        precision = metrics.precision_score(torch.flatten(target.cpu()), torch.flatten(labels.cpu()),
                                            average=self.average, zero_division=0, labels=self.labels)
        return ScoreValue(Precision=precision)


class Recall(Scorer):
    def __init__(self, labels: List[int] = None, average: str = 'micro'):
        super().__init__()
        self.labels = labels
        self.average = average

    def score(self, predictions: Tensor, target: Tensor) -> ScoreValue:
        labels = self._get_labels(predictions)
        precision = metrics.recall_score(torch.flatten(target.cpu()), torch.flatten(labels.cpu()),
                                         average=self.average, zero_division=0, labels=self.labels)
        return ScoreValue(Recall=precision)


class F1Score(Scorer):
    def __init__(self, labels: List[int] = None, average: str = 'micro'):
        super().__init__()
        self.labels = labels
        self.average = average

    def score(self, predictions: Tensor, target: Tensor) -> ScoreValue:
        labels = self._get_labels(predictions)
        f1_score = metrics.f1_score(torch.flatten(target.cpu()), torch.flatten(labels.cpu()), average=self.average,
                                    zero_division=0, labels=self.labels)
        return ScoreValue(F1=f1_score)


class ROUGE(Scorer):
    def __init__(self, vocab: Vocab, *score_types: str):
        super().__init__()
        self.vocab = vocab
        self.score_types = score_types
        self.scorer = rouge_scorer.RougeScorer(score_types, use_stemmer=False)
        self.eos_index = vocab.stoi[SpecialTokens.EOS.value]

    def score(self, predictions: Tensor, target: Tensor) -> ScoreValue:
        labels = self._get_labels(predictions)
        scores = dict.fromkeys(self.score_types, 0)
        batch_size = labels.shape[1]
        labels = utils.clean_predicted_tokens(labels, self.eos_index)
        for i in range(batch_size):
            hypothesis = utils.remove_unnecessary_padding(labels[:, i])
            reference = utils.remove_unnecessary_padding(target[:, i])

            hypothesis = utils.tensor_to_string(self.vocab, hypothesis)
            reference = utils.tensor_to_string(self.vocab, reference)

            rouge = self.scorer.score(hypothesis, reference)
            for name, value in rouge.items():
                scores[name] += value.fmeasure

        scores = {f"ROUGE-{name.replace('rouge', '')}": value / batch_size for name, value in scores.items()}
        return ScoreValue(**scores)


class METEOR(Scorer):  # TODO add exact match METEOR
    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        self.eos_index = vocab.stoi[SpecialTokens.EOS.value]

    def score(self, predictions: Tensor, target: Tensor) -> ScoreValue:
        labels = self._get_labels(predictions)
        batch_size = labels.shape[1]
        meteor = 0
        labels = utils.clean_predicted_tokens(labels, self.eos_index)
        for i in range(batch_size):
            hypothesis = utils.remove_unnecessary_padding(labels[:, i])
            reference = utils.remove_unnecessary_padding(target[:, i])

            hypothesis = utils.tensor_to_string(self.vocab, hypothesis)
            reference = utils.tensor_to_string(self.vocab, reference)
            meteor += meteor_score.single_meteor_score(reference, hypothesis)

        meteor = meteor / batch_size
        return ScoreValue(METEOR=meteor)
