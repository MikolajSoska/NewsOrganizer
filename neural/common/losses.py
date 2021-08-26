import abc
from typing import List, Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import nll_loss
from torchtext.vocab import Vocab

from neural.common.scores import ROUGE
from neural.common.utils import add_words_to_vocab, remove_words_from_vocab


class LossWithReduction(nn.Module, abc.ABC):
    def __init__(self, reduction: str):
        super().__init__()
        if reduction == 'mean':
            self.reduction = torch.mean
        elif reduction == 'sum':
            self.reduction = torch.sum
        elif reduction == 'none':
            self.reduction = None
        else:
            raise ValueError(f'{reduction} is not a valid value for reduction')

    @abc.abstractmethod
    def forward(self, *args: Any) -> Tensor:
        pass


class SummarizationLoss(LossWithReduction):
    def __init__(self, epsilon: float = 1e-12, reduction: str = 'mean'):
        super().__init__(reduction)
        self.epsilon = epsilon

    def forward(self, predictions: Tensor, targets: Tensor, target_lengths: Tensor) -> Tensor:
        padding_mask = torch.clip(targets, min=0, max=1)
        gathered_probabilities = torch.gather(predictions, 2, targets.unsqueeze(2)).squeeze()
        loss = -torch.log(gathered_probabilities + self.epsilon) * padding_mask
        loss = torch.sum(loss, dim=0) / target_lengths

        return self.reduction(loss)


class CoverageLoss(LossWithReduction):
    def __init__(self, weight: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction)
        self.weight = weight

    def forward(self, attention: Tensor, coverage: Tensor, targets: Tensor) -> Tensor:
        padding_mask = torch.clip(targets, min=0, max=1)
        loss = self.weight * torch.sum(torch.min(attention, coverage), dim=1) * padding_mask
        if self.reduction is not None:
            return self.reduction(loss)
        else:
            return loss


class LabelSmoothingCrossEntropy(LossWithReduction):
    def __init__(self, smoothing: float, reduction: str = 'mean'):
        super().__init__(reduction)
        assert 0 <= smoothing < 1, 'Smoothing value has to be from 0 (inclusively) to 1 (exclusively)'
        self.smoothing = smoothing
        self.confidence = 1 - smoothing

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        predictions = torch.flatten(predictions, end_dim=1)
        targets = torch.flatten(targets)
        class_number = predictions.shape[-1]

        predictions_log = torch.log_softmax(predictions, dim=-1)
        encoding = torch.full_like(predictions_log, self.smoothing / (class_number - 1))
        encoding = torch.scatter(encoding, 1, targets.unsqueeze(1), self.confidence)
        loss = torch.sum(-encoding * predictions_log, dim=-1)

        return self.reduction(loss)


class MLLoss(LossWithReduction):
    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction)

    def forward(self, predictions: Tensor, targets: Tensor, target_lengths: Tensor) -> Tensor:
        predictions = torch.transpose(predictions, 1, 2)
        padding_mask = torch.clip(targets, min=0, max=1)
        log_probabilities = torch.log(predictions + 1e-4)  # Add small constant to prevent infinity from log(0)
        loss = nll_loss(log_probabilities, targets, reduction='none')
        loss = torch.sum(loss * padding_mask, dim=0) / target_lengths

        return self.reduction(loss)


class PolicyLearning(LossWithReduction):
    def __init__(self, vocab: Vocab, reduction: str = 'mean'):
        super().__init__(reduction)
        self.vocab = vocab
        self.reward = ROUGE(vocab, 'rougeL')  # Use ROUGE-L score as reward function

    def __compute_reward(self, predictions: Tensor, targets: Tensor, oov_list: Tuple[List[str]]) -> Tensor:
        batch_size = targets.shape[1]
        scores = []
        # Due to different OOV words for each sequence in a batch, it has to scored separately
        for i in range(batch_size):
            add_words_to_vocab(self.vocab, oov_list[i])
            prediction_tokens = predictions[:, i].unsqueeze(dim=1)
            target_tokens = targets[:, i].unsqueeze(dim=1)
            scores.append(self.reward.score(prediction_tokens, target_tokens)['ROUGE-L'])
            remove_words_from_vocab(self.vocab, oov_list[i])

        return torch.tensor(scores, device=predictions.device)

    def forward(self, log_probabilities: Tensor, predicted_tokens: Tensor, baseline_tokens: Tensor,
                targets: Tensor, oov_list: Tuple[List[str]]) -> Tensor:
        predicted_reward = self.__compute_reward(predicted_tokens, targets, oov_list)
        baseline_reward = self.__compute_reward(baseline_tokens, targets, oov_list)
        loss = (baseline_reward - predicted_reward) * torch.sum(log_probabilities, dim=0)

        return self.reduction(loss)


class MixedRLLoss(nn.Module):
    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = gamma

    def forward(self, ml_loss: Tensor, rl_loss: Tensor) -> Tensor:
        return self.gamma * rl_loss + (1 - self.gamma) * ml_loss
