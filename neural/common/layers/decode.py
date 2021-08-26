import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class CRF(nn.Module):
    def __init__(self, labels_number: int):
        super().__init__()
        self.transitions = nn.Parameter(torch.randn(labels_number, labels_number))
        self.start_scores = nn.Parameter(torch.randn(labels_number))
        self.end_scores = nn.Parameter(torch.randn(labels_number))

    def __score_numerator(self, predictions: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        batch_indexes = torch.arange(predictions.shape[1])

        score = self.start_scores[targets[0]] + predictions[0, batch_indexes, targets[0]]
        for i in range(1, predictions.shape[0]):
            step_score = self.transitions[targets[i - 1], targets[i]] + predictions[i, batch_indexes, targets[i]]
            score = score + step_score * mask[i]

        last_indexes = torch.sum(mask.int(), dim=0) - 1  # Get indexes of last tokens before padding
        score = score + self.end_scores[targets[last_indexes, batch_indexes]]

        return score

    def __score_denominator(self, predictions: Tensor, mask: Tensor) -> Tensor:
        score = self.start_scores + predictions[0]
        mask = mask.bool()

        for i in range(1, predictions.shape[0]):
            score_step = score.unsqueeze(2)
            predictions_step = predictions[i].unsqueeze(1)
            score_step = score_step + self.transitions + predictions_step
            score_step = torch.logsumexp(score_step, dim=1)
            score = torch.where(mask[i].unsqueeze(1), score_step, score)

        score = score + self.end_scores
        return torch.logsumexp(score, dim=1)

    def forward(self, predictions: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        numerator = self.__score_numerator(predictions, targets, mask)
        denominator = self.__score_denominator(predictions, mask)
        score = denominator - numerator  # Change order to get positive value

        return torch.mean(score)

    def decode(self, predictions: Tensor, mask: Tensor = None) -> Tensor:
        sequence_length, batch_size, _ = predictions.shape
        if mask is None:
            mask = torch.ones((sequence_length, batch_size), device=predictions.device, dtype=torch.bool)
        else:
            mask = mask.bool()

        score = self.start_scores + predictions[0]  # Starting scores
        best_indexes = []

        for i in range(1, sequence_length):
            step_score = score.unsqueeze(2)
            step_predictions = predictions[i].unsqueeze(1)

            step_score = step_score + self.transitions + step_predictions
            step_score, indexes = torch.max(step_score, dim=1)

            score = torch.where(mask[i].unsqueeze(1), step_score, score)
            best_indexes.append(indexes)

        score = score + self.end_scores  # Ending scores
        last_indexes = torch.sum(mask.int(), dim=0) - 1  # Get indexes of last tokens before padding
        best_tags_list = []

        for i in range(batch_size):
            best_tags = [torch.argmax(score[i], dim=0).item()]
            best_tags += [indexes[i][best_tags[-1]].item() for indexes in reversed(best_indexes[:last_indexes[i]])]
            best_tags.reverse()
            best_tags_list.append(torch.tensor(best_tags, device=predictions.device))

        result = pad_sequence(best_tags_list, padding_value=-1)

        return result
