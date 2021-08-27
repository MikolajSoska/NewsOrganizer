from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional

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


class BaseRNNDecoder(nn.Module, ABC):
    def __init__(self, bos_index: int, max_output_length: int, embedding: nn.Embedding):
        super().__init__()
        self.bos_index = bos_index
        self.max_output_length = max_output_length
        self.embedding = embedding

    def __validate_outputs(self, outputs: Optional[Tensor], teacher_forcing_ratio: float, batch_size: int,
                           device: str) -> Tuple[Tensor, float]:
        if self.training:
            if outputs is None:
                if teacher_forcing_ratio > 0:
                    raise AttributeError('During training with teacher forcing reference summaries must be provided.')
                else:
                    outputs = torch.full((self.max_output_length, batch_size), self.bos_index, dtype=torch.long,
                                         device=device)
        else:  # In validation phase never use passed summaries
            outputs = torch.full((self.max_output_length, batch_size), self.bos_index, dtype=torch.long, device=device)
            teacher_forcing_ratio = 0.  # In validation phase teacher forcing is not used

        return outputs, teacher_forcing_ratio

    def forward(self, outputs: Optional[Tensor], teacher_forcing_ratio: float, batch_size: int, device: str,
                cyclic_inputs: Tuple[Any, ...], constant_inputs: Tuple[Any, ...]) -> Tuple[Tensor, Tensor,
                                                                                           List[Tuple[Any, ...]]]:
        outputs, teacher_forcing_ratio = self.__validate_outputs(outputs, teacher_forcing_ratio, batch_size, device)
        decoder_input = outputs[0, :]

        predictions = []
        predicted_tokens = []
        decoder_outputs = []
        for i in range(self.max_output_length):
            decoder_input = self.embedding(decoder_input)
            prediction, cyclic_inputs, decoder_out = self.decoder_step(decoder_input, cyclic_inputs, constant_inputs)
            predictions.append(prediction)
            decoder_outputs.append(decoder_out)

            tokens = self.get_new_decoder_inputs(prediction)
            tokens = tokens.detach()
            predicted_tokens.append(tokens)

            if i + 1 < self.max_output_length:
                if teacher_forcing_ratio <= 1:
                    use_predictions = torch.as_tensor(torch.rand(batch_size, device=device) >= teacher_forcing_ratio)
                    # Depending on the value of `use_predictions` in next step decoder will use predicted tokens or
                    # ground truth values (teacher forcing)
                    decoder_input = torch.where(use_predictions, tokens, outputs[i + 1, :])
                else:  # If teacher forcing is used in every step
                    decoder_input = outputs[i + 1, :]

        predictions = torch.stack(predictions)
        predicted_tokens = torch.stack(predicted_tokens)

        return predictions, predicted_tokens, decoder_outputs

    @abstractmethod
    def decoder_step(self, decoder_input: Tensor, cyclic_inputs: Tuple[Any, ...],
                     constant_inputs: Tuple[Any, ...]) -> Tuple[Tensor, Tuple[Any, ...], Tuple[Any, ...]]:
        pass

    @abstractmethod
    def get_new_decoder_inputs(self, predictions: Tensor) -> Tensor:
        pass
