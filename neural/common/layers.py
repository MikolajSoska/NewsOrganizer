from typing import Tuple, Any, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        x_reshape = x.contiguous().view(-1, x.size(-1))

        y = self.module(x_reshape)
        y = y.view(-1, x.size(1), y.size(-1))

        return y


class PackedRNN(nn.Module):
    def __init__(self, rnn_module: nn.RNNBase):
        super().__init__()
        self.rnn_module = rnn_module

    def forward(self, sequence: Tensor, sequence_lengths: Tensor,
                *rnn_args: Any) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, ...]]]:
        lengths = sequence_lengths.squeeze().tolist()
        if sequence_lengths.shape[0] == 1:
            lengths = [lengths]

        sequence_packed = pack_padded_sequence(sequence, lengths, enforce_sorted=False)
        output, hidden = self.rnn_module(sequence_packed, *rnn_args)
        output_padded, _ = pad_packed_sequence(output, total_length=int(sequence_lengths.max()))

        return output_padded, hidden


class View(nn.Module):
    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def forward(self, x_in: Tensor) -> Tensor:
        return x_in.view(self.shape)


class Permute(nn.Module):
    def __init__(self, *dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    def forward(self, x_in: Tensor) -> Tensor:
        return x_in.permute(self.dimensions)


class Transpose(nn.Module):
    def __init__(self, first_dimension: int, second_dimension: int):
        super().__init__()
        self.first_dimension = first_dimension
        self.second_dimension = second_dimension

    def forward(self, x_in: Tensor) -> Tensor:
        return x_in.transpose(self.first_dimension, self.second_dimension)


class Squeeze(nn.Module):
    def __init__(self, dimension: int = None):
        super().__init__()
        self.dimension = dimension

    def forward(self, x_in: Tensor) -> Tensor:
        if self.dimension is not None:  # Can't passed None as dim to this method
            return x_in.squeeze(dim=self.dimension)
        else:
            return x_in.squeeze()


class Unsqueeze(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, x_in: Tensor) -> Tensor:
        return x_in.unsqueeze(dim=self.dimension)


class Contiguous(nn.Module):
    @staticmethod
    def forward(tensor: Tensor) -> Tensor:
        return tensor.contiguous()


class Concatenate(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, *tensors: Tensor) -> Tensor:
        return torch.cat(tensors, dim=self.dimension)


class Normalize(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, x_in: Tensor) -> Tensor:
        factor = torch.sum(x_in, dim=self.dimension, keepdim=True)
        return x_in / factor


class Multiply(nn.Module):
    def __init__(self, value: Union[float, int]):
        super().__init__()
        self.value = value

    def forward(self, x_in: Tensor) -> Tensor:
        return x_in * self.value


class MatrixProduct(nn.Module):
    @staticmethod
    def forward(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        return torch.bmm(tensor_1, tensor_2)


class Exponential(nn.Module):
    @staticmethod
    def forward(x_in: Tensor) -> Tensor:
        return torch.exp(x_in)


class Max(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, x_in: Tensor) -> Tensor:
        return torch.max(x_in, dim=self.dimension)[0]


class Residual(nn.Module):
    def __init__(self, module: nn.Module, input_position: int = 0):
        super().__init__()
        self.module = module
        self.input_position = input_position

    def forward(self, *inputs: Tensor) -> Tensor:
        residual = inputs[self.input_position]
        output = self.module(*inputs)

        return output + residual


class SequentialMultiInput(nn.Sequential):
    def forward(self, *inputs: Any) -> Any:
        for module in self:
            if not isinstance(inputs, tuple):
                inputs = (inputs,)
            inputs = module(*inputs)

        return inputs


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
