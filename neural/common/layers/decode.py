from __future__ import annotations

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


class BeamSearchNode:
    def __init__(self, bos_index: int, eos_index: int, cyclic_input: Tuple[Any, ...], device: str,
                 create_empty: bool = False):
        self.score = 0.
        self.sequence = [torch.tensor(bos_index, device=device)] if not create_empty else []
        self.predictions = []
        self.cyclic_inputs = [cyclic_input] if not create_empty else []
        self.decoder_outputs = []
        self.eos_mask = 1
        self.eos_index = eos_index

    @classmethod
    def create_new_node(cls, node: BeamSearchNode, token: Tensor, score: float, prediction: Tensor,
                        cyclic_inputs: Tuple[Any, ...], decoder_outputs: Tuple[Any, ...]) -> BeamSearchNode:
        new_node = cls(0, 0, (), '', create_empty=True)  # Create empty node and override its data
        new_node.score = node.score + score * node.eos_mask
        new_node.sequence = node.sequence + [token]
        new_node.predictions = node.predictions + [prediction]
        new_node.cyclic_inputs = node.cyclic_inputs + [cyclic_inputs]
        new_node.decoder_outputs = node.decoder_outputs + [decoder_outputs]
        new_node.eos_index = node.eos_index
        new_node.eos_mask = node.eos_mask
        if new_node.eos_mask == 1 and token == node.eos_index:
            new_node.eos_mask = 0

        return new_node

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, BeamSearchNode):
            raise ValueError(f'{self.__class__.__name__} can\'t be compared with different classes.')

        return self.score < other.score

    def __str__(self) -> str:
        return f'{self.__class__.__name__}: (score: {self.score}, sequence: {self.sequence})'

    def __repr__(self) -> str:
        return self.__str__()


class BeamSearchDecoder(nn.Module, ABC):
    def __init__(self, bos_index: int, eos_index: int, max_output_length: int, beam_size: int,
                 embedding_before_step: bool = True):
        super().__init__()
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.max_output_length = max_output_length
        self.beam_size = beam_size
        self.embedding_before_step = embedding_before_step

    @abstractmethod
    def decoder_step(self, decoder_input: Tensor, cyclic_inputs: Tuple[Any, ...],
                     constant_inputs: Tuple[Any, ...]) -> Tuple[Tensor, Tuple[Any, ...], Tuple[Any, ...]]:
        pass

    def _get_predicted_tokens(self, predictions: Tensor) -> Tensor:
        return torch.argmax(predictions, dim=-1)

    def _preprocess_decoder_inputs(self, decoder_inputs: Tensor) -> Tensor:
        return decoder_inputs

    def _preprocess_beam_search_inputs(self, nodes: Tuple[BeamSearchNode]) -> Tuple[Tensor, Tuple[Any, ...]]:
        decoder_input = torch.stack([node.sequence[-1] for node in nodes])
        cyclic_inputs = self.__merge_batched_data([node.cyclic_inputs[-1] for node in nodes])
        return self._preprocess_decoder_inputs(decoder_input), cyclic_inputs

    def _validate_outputs(self, outputs: Optional[Tensor], teacher_forcing_ratio: float, batch_size: int,
                          device: str) -> Tensor:
        if outputs is None:
            if teacher_forcing_ratio > 0:
                raise AttributeError('During training with teacher forcing reference summaries must be provided.')
            else:
                outputs = torch.full((self.max_output_length, batch_size), self.bos_index, dtype=torch.long,
                                     device=device)

        return outputs

    def __divide_batched_data(self, batched_data: Tuple[Any, ...], batch_size: int) -> List[Tuple[Any, ...]]:
        divided_data = [[] for _ in range(batch_size)]
        for data in batched_data:
            if isinstance(data, Tensor):
                for i in range(batch_size):
                    divided_data[i].append(data[i:i + 1])  # One element slice to keep tensor dim
            elif isinstance(data, tuple):
                for i, divided in enumerate(self.__divide_batched_data(data, batch_size)):
                    divided_data[i].append(divided)
            else:
                for i in range(batch_size):
                    divided_data[i].append(data)

        return [tuple(data) for data in divided_data]

    def __merge_batched_data(self, batched_data: List[Tuple[Any, ...]]) -> Tuple[Any, ...]:
        merged_data = []
        for data in zip(*batched_data):
            if isinstance(data[0], Tensor):
                merged_data.append(torch.cat(data, dim=0))
            elif isinstance(data[0], tuple):
                merged_data.append(self.__merge_batched_data(list(data)))
            else:
                merged_data.append(data[0])

        return tuple(merged_data)

    def __beam_search(self, embedding: nn.Embedding, batch_size: int, device: str, cyclic_inputs: Tuple[Any, ...],
                      constant_inputs: Tuple[Any, ...]) -> Tuple[Tensor, Tensor, List[Tuple[Any, ...]]]:
        # Prepare initial data
        divided_cyclic = self.__divide_batched_data(cyclic_inputs, batch_size)
        search_nodes = [[BeamSearchNode(self.bos_index, self.eos_index, cyclic, device)] for cyclic in divided_cyclic]

        for _ in range(self.max_output_length):
            new_nodes = [[] for _ in range(batch_size)]
            # For each batch of nodes
            for nodes in zip(*search_nodes):
                # Divided data is merged into single batch
                decoder_input, cyclic_inputs = self._preprocess_beam_search_inputs(nodes)
                if self.embedding_before_step:
                    decoder_input = embedding(decoder_input)
                predictions, cyclic_inputs, decoder_out = self.decoder_step(decoder_input, cyclic_inputs,
                                                                            constant_inputs)
                # Divide batched data and get top predictions
                cyclic_inputs = self.__divide_batched_data(cyclic_inputs, batch_size)
                decoder_out = self.__divide_batched_data(decoder_out, batch_size)
                top_scores, top_tokens = torch.topk(predictions, self.beam_size)
                top_scores = torch.log(top_scores).tolist()

                # Update nodes with new tokens and scores
                for i, node in enumerate(nodes):
                    batch_nodes = []
                    for token, score in zip(top_tokens[i], top_scores[i]):
                        new_node = BeamSearchNode.create_new_node(node, token, score, predictions[i],
                                                                  cyclic_inputs[i], decoder_out[i])
                        batch_nodes.append(new_node)
                    new_nodes[i] += batch_nodes

            # Set new nodes with `k` best ones
            search_nodes.clear()
            for nodes in new_nodes:
                nodes = sorted(nodes, reverse=True)[:self.beam_size]  # Get only `k` nodes with highest scores
                search_nodes.append(nodes)

        best_nodes = next(zip(*search_nodes))
        # Get sequences without BOS token
        tokens = torch.stack([torch.stack(node.sequence[1:]) for node in best_nodes])
        # Merge data into single batch
        predictions = torch.stack([torch.stack(node.predictions) for node in best_nodes])
        decoder_out = list(self.__merge_batched_data([node.decoder_outputs for node in best_nodes]))
        tokens = torch.transpose(tokens, 0, 1)
        predictions = torch.transpose(predictions, 0, 1)

        return predictions, tokens, decoder_out

    def __decode_training(self, outputs: Optional[Tensor], embedding: nn.Embedding, teacher_forcing_ratio: float,
                          batch_size: int, device: str, cyclic_inputs: Tuple[Any, ...],
                          constant_inputs: Tuple[Any, ...]) -> Tuple[Tensor, Tensor, List[Tuple[Any, ...]]]:
        outputs = self._validate_outputs(outputs, teacher_forcing_ratio, batch_size, device)
        sequence_length = outputs.shape[0]  # If data is not padded to max, decoding steps are shorter than max
        decoder_input = outputs[0, :]
        predictions = []
        predicted_tokens = []
        decoder_outputs = []
        for i in range(sequence_length):
            decoder_input = self._preprocess_decoder_inputs(decoder_input)
            decoder_input = embedding(decoder_input)
            prediction, cyclic_inputs, decoder_out = self.decoder_step(decoder_input, cyclic_inputs, constant_inputs)
            predictions.append(prediction)
            decoder_outputs.append(decoder_out)

            tokens = self._get_predicted_tokens(prediction)
            tokens = tokens.detach()
            predicted_tokens.append(tokens)

            if i + 1 < sequence_length:
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

    def forward(self, outputs: Optional[Tensor], embedding: nn.Embedding, teacher_forcing_ratio: float, batch_size: int,
                device: str, cyclic_inputs: Tuple[Any, ...],
                constant_inputs: Tuple[Any, ...]) -> Tuple[Tensor, Tensor, List[Tuple[Any, ...]]]:
        if self.training:
            return self.__decode_training(outputs, embedding, teacher_forcing_ratio, batch_size, device, cyclic_inputs,
                                          constant_inputs)
        else:
            if self.beam_size > 1:
                return self.__beam_search(embedding, batch_size, device, cyclic_inputs, constant_inputs)
            else:  # Beam size = 1 is the same as greedy decoding
                return self.__decode_training(None, embedding, 0, batch_size, device,
                                              cyclic_inputs, constant_inputs)
