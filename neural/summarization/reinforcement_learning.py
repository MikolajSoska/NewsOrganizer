from typing import Tuple, Optional, Any, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical

import neural.common.layers as layers
from neural.common.layers.decode import BeamSearchDecoder


class Encoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.lstm = layers.PackedRNN(nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size // 2,
                                             bidirectional=True))
        self.features = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.transform_state = layers.View(-1, hidden_size)

    def forward(self, inputs: Tensor, inputs_lengths: Tensor) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
        out, (hidden, cell) = self.lstm(inputs, inputs_lengths)
        features = self.features(out)
        hidden = self.transform_state(hidden)
        cell = self.transform_state(cell)

        return out, features, (hidden, cell)


class IntraTemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
        )
        self.attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
            layers.Permute(1, 2, 0),
            layers.Exponential()
        )
        self.normalize = layers.Normalize(-1)
        self.context = layers.SequentialMultiInput(
            layers.MatrixProduct(),
            layers.Transpose(0, 1)
        )

    def forward(self, outputs_hidden: Tensor, encoder_out: Tensor, encoder_features: Tensor, encoder_mask: Tensor,
                temporal_scores_sum: Tensor = None) -> Tuple[Tensor, Tensor, Tensor]:
        decoder_features = self.features(outputs_hidden)
        temporal_attention = self.attention(decoder_features + encoder_features)

        if temporal_scores_sum is not None:
            attention = temporal_attention / temporal_scores_sum
            temporal_scores_sum = temporal_attention + temporal_scores_sum
        else:
            attention = temporal_attention
            temporal_scores_sum = temporal_attention

        attention = attention * encoder_mask
        attention = self.normalize(attention)
        encoder_out = torch.transpose(encoder_out, 0, 1)
        context = self.context(attention, encoder_out)

        return context, attention, temporal_scores_sum


class IntraDecoderAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention_first = nn.Sequential(
            layers.Transpose(0, 1),
            nn.Linear(hidden_size, hidden_size, bias=False),
            layers.Permute(1, 2, 0)
        )
        self.attention_second = layers.SequentialMultiInput(
            layers.MatrixProduct(),
            nn.Softmax(dim=-1)
        )
        self.context = layers.SequentialMultiInput(
            layers.MatrixProduct(),
            layers.Transpose(0, 1)
        )

    def forward(self, decoder_hidden: Tensor, previous_hidden: Tensor = None) -> Tuple[Tensor, Tensor]:
        if previous_hidden is None:
            context = torch.zeros_like(decoder_hidden)
            return context, decoder_hidden.transpose(0, 1)

        decoder_hidden = torch.transpose(decoder_hidden, 0, 1)
        attention = self.attention_first(previous_hidden)
        attention = self.attention_second(decoder_hidden, attention)
        context = self.context(attention, previous_hidden)
        previous_hidden = torch.cat((previous_hidden, decoder_hidden), dim=1)

        return context, previous_hidden


class Decoder(BeamSearchDecoder):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, bos_index: int, eos_index: int,
                 unk_index: int, max_summary_length: int, use_intra_attention: bool, beam_size: int):
        super().__init__(bos_index, eos_index, max_summary_length, beam_size)
        self.vocab_size = vocab_size
        self.unk_index = unk_index
        self.lstm = nn.LSTMCell(input_size=embedding_dim, hidden_size=hidden_size)
        self.encoder_attention = IntraTemporalAttention(hidden_size)
        if use_intra_attention:
            self.decoder_attention = IntraDecoderAttention(hidden_size)
            hidden_multiplier = 3
        else:
            self.decoder_attention = None
            hidden_multiplier = 2

        self.vocab_distribution = layers.SequentialMultiInput(
            layers.Concatenate(-1),
            layers.Squeeze(0),
            nn.Linear(hidden_multiplier * hidden_size, embedding_dim, bias=False),
            nn.Linear(embedding_dim, vocab_size),
            nn.Softmax(dim=1)
        )
        self.pointer_probability = layers.SequentialMultiInput(
            layers.Concatenate(-1),
            layers.Squeeze(0),
            nn.Linear(hidden_multiplier * hidden_size, 1),
            nn.Sigmoid()
        )
        # Used in during decoding loop
        self.train_rl = False
        self.log_probabilities = []

    def start_decoding(self, train_rl: bool) -> None:
        self.train_rl = train_rl
        self.log_probabilities.clear()

    def forward(self, outputs: Optional[Tensor], embedding: nn.Embedding, teacher_forcing_ratio: float, batch_size: int,
                device: str, cyclic_inputs: Tuple[Any, ...],
                constant_inputs: Tuple[Any, ...]) -> Tuple[Tensor, Tensor, List[Tuple[Any, ...]]]:
        predictions, tokens, attention_list = super().forward(outputs, embedding, teacher_forcing_ratio, batch_size,
                                                              device, cyclic_inputs, constant_inputs)
        decoder_output = [(log_prob, attention[0]) for log_prob, attention in
                          zip(self.log_probabilities, attention_list)]
        return predictions, tokens, decoder_output

    def decoder_step(self, decoder_input: Tensor, cyclic_inputs: Tuple[Tuple[Tensor, Tensor], Tensor, Tensor],
                     constant_inputs: Tuple[Tensor, Tensor, Tensor, Tensor, int]) -> \
            Tuple[Tensor, Tuple[Tuple[Tensor, Tensor], Tensor, Tensor], Tuple[Tensor]]:
        encoder_hidden, temporal_scores_sum, previous_hidden = cyclic_inputs
        encoder_out, encoder_features, encoder_mask, inputs_extended, oov_size = constant_inputs

        hidden, cell = self.lstm(decoder_input, encoder_hidden)
        decoder_hidden = torch.unsqueeze(hidden, 0)
        encoder_attention, temporal_attention, temporal_scores_sum = \
            self.encoder_attention(decoder_hidden, encoder_out, encoder_features, encoder_mask, temporal_scores_sum)

        if self.decoder_attention is not None:
            decoder_attention, previous_hidden = self.decoder_attention(decoder_hidden, previous_hidden)
            vocab_distribution = self.vocab_distribution(decoder_hidden, encoder_attention, decoder_attention)
            pointer_probability = self.pointer_probability(decoder_hidden, encoder_attention, decoder_attention)
        else:
            vocab_distribution = self.vocab_distribution(decoder_hidden, encoder_attention)
            pointer_probability = self.pointer_probability(decoder_hidden, encoder_attention)

        vocab_distribution = (1 - pointer_probability) * vocab_distribution
        attention = pointer_probability * temporal_attention.squeeze()

        if oov_size > 0:  # Add distribution for OOV words (with 0 value) to match dims
            batch_size = vocab_distribution.shape[0]
            oov_zeros = torch.zeros((batch_size, oov_size), device=vocab_distribution.device)
            vocab_distribution = torch.cat((vocab_distribution, oov_zeros), dim=1)

        final_distribution = torch.scatter_add(vocab_distribution, 1, inputs_extended.transpose(1, 0), attention)

        return final_distribution, ((hidden, cell), temporal_scores_sum, previous_hidden), (temporal_attention,)

    def _get_predicted_tokens(self, predictions: Tensor) -> Tensor:
        if self.train_rl:  # In RL training draw tokens from categorical distribution
            distribution = Categorical(predictions)
            tokens = distribution.sample()
            self.log_probabilities.append(distribution.log_prob(tokens))
        else:  # Use simple greedy approach
            tokens = torch.argmax(predictions, dim=1)
            self.log_probabilities.append(None)  # To match other outputs length

        return tokens

    def _preprocess_decoder_inputs(self, decoder_inputs: Tensor) -> Tensor:
        decoder_inputs[decoder_inputs >= self.vocab_size] = self.unk_index  # Remove OOV tokens

        return decoder_inputs.detach()  # This can be done during training so detach in necessary


class ReinforcementSummarization(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_summary_length: int, bos_index: int, eos_index: int,
                 unk_index: int, beam_size: int, embedding_dim: int = None, embeddings: Tensor = None,
                 use_intra_attention: bool = True):
        if embeddings is None and embedding_dim is None:
            raise ValueError('Either embeddings vector or embedding dim must be passed.')

        super().__init__()
        self.vocab_size = vocab_size
        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=0)
            embedding_dim = self.embedding.embedding_dim
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.encoder = Encoder(embedding_dim, hidden_size)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_size, bos_index, eos_index, unk_index,
                               max_summary_length, use_intra_attention, beam_size)

    def forward(self, inputs: Tensor, inputs_length: Tensor, inputs_extended: Tensor, oov_size: int,
                outputs: Tensor = None, teacher_forcing_ratio: float = 1.0,
                train_rl: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        device = inputs.device
        batch_size = inputs.shape[1]
        inputs_embedded = self.embedding(inputs)
        encoder_out, encoder_features, encoder_hidden = self.encoder(inputs_embedded, inputs_length)
        encoder_mask = torch.clip(inputs, max=1)
        encoder_mask = torch.transpose(encoder_mask, 0, 1)
        encoder_mask = torch.unsqueeze(encoder_mask, 1)

        self.decoder.start_decoding(train_rl)
        predictions, tokens, outputs = self.decoder(outputs, self.embedding, teacher_forcing_ratio, batch_size, device,
                                                    cyclic_inputs=(encoder_hidden, None, None),
                                                    constant_inputs=(encoder_out, encoder_features, encoder_mask,
                                                                     inputs_extended, oov_size))
        log_probabilities, attention = zip(*outputs)
        attention = torch.cat(attention, dim=1)
        if train_rl:  # Return log probabilities used in RL loss function
            return torch.stack(log_probabilities), tokens, attention
        else:
            return predictions, tokens, attention
