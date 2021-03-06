import argparse
import itertools
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Callable, Any

import torch
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torch import Tensor
from torchtext.vocab import Vocab

__detokenizer = TreebankWordDetokenizer()


class JSONPathEncoder(json.JSONEncoder):
    def default(self, data_object: Any) -> Any:
        if isinstance(data_object, Path):
            return str(data_object.as_posix())
        else:
            return super().default(data_object)


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)


def convert_bytes_to_megabytes(bytes_number: int) -> float:
    return round(bytes_number / (1024 ** 2), 2)


def get_device(use_cuda: bool, cuda_index: int = 0, log_method: Callable[[str], None] = print) -> torch.device:
    if use_cuda:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{cuda_index}')
            device_properties = torch.cuda.get_device_properties(device)
            device_name = f'CUDA ({device_properties.name}), ' \
                          f'Total memory: {convert_bytes_to_megabytes(device_properties.total_memory):g} MB'
        else:
            log_method('CUDA device is not available.')
            device = torch.device('cpu')
            device_name = 'CPU'
    else:
        device = torch.device('cpu')
        device_name = 'CPU'

    log_method(f'Using device: {device_name}')
    return device


def convert_input_to_device(inputs: Tuple[Any, ...], device: torch.device) -> Tuple[Any, ...]:
    inputs_in_device = []
    for tensor_in in inputs:
        if isinstance(tensor_in, Tensor):
            tensor_in = tensor_in.to(device=device)
        inputs_in_device.append(tensor_in)

    return tuple(inputs_in_device)


def tensor_to_string(vocab: Vocab, tensor: Tensor, detokenize: bool = False) -> str:
    tokens = []
    if len(tensor.shape) == 0:  # If tensor is 0-dimensional
        tensor = torch.unsqueeze(tensor, 0)

    for token_id in tensor:
        try:  # Try-catch is much faster in this case than if-else
            token = vocab.itos[token_id]
        except IndexError:
            token = vocab.UNK
        tokens.append(token)

    if detokenize:
        sentences = []
        for result, group in itertools.groupby(tokens, lambda item: item == '.'):
            if not result:
                sentence = list(group) + ['.']
                if '<eos>' in sentence:
                    sentence.remove('<eos>')
                sentence = __detokenizer.detokenize(sentence)
                sentences.append(sentence.capitalize())
        tokens = sentences

    return ' '.join(tokens)


def add_words_to_vocab(vocab: Vocab, words: List[str]) -> None:
    for word in words:
        vocab.itos.append(word)
        vocab.stoi[word] = len(vocab.itos)


def remove_words_from_vocab(vocab: Vocab, words: List[str]) -> None:
    for word in words:
        del vocab.itos[-1]
        del vocab.stoi[word]


def dump_args_to_file(args: argparse.Namespace, filepath: Path) -> None:
    args_dict = vars(args)
    filepath.mkdir(parents=True, exist_ok=True)
    with open(filepath / 'args.json', 'w') as file:
        json.dump(args_dict, file, indent=2, cls=JSONPathEncoder)


def load_args_from_file(filepath: Path) -> Dict[str, Any]:
    with open(filepath / 'args.json', 'r') as file:
        args = json.load(file)

    return args


def clean_predicted_tokens(tokens: Tensor, eos_index: int) -> Tensor:
    eos_mask = torch.as_tensor(tokens == eos_index, dtype=torch.bool)
    for i, j in zip(*torch.nonzero(eos_mask, as_tuple=True)):
        tokens[i + 1:, j] = 0

    return tokens


def remove_unnecessary_padding(tokens: Tensor) -> Tensor:
    # Add batch dim if necessary
    if len(tokens.shape) == 1:
        tokens = tokens.unsqueeze(1)

    padding_mask = torch.clip(torch.sum(tokens, dim=1), max=1)
    last_index = torch.sum(padding_mask)
    tokens = torch.squeeze(tokens[:last_index, :])

    return tokens
