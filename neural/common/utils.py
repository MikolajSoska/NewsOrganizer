import argparse
import json
import random
from pathlib import Path
from typing import List, Callable, Any

import numpy as np
import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from torch import Tensor
from torchtext.vocab import Vocab


class JSONPathEncoder(json.JSONEncoder):
    def default(self, data_object: Any) -> Any:
        if isinstance(data_object, Path):
            return str(data_object.as_posix())
        else:
            return super().default(data_object)


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def convert_bytes_to_megabytes(bytes_number: int) -> float:
    return round(bytes_number / (1024 ** 2), 2)


def get_device(use_cuda: bool, log_method: Callable[[str], None] = print) -> torch.device:
    if use_cuda:
        if torch.cuda.is_available():
            device = torch.device('cuda')
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


def tokenize_text_content(text: str, word_tokenizer: Callable = None, sentence_tokenizer: Callable = None) -> List[str]:
    if sentence_tokenizer is None:
        sentence_tokenizer = sent_tokenize
    if word_tokenizer is None:
        word_tokenizer = word_tokenize

    content = []
    for sentence in sentence_tokenizer(text):
        for word in word_tokenizer(sentence):
            content.append(word)

    return content


def tensor_to_string(vocab: Vocab, tensor: Tensor) -> str:
    tokens = []
    for token_id in tensor:
        try:  # Try-catch is much faster in this case than if-else
            token = vocab.itos[token_id]
        except IndexError:
            token = vocab.UNK
        tokens.append(token)

    return ' '.join(tokens)


def dump_args_to_file(args: argparse.Namespace, filepath: Path) -> None:
    args_dict = vars(args)
    filepath.mkdir(parents=True, exist_ok=True)
    with open(filepath / 'args.json', 'w') as file:
        json.dump(args_dict, file, indent=2, cls=JSONPathEncoder)


def load_args_from_file(filepath: Path) -> argparse.Namespace:
    with open(filepath / 'args.json', 'r') as file:
        args = json.load(file)

    return argparse.Namespace(**args)
