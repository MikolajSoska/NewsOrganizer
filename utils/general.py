import random
from typing import List, Callable

import numpy as np
import torch
from nltk.tokenize import sent_tokenize, word_tokenize


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def convert_bytes_to_megabytes(bytes_number: int) -> float:
    return round(bytes_number / (1024 ** 2), 2)


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
