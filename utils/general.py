import re
from typing import List, Callable

from nltk.tokenize import sent_tokenize, word_tokenize


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


def preprocess_token(token: str, lowercase: bool, digits_to_zero: bool):
    token = token.lower() if lowercase else token
    token = re.sub(r'\d', '0', token) if digits_to_zero else token
    return token
