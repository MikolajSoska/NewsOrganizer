import random

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def convert_bytes_to_megabytes(bytes_number: int) -> float:
    return round(bytes_number / (1024 ** 2), 2)
