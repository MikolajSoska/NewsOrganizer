from typing import Union

import torch
import torch.nn as nn
from torch import Tensor


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
