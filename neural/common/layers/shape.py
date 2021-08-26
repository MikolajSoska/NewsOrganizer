import torch
import torch.nn as nn
from torch import Tensor


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
