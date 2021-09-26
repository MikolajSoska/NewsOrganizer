from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

import torch.nn as nn
from torch import Tensor


class BaseModel(nn.Module, ABC):
    @classmethod
    @abstractmethod
    def create_from_args(cls, args: Dict[Any], **additional_parameters: Any) -> BaseModel:
        pass

    @abstractmethod
    def prediction_step(self, inputs: Tuple[Any, ...]) -> Tuple[Tensor, Any, ...]:
        pass
