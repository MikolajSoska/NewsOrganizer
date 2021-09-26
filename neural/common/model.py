from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any

import torch.nn as nn
from torch import Tensor


class BaseModel(nn.Module, ABC):
    @classmethod
    @abstractmethod
    def create_from_args(cls, args: Dict[str, Any], **kwargs: Any) -> BaseModel:
        pass

    @abstractmethod
    def predict(self, *inputs: Any) -> Tensor:
        pass
