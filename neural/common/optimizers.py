from typing import Tuple, Iterable, Optional, Callable

from torch import optim


class TransformerAdam(optim.Adam):
    def __init__(self, params: Iterable, betas: Tuple[float, float], eps: float, model_dim: int, warmup_steps: int):
        self.steps = 1
        self.model_dim = model_dim ** -0.5
        self.warmup_steps = warmup_steps
        super().__init__(params, lr=self.__update_learning_rate(), betas=betas, eps=eps)  # Get initial learning rate

    def __update_learning_rate(self) -> float:
        return self.model_dim * min(self.steps ** -0.5, self.steps * self.warmup_steps ** -1.5)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        result = super().step(closure)
        self.steps += 1
        learning_rate = self.__update_learning_rate()
        for param_group in self.param_groups:
            print(param_group['lr'])
            param_group['lr'] = learning_rate

        return result
