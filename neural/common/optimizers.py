from typing import Tuple, Iterable, Optional, Callable

from torch import optim


class TransformerAdam(optim.Adam):
    def __init__(self, params: Iterable, betas: Tuple[float, float], eps: float, model_dim: int, warmup_steps: int,
                 batch_size: int):
        self.steps = 1
        self.batch_size = batch_size
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
            param_group['lr'] = learning_rate

        return result

    def state_dict(self) -> dict:
        state_dict = super().state_dict()
        state_dict['steps'] = self.steps
        state_dict['batch_size'] = self.batch_size
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        # Reconstruct steps number from previous and current batch sizes
        self.steps = state_dict['steps'] * state_dict['batch_size'] // self.batch_size
