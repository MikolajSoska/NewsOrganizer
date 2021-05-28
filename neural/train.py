from __future__ import annotations

import gc
import os
import re
import time
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Any

import torch
import torch.nn as nn
from dotmap import DotMap
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader

from utils.general import convert_bytes_to_megabytes


class Trainer:
    def __init__(self, train_step: Callable[[Trainer, Tuple[Any, ...]], Tensor], train_loader: DataLoader, epochs: int,
                 batch_size: int, save_path: str, model_name: str, use_cuda: bool, load_checkpoint: bool,
                 verbosity: int = 50, save_interval: int = 50, max_model_backup: int = 3, **params: Any):
        self.model: Optional[DotMap[str, nn.Module]] = None
        self.criterion: Optional[DotMap[str, nn.Module]] = None
        self.optimizer: Optional[DotMap[str, Optimizer]] = None
        self.train_step = train_step
        self.train_loader = train_loader
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_path = Path(save_path)
        self.model_name = model_name
        self.device = self.__get_device(use_cuda)
        self.load_checkpoint = load_checkpoint
        self.verbosity = verbosity
        self.save_interval = save_interval
        self.max_model_backup = max_model_backup
        self.params = DotMap(params)

        self.current_epoch = 0
        self.current_iteration = 0

    def set_models(self, **model: nn.Module) -> None:
        self.model = DotMap(model)

    def set_criterion(self, **criterion: nn.Module) -> None:
        self.criterion = DotMap(criterion)

    def set_optimizer(self, **optimizer: Optimizer) -> None:
        self.optimizer = DotMap(optimizer)

    def train(self) -> None:
        self.__check_initialization()
        self.save_path.mkdir(parents=True, exist_ok=True)
        oom_occurred = False

        try:
            self.__train()
        except RuntimeError as error:
            if 'CUDA out of memory' in str(error):
                print(f'Caught OOM exception: {error}. Restarting training from most recent checkpoint.')
                oom_occurred = True
            else:
                raise error

        if oom_occurred:
            torch.cuda.empty_cache()
            gc.collect()
            self.load_checkpoint = True  # Restart from checkpoint
            self.train()

    def __train(self) -> None:
        if self.load_checkpoint:
            iteration, epoch_start = self.__load_checkpoint()
        else:
            iteration = 0
            epoch_start = 0

        for name, model in self.model.items():
            self.model.name = model.to(self.device)
            self.model.name.train()

        for optimizer in self.optimizer.values():  # Reload optimizer to get correct model device
            optimizer.load_state_dict(optimizer.state_dict())

        self.current_iteration = (epoch_start + 1) * iteration
        running_loss = []
        memory_usage = []

        for epoch in range(epoch_start, self.epochs):
            self.current_epoch = epoch
            self.__train_epoch(running_loss, memory_usage, epoch, iteration)
            iteration = 0

    def __train_epoch(self, running_loss: List[float], memory_usage: List[float], epoch: int,
                      from_iteration: int) -> None:
        time_start = time.time()
        gc.collect()
        for i, inputs in enumerate(self.train_loader):
            if i < from_iteration:
                continue

            for optimizer in self.optimizer.values():
                optimizer.zero_grad(set_to_none=True)

            inputs = self.__convert_input_to_device(inputs)
            loss = self.train_step(self, inputs)
            loss.backward()
            for optimizer in self.optimizer.values():
                optimizer.step()

            running_loss.append(loss.item())
            del loss
            if self.device.type == 'cuda':
                memory_usage.append(convert_bytes_to_megabytes(torch.cuda.memory_reserved(0)))
            self.current_iteration += 1

            if self.current_iteration % self.save_interval == 0:
                self.__save_checkpoint(epoch, i + 1)

            if self.current_iteration % self.verbosity == 0:
                self.__log_progress(running_loss, memory_usage, time_start, epoch, i)
                time_start = time.time()
                running_loss.clear()
                memory_usage.clear()

    @staticmethod
    def __get_device(use_cuda: bool) -> torch.device:
        if use_cuda:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                device_properties = torch.cuda.get_device_properties(device)
                device_name = f'CUDA ({device_properties.name}), ' \
                              f'Total memory: {convert_bytes_to_megabytes(device_properties.total_memory):g} MB'
            else:
                print('CUDA device is not available.')
                device = torch.device('cpu')
                device_name = 'CPU'
        else:
            device = torch.device('cpu')
            device_name = 'CPU'

        print(f'Using device: {device_name}')
        return device

    def __check_initialization(self) -> None:
        if self.model is None:
            raise AttributeError('Models are not initialized.')
        if self.criterion is None:
            raise AttributeError('Criteria are not initialized.')
        if self.optimizer is None:
            raise AttributeError('Optimizers are not initialized.')

    def __load_checkpoint(self) -> Tuple[int, int]:
        print('Loading checkpoint...')

        checkpoints = self.__get_checkpoint_list()
        if len(checkpoints) > 0:
            checkpoint = torch.load(self.save_path / Path(checkpoints[0]))
            for name, model in self.model.items():
                model.load_state_dict(checkpoint[f'{name}_state_dict'])
            for name, optimizer in self.optimizer.items():
                optimizer.load_state_dict(checkpoint[f'{name}-optimizer_state_dict'])
            epoch_start = checkpoint['epoch']
            previous_batch_size = checkpoint['batch_size']
            iteration = checkpoint['iteration'] * previous_batch_size // self.batch_size
            print(f'Epoch set to {epoch_start}. Iteration set to {iteration}.')
            del checkpoint
        else:
            print(f'Checkpoint for model {self.model_name} in {self.save_path} doesn\'t exist. Training from scratch.')
            iteration = 0
            epoch_start = 0

        return iteration, epoch_start

    def __save_checkpoint(self, epoch: int, iteration: int) -> None:
        checkpoint = {
            'epoch': epoch,
            'iteration': iteration,
            'batch_size': self.batch_size,
        }

        for name, model in self.model.items():
            checkpoint[f'{name}_state_dict'] = model.state_dict()

        for name, optimizer in self.optimizer.items():
            checkpoint[f'{name}-optimizer_state_dict'] = optimizer.state_dict()

        checkpoint_name = Path(f'{self.model_name}-e{epoch}i{iteration * self.batch_size}.pt')
        torch.save(checkpoint, self.save_path / checkpoint_name)
        self.__remove_old_checkpoints()

    def __get_checkpoint_list(self) -> List[str]:
        name_pattern = re.compile(rf'^{self.model_name}-e\d+i\d+\.pt$')
        return list(sorted(filter(name_pattern.match, os.listdir(self.save_path)), reverse=True))

    def __remove_old_checkpoints(self) -> None:
        checkpoints = self.__get_checkpoint_list()[self.max_model_backup:]
        for checkpoint in checkpoints:
            (self.save_path / Path(checkpoint)).unlink()

    def __convert_input_to_device(self, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
        inputs_in_device = []
        for tensor_in in inputs:
            if isinstance(tensor_in, Tensor):
                tensor_in = tensor_in.to(device=self.device)
            inputs_in_device.append(tensor_in)

        return tuple(inputs_in_device)

    def __log_progress(self, running_loss: List[float], memory_usage: List[float], time_start: float, epoch: int,
                       iteration: int) -> None:
        torch.cuda.empty_cache()
        average_loss = sum(running_loss) / len(running_loss)
        time_iter = round(time.time() - time_start, 2)
        status_message = f'Epoch: {epoch} Iter: {iteration}/{len(self.train_loader)} Loss: {average_loss}, ' \
                         f'Time: {time_iter} seconds'
        if len(memory_usage) > 0:
            memory = sum(memory_usage) / len(memory_usage)
            status_message = f'{status_message}, Average memory use: {memory} MB'

        print(status_message)
