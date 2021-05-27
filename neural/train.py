from __future__ import annotations

import gc
import os
import time
from typing import List, Tuple, Callable, Optional, Any

import torch
import torch.nn as nn
from dotmap import DotMap
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader

from utils.general import convert_bytes_to_megabytes


class Trainer:
    def __init__(self, train_step: Callable[[Trainer, Tuple[Any, ...]], Tensor], train_loader: DataLoader,
                 epochs: int, batch_size: int, save_path: str, use_cuda: bool, **params: Any):
        self.model: Optional[DotMap[str, nn.Module]] = None
        self.criterion: Optional[DotMap[str, nn.Module]] = None
        self.optimizer: Optional[DotMap[str, Optimizer]] = None
        self.train_step = train_step
        self.train_loader = train_loader
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_path = save_path
        self.device = self.__get_device(use_cuda)
        self.params = DotMap(params)

        self.current_epoch = 0
        self.current_iteration = 0

    def set_models(self, **model: nn.Module) -> None:
        self.model = DotMap(model)

    def set_criterion(self, **criterion: nn.Module) -> None:
        self.criterion = DotMap(criterion)

    def set_optimizer(self, **optimizer: Optimizer) -> None:
        self.optimizer = DotMap(optimizer)

    def train(self, load_checkpoint: bool, verbosity: int, save_interval: int) -> None:
        self.__check_initialization()
        if load_checkpoint:
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
            running_loss = self.__train_epoch(running_loss, memory_usage, epoch, iteration, save_interval, verbosity)

    def __train_epoch(self, running_loss: List[float], memory_usage: List[float], epoch: int, from_iteration: int,
                      save_interval: int, verbosity: int) -> List[float]:
        time_start = time.time()
        gc.collect()
        for i, inputs in enumerate(self.train_loader):
            if i < from_iteration:
                continue
            else:
                from_iteration = 0

            for optimizer in self.optimizer.values():
                optimizer.zero_grad(set_to_none=True)

            torch.cuda.empty_cache()
            inputs = self.__convert_input_to_device(inputs)
            loss = self.train_step(self, inputs)
            loss.backward()
            for optimizer in self.optimizer.values():
                optimizer.step()

            running_loss.append(loss.item())
            if self.device.type == 'cuda':
                memory_usage.append(convert_bytes_to_megabytes(torch.cuda.memory_reserved(0)))
            self.current_iteration += 1

            if self.current_iteration % save_interval == 0:
                self.__save_checkpoint(epoch, i + 1)

            if self.current_iteration % verbosity == 0:
                self.__log_progress(running_loss, memory_usage, time_start, epoch, i)
                time_start = time.time()
                running_loss.clear()
                memory_usage.clear()

        return running_loss

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
        if os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path)
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
            print(f'Checkpoint {self.save_path} doesn\'t exist. Starting training from 0.')
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

        torch.save(checkpoint, self.save_path)

    def __convert_input_to_device(self, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
        inputs_in_device = []
        for tensor_in in inputs:
            if isinstance(tensor_in, Tensor):
                tensor_in = tensor_in.to(device=self.device)
            inputs_in_device.append(tensor_in)

        return tuple(inputs_in_device)

    def __log_progress(self, running_loss: List[float], memory_usage: List[float], time_start: float, epoch: int,
                       iteration: int) -> None:
        average_loss = sum(running_loss) / len(running_loss)
        time_iter = round(time.time() - time_start, 2)
        status_message = f'Epoch: {epoch} Iter: {iteration}/{len(self.train_loader)} Loss: {average_loss}, ' \
                         f'Time: {time_iter} seconds'
        if len(memory_usage) > 0:
            memory = sum(memory_usage) / len(memory_usage)
            status_message = f'{status_message}, Average memory use: {memory} MB'

        print(status_message)
