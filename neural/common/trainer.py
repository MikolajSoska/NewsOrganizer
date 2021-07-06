from __future__ import annotations

import gc
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Any

import torch
import torch.nn as nn
import tqdm
from dotmap import DotMap
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader

from neural.common.scores import Scorer, ScoreValue
from utils.general import convert_bytes_to_megabytes


class Trainer:
    def __init__(self, train_step: Callable[[Trainer, Tuple[Any, ...]], Tuple[Tensor, ScoreValue]], epochs: int,
                 batch_size: int, max_gradient_norm: Optional[int], save_path: str, model_name: str, use_cuda: bool,
                 load_checkpoint: bool, max_model_backup: int = 3, scores: List[Scorer] = None,
                 validation_scores: List[Scorer] = None, test_scores: List[Scorer] = None, **params: Any):
        self.model: Optional[DotMap[str, nn.Module]] = None
        self.criterion: Optional[DotMap[str, nn.Module]] = None
        self.optimizer: Optional[DotMap[str, Optimizer]] = None
        self.train_step = train_step
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_gradient_norm = max_gradient_norm
        self.save_path = Path(save_path)
        self.model_name = model_name
        self.device = self.__get_device(use_cuda)
        self.load_checkpoint = load_checkpoint
        self.max_model_backup = max_model_backup
        self.train_scores = scores
        self.validation_scores = validation_scores or scores
        self.test_scores = test_scores or validation_scores or scores
        self.params = DotMap(params)

        self.current_epoch = 0
        self.current_iteration = 0
        self.current_phase = 'train'

    def set_models(self, **model: nn.Module) -> None:
        self.model = DotMap(model)

    def set_criterion(self, **criterion: nn.Module) -> None:
        self.criterion = DotMap(criterion)

    def set_optimizer(self, **optimizer: Optimizer) -> None:
        self.optimizer = DotMap(optimizer)

    def train(self, train_loader: DataLoader, validation_loader: DataLoader = None, verbosity: int = 50,
              save_interval: int = 50) -> None:
        self.__check_initialization()
        self.save_path.mkdir(parents=True, exist_ok=True)
        oom_occurred = False

        try:
            self.__train(train_loader, validation_loader, verbosity, save_interval)
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
            self.train(train_loader, validation_loader, verbosity, save_interval)

    def score(self, predictions: Tensor, targets: Tensor) -> ScoreValue:
        if self.current_phase == 'train':
            scorers = self.train_scores
        elif self.current_phase == 'validation':
            scorers = self.validation_scores
        elif self.current_phase == 'test':
            scorers = self.test_scores
        else:
            raise ValueError(f'Invalid current phase: {self.current_phase}')

        score = ScoreValue()
        if scorers is not None:
            for scorer in scorers:
                score += scorer.score(predictions, targets)

        return score

    def __train(self, train_loader: DataLoader, validation_loader: Optional[DataLoader], verbosity: int,
                save_interval: int) -> None:
        if self.load_checkpoint:
            iteration, epoch_start = self.__load_checkpoint(len(train_loader))
        else:
            iteration = 0
            epoch_start = 0

        for name, model in self.model.items():
            self.model.name = model.to(self.device)
            model.train()

        for optimizer in self.optimizer.values():  # Reload optimizer to get correct model device
            optimizer.load_state_dict(optimizer.state_dict())

        self.current_iteration = (epoch_start + 1) * iteration
        score = ScoreValue()
        running_loss = []
        memory_usage = []

        for epoch in range(epoch_start, self.epochs):
            self.current_phase = 'train'
            self.current_epoch = epoch
            score = self.__train_epoch(train_loader, running_loss, memory_usage, score, epoch, iteration, verbosity,
                                       save_interval)
            print(f'Finished training epoch {epoch}.')
            if validation_loader is not None:
                self.current_phase = 'validation'
                self.__validate_epoch(validation_loader)
            iteration = 0

    def __train_epoch(self, train_loader: DataLoader, running_loss: List[float], memory_usage: List[float],
                      running_score: ScoreValue, epoch: int, from_iteration: int, verbosity: int,
                      save_interval: int) -> ScoreValue:
        time_start = time.time()
        train_length = len(train_loader)
        gc.collect()
        for i, inputs in enumerate(train_loader):
            if i < from_iteration:
                continue

            for optimizer in self.optimizer.values():
                optimizer.zero_grad(set_to_none=True)

            inputs = self.__convert_input_to_device(inputs)
            loss, score = self.train_step(self, inputs)
            running_score += score
            loss.backward()
            self.__clip_gradients()

            for optimizer in self.optimizer.values():
                optimizer.step()

            running_loss.append(loss.item())
            del loss
            if self.device.type == 'cuda':
                memory_usage.append(convert_bytes_to_megabytes(torch.cuda.memory_reserved(0)))
            self.current_iteration += self.batch_size

            if (self.current_iteration // self.batch_size) % save_interval == 0:
                self.__save_checkpoint(epoch, i + 1)

            if (self.current_iteration // self.batch_size) % verbosity == 0:
                self.__log_progress(running_loss, memory_usage, running_score, time_start, epoch, i, train_length)
                time_start = time.time()
                running_loss.clear()
                memory_usage.clear()
                running_score = ScoreValue()

        return running_score

    def __validate_epoch(self, validation_loader: DataLoader) -> None:
        print('Starting validation phase...')
        time_start = time.time()
        validation_length = len(validation_loader)
        running_loss = []
        running_score = ScoreValue()
        gc.collect()

        with torch.no_grad():  # TODO add model.eval() (aktualnie wywala jakiś błąd pamięci CUDA przy użyciu eval
            for inputs in tqdm.tqdm(validation_loader, file=sys.stdout):
                inputs = self.__convert_input_to_device(inputs)
                loss, score = self.train_step(self, inputs)

                running_loss.append(loss.item())
                running_score += score

                del loss

        print('Validation result: ', end='')
        self.__log_progress(running_loss, [], running_score, time_start, self.current_epoch, validation_length,
                            validation_length)

    def __clip_gradients(self) -> None:
        if self.max_gradient_norm is not None:
            for model in self.model.values():
                nn.utils.clip_grad_norm_(model.parameters(), self.max_gradient_norm)

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

    def __load_checkpoint(self, iteration_number: int) -> Tuple[int, int]:
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
            print(f'Epoch set to {epoch_start}. Iteration set to {iteration} of {iteration_number}.')
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
            try:
                (self.save_path / Path(checkpoint)).unlink()
            except PermissionError as error:
                print(f'Can\'t remove old checkpoint {checkpoint}. Permission error: {error}.')

    def __convert_input_to_device(self, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
        inputs_in_device = []
        for tensor_in in inputs:
            if isinstance(tensor_in, Tensor):
                tensor_in = tensor_in.to(device=self.device)
            inputs_in_device.append(tensor_in)

        return tuple(inputs_in_device)

    @staticmethod
    def __log_progress(running_loss: List[float], memory_usage: List[float], running_score: ScoreValue,
                       time_start: float, epoch: int, iteration: int, iteration_max: int) -> None:
        torch.cuda.empty_cache()
        average_loss = sum(running_loss) / len(running_loss)
        score = running_score / len(running_loss)
        time_iter = round(time.time() - time_start, 2)
        status_message = f'Epoch: {epoch} Iter: {iteration}/{iteration_max}, Time: {time_iter} seconds, ' \
                         f'Loss: {average_loss}'

        if len(score) > 0:
            status_message = f'{status_message}, {score}'

        if len(memory_usage) > 0:
            memory = round(sum(memory_usage) / len(memory_usage), 2)
            status_message = f'{status_message}, Average memory use: {memory} MB'

        print(status_message)
