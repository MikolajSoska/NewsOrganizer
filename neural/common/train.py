from __future__ import annotations

import argparse
import gc
import logging
import os
import re
import sys
import time
from functools import cmp_to_key
from pathlib import Path
from typing import List, Tuple, Dict, Callable, Optional, Any

import torch
import torch.nn as nn
import tqdm
from dotmap import DotMap
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader

from neural.common.scores import Scorer, ScoreValue
from neural.common.utils import convert_bytes_to_megabytes, get_device, convert_input_to_device


class Trainer:
    def __init__(self, train_step: Callable[[Trainer, Tuple[Any, ...]], Tuple[Tensor, ScoreValue]], epochs: int,
                 max_gradient_norm: Optional[int], model_save_path: Path, log_save_path: Path, model_name: str,
                 use_cuda: bool, cuda_index: int, max_model_backup: int = 3, scores: List[Scorer] = None,
                 validation_scores: List[Scorer] = None, test_scores: List[Scorer] = None, **params: Any):
        self.model: Optional[DotMap[str, nn.Module]] = None
        self.criterion: Optional[DotMap[str, nn.Module]] = None
        self.optimizer: Optional[DotMap[str, Optimizer]] = None
        self.train_step = train_step
        self.epochs = epochs
        self.max_gradient_norm = max_gradient_norm
        self.model_name = model_name
        self.model_save_path = self.__get_save_dir(model_save_path)
        self.log_save_path = self.__get_save_dir(log_save_path)
        self.logger = self.__setup_logger()
        self.device = get_device(use_cuda, cuda_index, log_method=self.logger.info)
        self.cuda_index = cuda_index
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
        for name, model in self.model.items():
            self.model.name = model.to(self.device)

    def set_criterion(self, **criterion: nn.Module) -> None:
        self.criterion = DotMap(criterion)

    def set_optimizer(self, **optimizer: Optimizer) -> None:
        self.optimizer = DotMap(optimizer)

    def train(self, train_loader: DataLoader, validation_loader: DataLoader = None, verbosity: int = 50,
              save_interval: int = 50) -> None:
        self.__check_initialization()
        oom_occurred = False

        try:
            self.__train(train_loader, validation_loader, verbosity, save_interval)
        except RuntimeError as error:
            if 'CUDA out of memory' in str(error):
                self.logger.error(f'Caught OOM exception: {error}. Restarting training from most recent checkpoint.')
                oom_occurred = True
            else:
                raise error

        if oom_occurred:
            torch.cuda.empty_cache()
            gc.collect()
            self.train(train_loader, validation_loader, verbosity, save_interval)

    def eval(self, test_loader: DataLoader, val_loader: DataLoader = None, full_validation: bool = False) -> None:
        self.__check_initialization()
        self.__setup_log_file_handler('full_eval' if full_validation else 'eval')

        if full_validation:
            assert val_loader is not None, 'During full evaluation phase validation loader has to be provided.'
            self.current_phase = 'validation'
            self.logger.info('Starting validation phase for each trained epoch...')
            self.__full_validation(val_loader)

        self.logger.info('Starting test phase...')
        self.__load_trained_model(f'{self.model_name}.pt')
        self.current_phase = 'test'
        self.__validate_model(test_loader)

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
        iteration, epoch_start = self.__load_checkpoint(len(train_loader), train_loader.batch_size)

        for model in self.model.values():
            model.train()

        for optimizer in self.optimizer.values():  # Reload optimizer to get correct model device
            optimizer.load_state_dict(optimizer.state_dict())

        self.current_iteration = (epoch_start * len(train_loader) + iteration) * train_loader.batch_size
        score = ScoreValue()
        running_loss = []
        memory_usage = []

        for epoch in range(epoch_start, self.epochs):
            self.__setup_log_file_handler(f'epoch-{epoch}')
            self.current_phase = 'train'
            self.current_epoch = epoch
            score = self.__train_epoch(train_loader, running_loss, memory_usage, score, epoch, iteration, verbosity,
                                       save_interval)
            self.__save_model_checkpoint(Path(f'{self.model_name}-epoch-{epoch}.pt'))
            self.logger.info(f'Finished training epoch {epoch}.')
            if validation_loader is not None:
                self.logger.info('Starting validation phase...')
                self.current_phase = 'validation'
                self.__validate_model(validation_loader)
            iteration = 0

        self.__save_model_checkpoint()

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

            inputs = convert_input_to_device(inputs, self.device)
            loss, score = self.train_step(self, inputs)
            running_score += score
            loss.backward()
            self.__clip_gradients()

            for optimizer in self.optimizer.values():
                optimizer.step()

            running_loss.append(loss.item())
            del loss
            if self.device.type == 'cuda':
                memory_usage.append(convert_bytes_to_megabytes(torch.cuda.memory_reserved(self.cuda_index)))
            self.current_iteration += train_loader.batch_size

            if (self.current_iteration // train_loader.batch_size) % save_interval == 0:
                self.__save_full_checkpoint(epoch, i + 1, train_loader.batch_size)

            if (self.current_iteration // train_loader.batch_size) % verbosity == 0:
                self.__log_progress(running_loss, memory_usage, running_score, time_start, epoch, i, train_length)
                time_start = time.time()
                running_loss.clear()
                memory_usage.clear()
                running_score = ScoreValue()

        return running_score

    def __validate_model(self, data_loader: DataLoader) -> None:
        time_start = time.time()
        data_length = len(data_loader)
        running_loss = []
        running_score = ScoreValue()
        gc.collect()

        with torch.no_grad():
            for model in self.model.values():
                model.eval()
            for inputs in tqdm.tqdm(data_loader, file=sys.stdout):
                inputs = convert_input_to_device(inputs, self.device)
                loss, score = self.train_step(self, inputs)

                running_loss.append(loss.item())
                running_score += score

                del loss

        for model in self.model.values():
            model.train()

        self.logger.info('Result:')
        self.__log_progress(running_loss, [], running_score, time_start, self.current_epoch, data_length, data_length)

    def __full_validation(self, validation_loader: DataLoader) -> None:
        checkpoints = self.__get_epoch_checkpoints()
        for i, checkpoint_name in enumerate(checkpoints):
            self.__load_trained_model(checkpoint_name)
            self.current_epoch = i
            self.__validate_model(validation_loader)

    def __load_trained_model(self, checkpoint_name: str) -> None:
        checkpoint = torch.load(self.model_save_path / Path(checkpoint_name), map_location=self.device)
        for name, model in self.model.items():
            model.load_state_dict(checkpoint[f'{name}_state_dict'])
        for name, optimizer in self.optimizer.items():
            optimizer.load_state_dict(checkpoint[f'{name}-optimizer_state_dict'])
        del checkpoint

    def __clip_gradients(self) -> None:
        if self.max_gradient_norm is not None:
            for model in self.model.values():
                nn.utils.clip_grad_norm_(model.parameters(), self.max_gradient_norm)

    def __get_save_dir(self, save_path: Path) -> Path:
        path = save_path / self.model_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def __setup_logger() -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)

        return logger

    def __setup_log_file_handler(self, file_label: str) -> None:
        if len(self.logger.handlers) == 2:  # If file handler is already set up
            self.logger.handlers.pop()

        logfile = self.log_save_path / f'{self.model_name}-log-{file_label}.log'
        file_handler = logging.FileHandler(logfile, mode='a')
        file_handler.setLevel(logging.INFO)

        self.logger.addHandler(file_handler)

    def __check_initialization(self) -> None:
        if self.model is None:
            raise AttributeError('Models are not initialized.')
        if self.criterion is None:
            raise AttributeError('Criteria are not initialized.')
        if self.optimizer is None and self.current_phase == 'train':
            raise AttributeError('Optimizers are not initialized.')

    def __load_checkpoint(self, iteration_number: int, batch_size: int) -> Tuple[int, int]:
        self.logger.info('Loading checkpoint...')

        checkpoints = self.__get_checkpoint_list()
        if len(checkpoints) > 0:
            checkpoint = torch.load(self.model_save_path / Path(checkpoints[0]), map_location=self.device)
            for name, model in self.model.items():
                model.load_state_dict(checkpoint[f'{name}_state_dict'])
            for name, optimizer in self.optimizer.items():
                current_rl = optimizer.param_groups[0]['lr']
                optimizer.load_state_dict(checkpoint[f'{name}-optimizer_state_dict'])
                optimizer.param_groups[0]['lr'] = current_rl  # Set learning rate from the current run

            epoch_start = checkpoint['epoch']
            previous_batch_size = checkpoint['batch_size']
            iteration = checkpoint['iteration'] * previous_batch_size // batch_size
            self.logger.info(f'Epoch set to {epoch_start}. Iteration set to {iteration} of {iteration_number}.')
            del checkpoint
        else:
            self.logger.info(f'Checkpoint for model {self.model_name} in {self.model_save_path} doesn\'t exist. '
                             f'Training from scratch.')
            iteration = 0
            epoch_start = 0

        return iteration, epoch_start

    def __save_full_checkpoint(self, epoch: int, iteration: int, batch_size: int) -> None:
        checkpoint = {
            'epoch': epoch,
            'iteration': iteration,
            'batch_size': batch_size,
        }

        checkpoint_name = Path(f'{self.model_name}-e{epoch}i{iteration * batch_size}.pt')
        self.__save_model_checkpoint(checkpoint_name, checkpoint)
        self.__remove_old_checkpoints()

    def __save_model_checkpoint(self, checkpoint_name: Path = None, checkpoint: Dict[str, Any] = None):
        if checkpoint_name is None:
            checkpoint_name = Path(f'{self.model_name}.pt')

        if checkpoint is None:
            checkpoint = {}

        for name, model in self.model.items():
            checkpoint[f'{name}_state_dict'] = model.state_dict()

        for name, optimizer in self.optimizer.items():
            checkpoint[f'{name}-optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, self.model_save_path / checkpoint_name)

    def __get_checkpoint_list(self) -> List[str]:
        def compare_checkpoints(checkpoint_1: str, checkpoint_2: str) -> int:
            epochs_1, iterations_1 = re.findall(r'\d+', checkpoint_1)[-2:]
            epochs_2, iterations_2 = re.findall(r'\d+', checkpoint_2)[-2:]

            if epochs_1 == epochs_2:
                return int(iterations_1) - int(iterations_2)
            else:
                return int(epochs_1) - int(epochs_2)

        name_pattern = re.compile(rf'^{self.model_name}-e\d+i\d+\.pt$')
        return list(sorted(filter(name_pattern.match, os.listdir(self.model_save_path)), reverse=True,
                           key=cmp_to_key(compare_checkpoints)))

    def __get_epoch_checkpoints(self) -> List[str]:
        def compare_checkpoints(checkpoint_1: str, checkpoint_2: str) -> int:
            epoch_1 = re.findall(r'\d+', checkpoint_1)[0]
            epoch_2 = re.findall(r'\d+', checkpoint_2)[0]

            return int(epoch_1) - int(epoch_2)

        name_pattern = re.compile(rf'^{self.model_name}-epoch-\d+\.pt$')
        return list(sorted(filter(name_pattern.match, os.listdir(self.model_save_path)),
                           key=cmp_to_key(compare_checkpoints)))

    def __remove_old_checkpoints(self) -> None:
        checkpoints = self.__get_checkpoint_list()[self.max_model_backup:]
        for checkpoint in checkpoints:
            try:
                (self.model_save_path / Path(checkpoint)).unlink()
            except PermissionError as error:
                self.logger.error(f'Can\'t remove old checkpoint {checkpoint}. Permission error: {error}.')

    def __log_progress(self, running_loss: List[float], memory_usage: List[float], running_score: ScoreValue,
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

        self.logger.info(status_message)


def add_base_train_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--use-gpu', action='store_true', help='Train with CUDA')
    parser.add_argument('--gpu-index', type=int, default=0, help='CUDA GPU index (only if multiple GPUs are available)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--eval-only', action='store_true', help='Run only validation phase')
    parser.add_argument('--full-validation', action='store_true', help='Runs evaluation for each epoch checkpoint')
    parser.add_argument('--vocab-path', type=Path, default='../data/saved/vocabs', help='Path to vocab files')
    parser.add_argument('--data-path', type=Path, default='../data/saved/datasets', help='Path to dataset files')
    parser.add_argument('--model-path', type=Path, default='../data/saved/models', help='Path to model files')
    parser.add_argument('--logs-path', type=Path, default='../data/saved/logs', help='Path to logs files')

    return parser
