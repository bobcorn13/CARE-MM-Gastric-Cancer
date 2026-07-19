import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn


class Batch(Protocol):
    def to(self, device: torch.device) -> "Batch": ...


LossFunction = Callable[[nn.Module, Batch], torch.Tensor]


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    steps: int
    examples: int
    mean_loss: float
    learning_rate: float
    seconds: float


@dataclass(frozen=True)
class EngineState:
    epoch: int
    global_step: int
    best_metric: float


class TrainingEngine:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_function: LossFunction,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        scaler: torch.cuda.amp.GradScaler | None = None,
        accumulation_steps: int = 1,
        gradient_clip: float | None = None,
    ) -> None:
        if accumulation_steps < 1:
            raise ValueError("accumulation steps must be positive")
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.scheduler = scheduler
        self.scaler = scaler
        self.accumulation_steps = accumulation_steps
        self.gradient_clip = gradient_clip
        self.state = EngineState(0, 0, float("-inf"))
        self.logger = logging.getLogger("care_mm.learning")

    def train_epoch(self, batches: Iterable[Batch], epoch: int) -> EpochResult:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        started = time.monotonic()
        total_loss = 0.0
        examples = 0
        steps = 0
        for step, batch in enumerate(batches, start=1):
            batch = batch.to(self.device)
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.scaler is not None,
            ):
                loss = self.loss_function(self.model, batch)
                scaled_loss = loss / self.accumulation_steps
            if self.scaler is None:
                scaled_loss.backward()
            else:
                self.scaler.scale(scaled_loss).backward()
            if step % self.accumulation_steps == 0:
                self._optimizer_step()
            total_loss += float(loss.detach())
            batch_size = self._batch_size(batch)
            examples += batch_size
            steps += 1
            self.state = EngineState(epoch, self.state.global_step + 1, self.state.best_metric)
        if steps and steps % self.accumulation_steps:
            self._optimizer_step()
        seconds = time.monotonic() - started
        rate = float(self.optimizer.param_groups[0]["lr"])
        result = EpochResult(epoch, steps, examples, total_loss / max(steps, 1), rate, seconds)
        self.logger.info(
            "epoch=%d steps=%d examples=%d loss=%.6f lr=%.8g seconds=%.2f",
            result.epoch,
            result.steps,
            result.examples,
            result.mean_loss,
            result.learning_rate,
            result.seconds,
        )
        return result

    def _optimizer_step(self) -> None:
        if self.scaler is None:
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
        else:
            if self.gradient_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()

    @staticmethod
    def _batch_size(batch: Batch) -> int:
        for value in vars(batch).values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return value.shape[0]
        return 0

    @torch.no_grad()
    def evaluate(self, batches: Iterable[Batch]) -> float:
        self.model.eval()
        total = 0.0
        count = 0
        for batch in batches:
            batch = batch.to(self.device)
            total += float(self.loss_function(self.model, batch))
            count += 1
        return total / max(count, 1)

    def update_best(self, metric: float) -> bool:
        if metric <= self.state.best_metric:
            return False
        self.state = EngineState(self.state.epoch, self.state.global_step, metric)
        return True
