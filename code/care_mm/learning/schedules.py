import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def constant_schedule(optimizer: Optimizer) -> LambdaLR:
    return LambdaLR(optimizer, lambda _: 1.0)


def linear_warmup_cosine(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    minimum_ratio: float = 0.0,
) -> LambdaLR:
    if warmup_steps < 0 or total_steps <= warmup_steps:
        raise ValueError("invalid warmup or total steps")
    if not 0 <= minimum_ratio <= 1:
        raise ValueError("minimum learning-rate ratio must lie in [0, 1]")

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_ratio + (1 - minimum_ratio) * cosine

    return LambdaLR(optimizer, multiplier)


def polynomial_decay(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    power: float = 1.0,
) -> LambdaLR:
    if power <= 0:
        raise ValueError("polynomial decay power must be positive")

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(1 - progress, 0.0) ** power

    return LambdaLR(optimizer, multiplier)
