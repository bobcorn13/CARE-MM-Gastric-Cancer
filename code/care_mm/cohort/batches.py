from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import Sampler


@dataclass(frozen=True)
class BatchComposition:
    indices: tuple[int, ...]
    positive_count: int
    settings: tuple[str, ...]


class PatientBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        labels: Sequence[int],
        settings: Sequence[str],
        batch_size: int,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if len(labels) != len(settings):
            raise ValueError("labels and settings must have equal length")
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        self.labels = tuple(labels)
        self.settings = tuple(settings)
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        pools: dict[tuple[int, str], list[int]] = {}
        for index, pair in enumerate(zip(self.labels, self.settings, strict=True)):
            pools.setdefault(pair, []).append(index)
        for values in pools.values():
            order = torch.randperm(len(values), generator=generator).tolist()
            values[:] = [values[item] for item in order]
        active = [key for key, values in pools.items() if values]
        batches: list[list[int]] = []
        current: list[int] = []
        cursor = 0
        while active:
            key = active[cursor % len(active)]
            current.append(pools[key].pop())
            if not pools[key]:
                active.remove(key)
                cursor = 0
            else:
                cursor += 1
            if len(current) == self.batch_size:
                batches.append(current)
                current = []
        if current and not self.drop_last:
            batches.append(current)
        order = torch.randperm(len(batches), generator=generator).tolist()
        for index in order:
            yield batches[index]

    def __len__(self) -> int:
        quotient, remainder = divmod(len(self.labels), self.batch_size)
        return quotient if self.drop_last or remainder == 0 else quotient + 1
