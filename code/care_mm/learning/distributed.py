import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def primary(self) -> bool:
        return self.rank == 0


def initialize_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return DistributedContext(rank, local_rank, world_size, device)


def wrap_distributed(model: nn.Module, context: DistributedContext) -> nn.Module:
    model = model.to(context.device)
    if context.world_size == 1:
        return model
    device_ids = [context.local_rank] if context.device.type == "cuda" else None
    return DistributedDataParallel(model, device_ids=device_ids, broadcast_buffers=False)


def reduce_mean(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    if context.world_size == 1:
        return value
    reduced = value.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced / context.world_size


def gather_variable(values: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    if context.world_size == 1:
        return values
    local_length = torch.tensor([values.shape[0]], device=values.device)
    lengths = [torch.zeros_like(local_length) for _ in range(context.world_size)]
    dist.all_gather(lengths, local_length)
    maximum = max(int(item.item()) for item in lengths)
    padding_shape = (maximum - values.shape[0],) + values.shape[1:]
    padded = torch.cat((values, values.new_zeros(padding_shape)), dim=0)
    gathered = [torch.zeros_like(padded) for _ in range(context.world_size)]
    dist.all_gather(gathered, padded)
    return torch.cat(
        [item[: int(length.item())] for item, length in zip(gathered, lengths, strict=True)]
    )


def finalize_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
