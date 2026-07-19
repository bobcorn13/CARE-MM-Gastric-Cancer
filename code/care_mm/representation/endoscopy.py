import math

import torch
from torch import nn

from care_mm.representation.common import FeatureProjector, TransformerStack, masked_mean


class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, width: int, maximum_length: int = 8192) -> None:
        super().__init__()
        positions = torch.arange(maximum_length, dtype=torch.float32).unsqueeze(1)
        divisors = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
        values = torch.zeros(maximum_length, width)
        values[:, 0::2] = torch.sin(positions * divisors)
        values[:, 1::2] = torch.cos(positions * divisors[: values[:, 1::2].shape[1]])
        self.register_buffer("values", values, persistent=False)

    def forward(self, length: int) -> torch.Tensor:
        if length > self.values.shape[0]:
            raise ValueError("sequence exceeds configured temporal encoding")
        return self.values[:length]


class TemporalAttentivePool(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(width))
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Sequential(nn.Linear(width, width), nn.LayerNorm(width))
        nn.init.normal_(self.query, std=width**-0.5)

    def forward(
        self, frames: torch.Tensor, padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = frames.shape[-1] ** -0.5
        logits = torch.einsum("btf,f->bt", self.key(frames), self.query) * scale
        logits = logits.masked_fill(padding_mask, -torch.inf)
        weights = torch.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        pooled = torch.einsum("bt,btf->bf", weights, self.value(frames))
        return self.output(pooled), weights


class EndoscopySequenceEncoder(nn.Module):
    def __init__(
        self,
        input_width: int,
        width: int = 512,
        depth: int = 3,
        heads: int = 8,
        dropout: float = 0.1,
        maximum_frames: int = 8192,
    ) -> None:
        super().__init__()
        self.project = FeatureProjector(input_width, width, dropout)
        self.time = SinusoidalTimeEncoding(width, maximum_frames)
        self.sequence = TransformerStack(width, depth, heads, 4, dropout)
        self.pool = TemporalAttentivePool(width, dropout)
        self.summary = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(width),
        )

    def forward(
        self, frames: torch.Tensor, padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if frames.ndim != 3:
            raise ValueError("endoscopy input must have batch, frame and feature dimensions")
        if padding_mask.shape != frames.shape[:2]:
            raise ValueError("endoscopy padding mask shape mismatch")
        projected = self.project(frames)
        projected = projected + self.time(frames.shape[1]).unsqueeze(0)
        contextual = self.sequence(projected, padding_mask)
        attended, weights = self.pool(contextual, padding_mask)
        average = masked_mean(contextual, padding_mask)
        return self.summary(torch.cat((attended, average), dim=-1)), weights
