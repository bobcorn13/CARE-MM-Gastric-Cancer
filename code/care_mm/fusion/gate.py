import math

import torch
from torch import nn


class MaskedSoftmaxGate(nn.Module):
    def __init__(self, width: int = 512, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError("fusion width must be divisible by attention heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Parameter(torch.empty(heads, self.head_width))
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.query, std=self.head_width**-0.5)

    def forward(
        self, tokens: torch.Tensor, available: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError("modality tokens must have batch, modality and feature dimensions")
        if tokens.shape[-1] != self.width:
            raise ValueError("modality token width mismatch")
        if available.shape != tokens.shape[:2] or available.dtype != torch.bool:
            raise ValueError("availability mask must be boolean and match modality axes")
        if torch.any(available.sum(dim=1) == 0):
            raise ValueError("every case must contain at least one modality")
        batch, modalities, _ = tokens.shape
        keys = self.key(tokens).reshape(batch, modalities, self.heads, self.head_width)
        values = self.value(tokens).reshape(batch, modalities, self.heads, self.head_width)
        logits = torch.einsum("bmhd,hd->bhm", keys, self.query) / math.sqrt(self.head_width)
        logits = logits.masked_fill(~available.unsqueeze(1), -torch.inf)
        weights = torch.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        fused = torch.einsum("bhm,bmhd->bhd", weights, values).reshape(batch, self.width)
        return self.output(fused), weights


class CrossModalBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.feed_norm = nn.LayerNorm(width)
        self.feed = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~available,
            need_weights=False,
        )
        tokens = tokens + attended
        tokens = tokens + self.feed(self.feed_norm(tokens))
        return tokens.masked_fill(~available.unsqueeze(-1), 0.0)


class FusionHead(nn.Module):
    def __init__(
        self,
        width: int = 512,
        depth: int = 4,
        heads: int = 8,
        classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.modality_identity = nn.Parameter(torch.empty(1, 3, width))
        self.blocks = nn.ModuleList([CrossModalBlock(width, heads, dropout) for _ in range(depth)])
        self.gate = MaskedSoftmaxGate(width, heads, dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, classes),
        )
        nn.init.normal_(self.modality_identity, std=0.02)

    def forward(
        self, tokens: torch.Tensor, available: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tokens.shape[1] != 3:
            raise ValueError("CARE-MM fusion expects three modality positions")
        values = tokens + self.modality_identity
        values = values.masked_fill(~available.unsqueeze(-1), 0.0)
        for block in self.blocks:
            values = block(values, available)
        fused, attention = self.gate(values, available)
        return self.classifier(fused), fused, attention
