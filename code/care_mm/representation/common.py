from collections.abc import Sequence

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, width: int, expansion: int, dropout: float) -> None:
        super().__init__()
        hidden = width * expansion
        self.network = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class PreNormBlock(nn.Module):
    def __init__(self, width: int, heads: int, expansion: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = FeedForward(width, expansion, dropout)

    def forward(
        self, values: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        normalized = self.attention_norm(values)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        values = values + attended
        return values + self.feedforward(self.feedforward_norm(values))


class TransformerStack(nn.Module):
    def __init__(self, width: int, depth: int, heads: int, expansion: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [PreNormBlock(width, heads, expansion, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(width)

    def forward(
        self, values: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for block in self.blocks:
            values = block(values, padding_mask)
        return self.norm(values)


class FeatureProjector(nn.Module):
    def __init__(self, input_width: int, output_width: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, output_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_width, output_width),
            nn.LayerNorm(output_width),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def padded_stack(sequences: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("at least one sequence is required")
    width = sequences[0].shape[-1]
    if any(item.ndim != 2 or item.shape[-1] != width for item in sequences):
        raise ValueError("sequences must share feature width")
    maximum = max(item.shape[0] for item in sequences)
    output = sequences[0].new_zeros((len(sequences), maximum, width))
    padding = torch.ones((len(sequences), maximum), dtype=torch.bool, device=output.device)
    for row, item in enumerate(sequences):
        output[row, : item.shape[0]] = item
        padding[row, : item.shape[0]] = False
    return output, padding


def masked_mean(values: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    valid = (~padding_mask).unsqueeze(-1).to(values.dtype)
    denominator = valid.sum(dim=1).clamp_min(1.0)
    return (values * valid).sum(dim=1) / denominator
