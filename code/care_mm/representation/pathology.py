import torch
from torch import nn

from care_mm.representation.common import FeatureProjector, TransformerStack, masked_mean


class GatedAttentionPool(nn.Module):
    def __init__(self, width: int, attention_width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.value = nn.Linear(width, width)
        self.tanh_branch = nn.Linear(width, attention_width)
        self.sigmoid_branch = nn.Linear(width, attention_width)
        self.score = nn.Linear(attention_width, heads)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(width * heads, width)

    def forward(
        self, tiles: torch.Tensor, padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left = torch.tanh(self.tanh_branch(tiles))
        right = torch.sigmoid(self.sigmoid_branch(tiles))
        logits = self.score(self.dropout(left * right)).transpose(1, 2)
        logits = logits.masked_fill(padding_mask.unsqueeze(1), -torch.inf)
        weights = torch.softmax(logits, dim=-1)
        values = self.value(tiles)
        pooled = torch.matmul(weights, values).reshape(tiles.shape[0], -1)
        return self.output(pooled), weights


class PathologyMILEncoder(nn.Module):
    def __init__(
        self,
        input_width: int,
        width: int = 512,
        depth: int = 2,
        heads: int = 8,
        attention_width: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.project = FeatureProjector(input_width, width, dropout)
        self.context = TransformerStack(width, depth, heads, 4, dropout)
        self.pool = GatedAttentionPool(width, attention_width, heads, dropout)
        self.summary = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(width),
        )

    def forward(
        self, tiles: torch.Tensor, padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tiles.ndim != 3:
            raise ValueError("pathology features must have batch, tile and feature dimensions")
        if padding_mask.shape != tiles.shape[:2]:
            raise ValueError("pathology padding mask shape mismatch")
        projected = self.project(tiles)
        contextual = self.context(projected, padding_mask)
        attended, weights = self.pool(contextual, padding_mask)
        average = masked_mean(contextual, padding_mask)
        return self.summary(torch.cat((attended, average), dim=-1)), weights


class MultiBackbonePathologyEncoder(nn.Module):
    def __init__(
        self, input_widths: tuple[int, ...], width: int = 512, dropout: float = 0.1
    ) -> None:
        super().__init__()
        if not input_widths:
            raise ValueError("at least one pathology backbone is required")
        self.encoders = nn.ModuleList(
            [PathologyMILEncoder(item, width=width, dropout=dropout) for item in input_widths]
        )
        self.source_gate = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, len(input_widths)),
        )
        self.output = nn.Sequential(nn.Linear(width, width), nn.LayerNorm(width))

    def forward(
        self,
        features: tuple[torch.Tensor, ...],
        padding_masks: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(features) != len(self.encoders) or len(padding_masks) != len(self.encoders):
            raise ValueError("pathology backbone input count mismatch")
        embeddings = []
        for encoder, values, mask in zip(self.encoders, features, padding_masks, strict=True):
            embedding, _ = encoder(values, mask)
            embeddings.append(embedding)
        stacked = torch.stack(embeddings, dim=1)
        consensus = stacked.mean(dim=1)
        weights = torch.softmax(self.source_gate(consensus), dim=-1)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        return self.output(fused), weights
