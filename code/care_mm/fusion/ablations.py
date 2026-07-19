from enum import Enum

import torch
from torch import nn

from care_mm.fusion.gate import FusionHead


class Imputation(str, Enum):
    ZERO = "zero"
    MEAN = "mean"


class ImputationFusion(nn.Module):
    def __init__(self, width: int, classes: int, method: Imputation, dropout: float = 0.1) -> None:
        super().__init__()
        self.method = method
        self.register_buffer("running_mean", torch.zeros(3, width))
        self.register_buffer("running_count", torch.zeros(3))
        self.classifier = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, classes),
        )

    def update_means(self, tokens: torch.Tensor, available: torch.Tensor) -> None:
        with torch.no_grad():
            for modality in range(3):
                selected = tokens[available[:, modality], modality]
                if selected.numel() == 0:
                    continue
                count = selected.shape[0]
                total = self.running_count[modality] + count
                previous = self.running_mean[modality] * self.running_count[modality]
                self.running_mean[modality] = (previous + selected.sum(dim=0)) / total
                self.running_count[modality] = total

    def forward(self, tokens: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        if self.training and self.method == Imputation.MEAN:
            self.update_means(tokens, available)
        if self.method == Imputation.ZERO:
            replacements = torch.zeros_like(tokens)
        else:
            replacements = self.running_mean.unsqueeze(0).expand_as(tokens)
        completed = torch.where(available.unsqueeze(-1), tokens, replacements)
        return self.classifier(completed.flatten(1))


class GateOnlyFusion(nn.Module):
    def __init__(self, width: int = 512, depth: int = 4, heads: int = 8, classes: int = 2) -> None:
        super().__init__()
        self.head = FusionHead(width, depth, heads, classes)

    def forward(self, tokens: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.head(tokens, available)
        return logits
