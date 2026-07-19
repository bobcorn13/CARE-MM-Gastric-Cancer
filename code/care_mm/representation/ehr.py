import torch
from torch import nn

from care_mm.representation.common import TransformerStack


class NumericalTokenizer(nn.Module):
    def __init__(self, features: int, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(features, width))
        self.bias = nn.Parameter(torch.empty(features, width))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.weight.shape[0]:
            raise ValueError("numerical feature count mismatch")
        return values.unsqueeze(-1) * self.weight + self.bias


class CategoricalTokenizer(nn.Module):
    def __init__(self, cardinalities: tuple[int, ...], width: int) -> None:
        super().__init__()
        offsets = torch.tensor((0,) + cardinalities[:-1]).cumsum(dim=0)
        self.register_buffer("offsets", offsets, persistent=False)
        self.embedding = nn.Embedding(sum(cardinalities), width)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.offsets.shape[0]:
            raise ValueError("categorical feature count mismatch")
        return self.embedding(values + self.offsets)


class FTTransformerEncoder(nn.Module):
    def __init__(
        self,
        numerical_features: int,
        categorical_cardinalities: tuple[int, ...],
        width: int = 512,
        depth: int = 4,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if numerical_features == 0 and not categorical_cardinalities:
            raise ValueError("EHR encoder requires at least one feature")
        self.numerical = (
            NumericalTokenizer(numerical_features, width) if numerical_features else None
        )
        self.categorical = (
            CategoricalTokenizer(categorical_cardinalities, width)
            if categorical_cardinalities
            else None
        )
        self.cls = nn.Parameter(torch.empty(1, 1, width))
        self.missing = nn.Parameter(torch.empty(1, 1, width))
        self.transformer = TransformerStack(width, depth, heads, 4, dropout)
        self.output = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width))
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.missing, std=0.02)

    def forward(
        self,
        numerical: torch.Tensor | None,
        categorical: torch.Tensor | None,
        missing_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        components: list[torch.Tensor] = []
        if self.numerical is not None:
            if numerical is None:
                raise ValueError("numerical values are required")
            components.append(self.numerical(numerical))
        if self.categorical is not None:
            if categorical is None:
                raise ValueError("categorical values are required")
            components.append(self.categorical(categorical))
        tokens = torch.cat(components, dim=1)
        if missing_mask is not None:
            if missing_mask.shape != tokens.shape[:2]:
                raise ValueError("EHR missing mask shape mismatch")
            tokens = torch.where(missing_mask.unsqueeze(-1), self.missing, tokens)
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        encoded = self.transformer(torch.cat((cls, tokens), dim=1))
        return self.output(encoded[:, 0])
