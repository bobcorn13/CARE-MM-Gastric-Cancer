from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn


class FeatureBackbone(Protocol):
    def __call__(self, values: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class BackboneIdentity:
    family: str
    variant: str
    feature_width: int
    input_size: int


class FrozenBackboneAdapter(nn.Module):
    def __init__(
        self, backbone: nn.Module, identity: BackboneIdentity, output_width: int = 512
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.identity = identity
        self.projection = nn.Sequential(
            nn.LayerNorm(identity.feature_width),
            nn.Linear(identity.feature_width, output_width),
            nn.GELU(),
            nn.LayerNorm(output_width),
        )
        self.backbone.requires_grad_(False)

    def train(self, mode: bool = True) -> "FrozenBackboneAdapter":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(values)
        if isinstance(features, tuple):
            features = features[0]
        if features.ndim > 2:
            features = features.flatten(2).mean(dim=-1)
        return self.projection(features)


class BackboneRegistry:
    def __init__(self) -> None:
        self._constructors: dict[str, callable] = {}
        self._identities: dict[str, BackboneIdentity] = {}

    def register(self, name: str, constructor: callable, identity: BackboneIdentity) -> None:
        if name in self._constructors:
            raise KeyError(f"backbone already registered: {name}")
        self._constructors[name] = constructor
        self._identities[name] = identity

    def create(self, name: str, weights: str | None = None) -> FrozenBackboneAdapter:
        if name not in self._constructors:
            raise KeyError(f"unknown backbone: {name}")
        backbone = self._constructors[name](weights)
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone constructor must return a torch module")
        return FrozenBackboneAdapter(backbone, self._identities[name])

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._constructors))

    def identity(self, name: str) -> BackboneIdentity:
        return self._identities[name]


class TileBatchEncoder:
    def __init__(
        self, adapter: FrozenBackboneAdapter, batch_size: int, device: torch.device
    ) -> None:
        if batch_size < 1:
            raise ValueError("tile batch size must be positive")
        self.adapter = adapter.to(device).eval()
        self.batch_size = batch_size
        self.device = device

    @torch.no_grad()
    def encode(self, tiles: torch.Tensor) -> torch.Tensor:
        outputs = []
        for start in range(0, len(tiles), self.batch_size):
            batch = tiles[start : start + self.batch_size].to(self.device)
            outputs.append(self.adapter(batch).cpu())
        return torch.cat(outputs)


class FrameSequenceEncoder:
    def __init__(
        self,
        adapter: FrozenBackboneAdapter,
        batch_size: int,
        frame_stride: int,
        device: torch.device,
    ) -> None:
        if frame_stride < 1:
            raise ValueError("frame stride must be positive")
        self.encoder = TileBatchEncoder(adapter, batch_size, device)
        self.frame_stride = frame_stride

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        selected = frames[:: self.frame_stride]
        if selected.shape[0] == 0:
            raise ValueError("endoscopy series contains no frames")
        return self.encoder.encode(selected)


def ensemble_features(
    features: tuple[torch.Tensor, ...], method: str = "concatenate"
) -> torch.Tensor:
    if not features:
        raise ValueError("feature ensemble cannot be empty")
    if any(item.shape[:-1] != features[0].shape[:-1] for item in features):
        raise ValueError("ensemble features must share leading dimensions")
    if method == "concatenate":
        return torch.cat(features, dim=-1)
    if method == "mean":
        if len({item.shape[-1] for item in features}) != 1:
            raise ValueError("mean ensemble requires equal feature widths")
        return torch.stack(features).mean(dim=0)
    raise ValueError(f"unknown feature ensemble method: {method}")
