from pathlib import Path

import torch


def require_file(path: str | Path, label: str) -> Path:
    value = Path(path)
    if not value.is_file():
        raise FileNotFoundError(f"{label} does not exist: {value}")
    return value


def require_directory(path: str | Path, label: str) -> Path:
    value = Path(path)
    if not value.is_dir():
        raise NotADirectoryError(f"{label} does not exist: {value}")
    return value


def require_shape(value: torch.Tensor, expected: tuple[int | None, ...], label: str) -> None:
    if value.ndim != len(expected):
        raise ValueError(f"{label} rank mismatch")
    for observed, required in zip(value.shape, expected, strict=True):
        if required is not None and observed != required:
            raise ValueError(f"{label} shape mismatch")


def require_finite(value: torch.Tensor, label: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} contains non-finite values")


def require_probabilities(value: torch.Tensor, label: str, tolerance: float = 1e-5) -> None:
    require_finite(value, label)
    if value.ndim != 2:
        raise ValueError(f"{label} must be a matrix")
    if torch.any(value < 0) or torch.any(value > 1):
        raise ValueError(f"{label} contains values outside [0, 1]")
    if not torch.allclose(
        value.sum(dim=1), torch.ones(value.shape[0], device=value.device), atol=tolerance
    ):
        raise ValueError(f"{label} rows must sum to one")


def require_binary_labels(value: torch.Tensor, label: str) -> None:
    if value.ndim != 1:
        raise ValueError(f"{label} must be a vector")
    if torch.any((value != 0) & (value != 1)):
        raise ValueError(f"{label} must be binary")


def require_availability(value: torch.Tensor) -> None:
    if value.dtype != torch.bool or value.ndim != 2 or value.shape[1] != 3:
        raise ValueError("availability must be a batch by three boolean matrix")
    if torch.any(value.sum(dim=1) == 0):
        raise ValueError("each case requires at least one available modality")


def require_patient_independence(*splits: tuple[str, ...]) -> None:
    seen: set[str] = set()
    for split in splits:
        current = set(split)
        if len(current) != len(split):
            raise ValueError("patient identifiers repeat within a split")
        overlap = seen & current
        if overlap:
            raise ValueError(f"patient leakage across splits: {sorted(overlap)[:3]}")
        seen.update(current)
