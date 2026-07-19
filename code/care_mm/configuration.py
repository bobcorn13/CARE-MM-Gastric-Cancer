from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfiguration:
    width: int
    depth: int
    heads: int
    classes: int
    modalities: int
    dropout: float

    def validate(self) -> None:
        if self.width != 512 or self.depth != 4 or self.heads != 8:
            raise ValueError("primary fusion architecture must use width 512, depth 4 and 8 heads")
        if self.modalities != 3:
            raise ValueError("CARE-MM requires three modality positions")
        if self.width % self.heads:
            raise ValueError("model width must be divisible by attention heads")


@dataclass(frozen=True)
class TrainingConfiguration:
    epochs: int
    learning_rate: float
    optimizer: str
    batch_size: int | None
    weight_decay: float | None
    warmup_steps: int | None
    precision: str | None
    gradient_clipping: float | None

    def validate(self) -> None:
        if self.epochs != 50:
            raise ValueError("primary training schedule must use 50 epochs")
        if self.learning_rate != 1e-4:
            raise ValueError("primary learning rate must be 1e-4")
        if self.optimizer.lower() != "adamw":
            raise ValueError("primary optimizer must be AdamW")
        missing = []
        for name in ("batch_size", "weight_decay", "warmup_steps", "precision"):
            if getattr(self, name) is None:
                missing.append(name)
        if missing:
            raise ValueError(f"paper did not report required runtime values: {', '.join(missing)}")


@dataclass(frozen=True)
class CalibrationConfiguration:
    miscoverage: float
    bins: int

    def validate(self) -> None:
        if self.miscoverage != 0.1:
            raise ValueError("primary conformal miscoverage must be 0.10")
        if self.bins < 2:
            raise ValueError("calibration bins must be at least two")


@dataclass(frozen=True)
class DecisionConfiguration:
    false_negative_cost: float
    false_positive_cost: float
    actions: tuple[str, ...]

    def validate(self) -> None:
        if self.false_negative_cost != 10 * self.false_positive_cost:
            raise ValueError("false-negative cost must be ten times false-positive cost")
        if len(self.actions) != 3:
            raise ValueError("three management actions are required")


@dataclass(frozen=True)
class DataConfiguration:
    manifest: Path
    availability: Path


@dataclass(frozen=True)
class OutputConfiguration:
    directory: Path


@dataclass(frozen=True)
class ExperimentConfiguration:
    seed: int
    model: ModelConfiguration
    training: TrainingConfiguration
    calibration: CalibrationConfiguration
    decision: DecisionConfiguration
    data: DataConfiguration
    output: OutputConfiguration

    def validate(self) -> None:
        self.model.validate()
        self.training.validate()
        self.calibration.validate()
        self.decision.validate()


def load_configuration(path: str | Path) -> ExperimentConfiguration:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError("configuration root must be a mapping")
    return ExperimentConfiguration(
        seed=int(payload["seed"]),
        model=ModelConfiguration(**payload["model"]),
        training=TrainingConfiguration(**payload["training"]),
        calibration=CalibrationConfiguration(**payload["calibration"]),
        decision=DecisionConfiguration(
            false_negative_cost=float(payload["decision"]["false_negative_cost"]),
            false_positive_cost=float(payload["decision"]["false_positive_cost"]),
            actions=tuple(payload["decision"]["actions"]),
        ),
        data=DataConfiguration(
            Path(payload["data"]["manifest"]),
            Path(payload["data"]["availability"]),
        ),
        output=OutputConfiguration(Path(payload["output"]["directory"])),
    )


def apply_overrides(payload: dict[str, Any], overrides: tuple[str, ...]) -> dict[str, Any]:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"invalid override: {override}")
        key, raw = override.split("=", 1)
        path = key.split(".")
        target = payload
        for component in path[:-1]:
            value = target.get(component)
            if not isinstance(value, dict):
                raise KeyError(f"unknown configuration path: {key}")
            target = value
        target[path[-1]] = yaml.safe_load(raw)
    return payload
