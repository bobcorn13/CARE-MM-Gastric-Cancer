import csv
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class NumericalField:
    name: str
    lower: float | None
    upper: float | None
    median: float
    mean: float
    standard_deviation: float


@dataclass(frozen=True)
class CategoricalField:
    name: str
    vocabulary: tuple[str, ...]
    unknown_index: int


@dataclass(frozen=True)
class LongitudinalEvent:
    case_id: str
    timestamp: datetime
    event_type: str
    value: float
    unit: str


class EHRSchema:
    def __init__(
        self,
        numerical: tuple[NumericalField, ...],
        categorical: tuple[CategoricalField, ...],
    ) -> None:
        self.numerical = numerical
        self.categorical = categorical
        names = [item.name for item in numerical] + [item.name for item in categorical]
        if len(names) != len(set(names)):
            raise ValueError("EHR field names must be unique")

    @classmethod
    def fit(
        cls,
        rows: Iterable[dict[str, str]],
        numerical_names: tuple[str, ...],
        categorical_names: tuple[str, ...],
    ) -> "EHRSchema":
        materialized = tuple(rows)
        numerical = []
        for name in numerical_names:
            values = np.asarray(
                [float(row[name]) for row in materialized if row.get(name, "") != ""],
                dtype=np.float64,
            )
            if values.size == 0:
                raise ValueError(f"numerical EHR field has no observations: {name}")
            numerical.append(
                NumericalField(
                    name,
                    float(values.min()),
                    float(values.max()),
                    float(np.median(values)),
                    float(values.mean()),
                    float(values.std()) if values.size > 1 else 1.0,
                )
            )
        categorical = []
        for name in categorical_names:
            values = sorted({row[name] for row in materialized if row.get(name, "") != ""})
            categorical.append(CategoricalField(name, tuple(values), len(values)))
        return cls(tuple(numerical), tuple(categorical))

    def transform(
        self, rows: Iterable[dict[str, str]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        materialized = tuple(rows)
        numerical = torch.zeros((len(materialized), len(self.numerical)), dtype=torch.float32)
        categorical = torch.zeros((len(materialized), len(self.categorical)), dtype=torch.long)
        missing = torch.zeros(
            (len(materialized), len(self.numerical) + len(self.categorical)),
            dtype=torch.bool,
        )
        for row_index, row in enumerate(materialized):
            for column, field in enumerate(self.numerical):
                raw = row.get(field.name, "")
                if raw == "":
                    value = field.median
                    missing[row_index, column] = True
                else:
                    value = float(raw)
                denominator = max(field.standard_deviation, 1e-8)
                numerical[row_index, column] = (value - field.mean) / denominator
            offset = len(self.numerical)
            for column, field in enumerate(self.categorical):
                raw = row.get(field.name, "")
                try:
                    value = field.vocabulary.index(raw)
                except ValueError:
                    value = field.unknown_index
                    missing[row_index, offset + column] = True
                categorical[row_index, column] = value
        return numerical, categorical, missing

    def cardinalities(self) -> tuple[int, ...]:
        return tuple(len(item.vocabulary) + 1 for item in self.categorical)


def load_ehr_rows(path: str | Path) -> tuple[dict[str, str], ...]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def load_longitudinal_events(path: str | Path) -> tuple[LongitudinalEvent, ...]:
    output = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            output.append(
                LongitudinalEvent(
                    row["case_id"],
                    datetime.fromisoformat(row["timestamp"]),
                    row["event_type"],
                    float(row["value"]),
                    row["unit"],
                )
            )
    return tuple(output)


def aggregate_events(
    events: tuple[LongitudinalEvent, ...],
    event_types: tuple[str, ...],
) -> dict[str, np.ndarray]:
    cases = sorted({item.case_id for item in events})
    output = {}
    for case_id in cases:
        vector = []
        case_events = [item for item in events if item.case_id == case_id]
        for event_type in event_types:
            values = [item.value for item in case_events if item.event_type == event_type]
            if values:
                vector.extend((values[-1], min(values), max(values), float(np.mean(values))))
            else:
                vector.extend((np.nan, np.nan, np.nan, np.nan))
        output[case_id] = np.asarray(vector, dtype=np.float32)
    return output
