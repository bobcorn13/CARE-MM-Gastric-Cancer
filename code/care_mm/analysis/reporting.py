import csv
import json
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def serializable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(serializable(payload), handle, indent=2, sort_keys=True, allow_nan=False)
    temporary.replace(destination)


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = tuple(rows)
    if not materialized:
        raise ValueError("cannot write an empty result table")
    fields = tuple(materialized[0])
    if any(tuple(row) != fields for row in materialized):
        raise ValueError("result rows must share ordered fields")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)
    temporary.replace(destination)


class ResultBook:
    def __init__(self) -> None:
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self.metadata: dict[str, Any] = {}

    def add_row(self, table: str, row: dict[str, Any]) -> None:
        rows = self.tables.setdefault(table, [])
        if rows and tuple(rows[0]) != tuple(row):
            raise ValueError(f"schema mismatch in table {table}")
        rows.append(row)

    def set_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def save(self, directory: str | Path) -> None:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "metadata.json", self.metadata)
        for name, rows in self.tables.items():
            write_csv(root / f"{name}.csv", rows)

    def table(self, name: str) -> tuple[dict[str, Any], ...]:
        return tuple(self.tables[name])

    def merge(self, other: "ResultBook") -> None:
        overlap = set(self.metadata) & set(other.metadata)
        if overlap:
            raise ValueError(f"duplicate metadata keys: {sorted(overlap)}")
        self.metadata.update(other.metadata)
        for name, rows in other.tables.items():
            for row in rows:
                self.add_row(name, row)


def comparison_row(
    method: str,
    modality: str,
    auroc: float,
    lower: float,
    upper: float,
    sensitivity: float,
    specificity: float,
) -> dict[str, Any]:
    return {
        "method": method,
        "modality": modality,
        "auroc": auroc,
        "ci_lower": lower,
        "ci_upper": upper,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def ablation_row(
    variant: str,
    auroc: float,
    lower: float,
    upper: float,
    ece: float,
    delta_auroc: float,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "auroc_endoscopy_only": auroc,
        "ci_lower": lower,
        "ci_upper": upper,
        "ece": ece,
        "delta_auroc": delta_auroc,
    }


def site_row(
    site: str,
    region: str,
    setting: str,
    availability: str,
    count: int,
    prospective: float,
    lower: float,
    upper: float,
    retrospective: float,
) -> dict[str, Any]:
    return {
        "site": site,
        "region": region,
        "setting": setting,
        "availability": availability,
        "n": count,
        "prospective_auroc": prospective,
        "ci_lower": lower,
        "ci_upper": upper,
        "retrospective_auroc": retrospective,
    }
