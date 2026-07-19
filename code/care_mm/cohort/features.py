import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


@dataclass(frozen=True)
class FeatureRecord:
    case_id: str
    features: torch.Tensor
    source: str
    version: str
    checksum: str


class HDF5FeatureStore:
    def __init__(self, path: str | Path, mode: str = "r") -> None:
        self.path = Path(path)
        self.mode = mode
        self._handle: h5py.File | None = None

    def open(self) -> "HDF5FeatureStore":
        self._handle = h5py.File(self.path, self.mode)
        return self

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "HDF5FeatureStore":
        return self.open()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @property
    def handle(self) -> h5py.File:
        if self._handle is None:
            raise RuntimeError("feature store is closed")
        return self._handle

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.handle.keys()))

    def contains(self, case_id: str) -> bool:
        return case_id in self.handle

    def read(self, case_id: str) -> FeatureRecord:
        group = self.handle[case_id]
        values = torch.from_numpy(np.asarray(group["features"]))
        return FeatureRecord(
            case_id,
            values,
            str(group.attrs.get("source", "unknown")),
            str(group.attrs.get("version", "unknown")),
            str(group.attrs.get("checksum", "")),
        )

    def write(self, record: FeatureRecord, compression: str = "gzip") -> None:
        if record.case_id in self.handle:
            raise KeyError(f"feature record already exists: {record.case_id}")
        group = self.handle.create_group(record.case_id)
        group.create_dataset(
            "features", data=record.features.detach().cpu().numpy(), compression=compression
        )
        group.attrs["source"] = record.source
        group.attrs["version"] = record.version
        group.attrs["checksum"] = record.checksum
        self.handle.flush()

    def delete(self, case_id: str) -> None:
        del self.handle[case_id]
        self.handle.flush()

    def metadata(self) -> dict[str, dict[str, Any]]:
        output = {}
        for key in self.keys():
            group = self.handle[key]
            output[key] = {
                "shape": tuple(group["features"].shape),
                "dtype": str(group["features"].dtype),
                "source": str(group.attrs.get("source", "unknown")),
                "version": str(group.attrs.get("version", "unknown")),
                "checksum": str(group.attrs.get("checksum", "")),
            }
        return output


class JSONFeatureIndex:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records: dict[str, dict[str, Any]] = {}

    def load(self) -> None:
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError("feature index must contain an object")
        self.records = payload

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.records, handle, indent=2, sort_keys=True)
        temporary.replace(self.path)

    def add(self, case_id: str, modality: str, path: str, shape: tuple[int, ...]) -> None:
        case = self.records.setdefault(case_id, {})
        if modality in case:
            raise KeyError(f"duplicate feature index entry: {case_id}/{modality}")
        case[modality] = {"path": path, "shape": list(shape)}

    def lookup(self, case_id: str, modality: str) -> dict[str, Any]:
        return self.records[case_id][modality]
