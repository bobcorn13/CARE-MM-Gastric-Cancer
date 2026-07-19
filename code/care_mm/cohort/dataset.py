from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from care_mm.cohort.features import HDF5FeatureStore
from care_mm.cohort.manifest import CohortManifest
from care_mm.representation.common import padded_stack

FeatureLoader = Callable[[Path], torch.Tensor]


@dataclass(frozen=True)
class CaseFeatures:
    case_id: str
    endoscopy: torch.Tensor | None
    pathology: torch.Tensor | None
    ehr_numerical: torch.Tensor | None
    ehr_categorical: torch.Tensor | None
    ehr_missing: torch.Tensor | None
    available: torch.Tensor
    label: torch.Tensor
    triage: torch.Tensor
    site: str
    setting: str


@dataclass(frozen=True)
class FeatureBatch:
    case_ids: tuple[str, ...]
    endoscopy: torch.Tensor | None
    endoscopy_padding: torch.Tensor | None
    pathology: torch.Tensor | None
    pathology_padding: torch.Tensor | None
    ehr_numerical: torch.Tensor | None
    ehr_categorical: torch.Tensor | None
    ehr_missing: torch.Tensor | None
    available: torch.Tensor
    labels: torch.Tensor
    triage: torch.Tensor
    sites: tuple[str, ...]
    settings: tuple[str, ...]

    def to(self, device: torch.device) -> "FeatureBatch":
        def move(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.to(device)

        return FeatureBatch(
            self.case_ids,
            move(self.endoscopy),
            move(self.endoscopy_padding),
            move(self.pathology),
            move(self.pathology_padding),
            move(self.ehr_numerical),
            move(self.ehr_categorical),
            move(self.ehr_missing),
            self.available.to(device),
            self.labels.to(device),
            self.triage.to(device),
            self.sites,
            self.settings,
        )


class MultimodalFeatureDataset(Dataset[CaseFeatures]):
    def __init__(
        self,
        manifest: CohortManifest,
        endoscopy_loader: FeatureLoader | None = None,
        pathology_loader: FeatureLoader | None = None,
        ehr_loader: Callable[[Path], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
    ) -> None:
        self.manifest = manifest
        self.endoscopy_loader = endoscopy_loader or torch.load
        self.pathology_loader = pathology_loader or torch.load
        self.ehr_loader = ehr_loader or load_ehr_tensor

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> CaseFeatures:
        record = self.manifest[index]
        endoscopy = (
            None if record.endoscopy_path is None else self.endoscopy_loader(record.endoscopy_path)
        )
        pathology = (
            None if record.pathology_path is None else self.pathology_loader(record.pathology_path)
        )
        if record.ehr_path is None:
            numerical = categorical = missing = None
        else:
            numerical, categorical, missing = self.ehr_loader(record.ehr_path)
        return CaseFeatures(
            record.case_id,
            endoscopy,
            pathology,
            numerical,
            categorical,
            missing,
            torch.tensor(record.mask, dtype=torch.bool),
            torch.tensor(record.label, dtype=torch.long),
            torch.tensor(record.triage, dtype=torch.long),
            record.site,
            record.setting,
        )


def load_ehr_tensor(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("EHR tensor file must contain a mapping")
    return payload["numerical"], payload["categorical"], payload["missing"]


def _optional_stack(
    values: Sequence[torch.Tensor | None],
    modality_index: int,
    available: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    present = [item for item in values if item is not None]
    if not present:
        return None, None
    template = present[0]
    completed = []
    for row, item in enumerate(values):
        if item is not None:
            completed.append(item)
        else:
            available[row, modality_index] = False
            completed.append(template.new_zeros((1, template.shape[-1])))
    return padded_stack(completed)


def collate_features(cases: Sequence[CaseFeatures]) -> FeatureBatch:
    if not cases:
        raise ValueError("cannot collate an empty batch")
    available = torch.stack([item.available for item in cases])
    endoscopy, endoscopy_padding = _optional_stack([item.endoscopy for item in cases], 0, available)
    pathology, pathology_padding = _optional_stack([item.pathology for item in cases], 1, available)
    numerical_values = [item.ehr_numerical for item in cases]
    categorical_values = [item.ehr_categorical for item in cases]
    missing_values = [item.ehr_missing for item in cases]
    if any(item is not None for item in numerical_values):
        numerical = torch.stack(
            [
                item
                if item is not None
                else torch.zeros_like(next(x for x in numerical_values if x is not None))
                for item in numerical_values
            ]
        )
        categorical = torch.stack(
            [
                item
                if item is not None
                else torch.zeros_like(next(x for x in categorical_values if x is not None))
                for item in categorical_values
            ]
        )
        missing = torch.stack(
            [
                item
                if item is not None
                else torch.ones_like(next(x for x in missing_values if x is not None))
                for item in missing_values
            ]
        )
    else:
        numerical = categorical = missing = None
    return FeatureBatch(
        tuple(item.case_id for item in cases),
        endoscopy,
        endoscopy_padding,
        pathology,
        pathology_padding,
        numerical,
        categorical,
        missing,
        available,
        torch.stack([item.label for item in cases]),
        torch.stack([item.triage for item in cases]),
        tuple(item.site for item in cases),
        tuple(item.setting for item in cases),
    )


class StoreBackedLoader:
    def __init__(self, store_path: str | Path) -> None:
        self.store_path = Path(store_path)

    def __call__(self, reference: Path) -> torch.Tensor:
        with HDF5FeatureStore(self.store_path) as store:
            return store.read(reference.stem).features
