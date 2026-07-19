from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import NamedTuple

import torch


class Modality(IntEnum):
    ENDOSCOPY = 0
    PATHOLOGY = 1
    EHR = 2


class Action(IntEnum):
    BIOPSY = 0
    ENDOSCOPIC_RESECTION = 1
    SURGICAL_REFERRAL = 2
    ABSTAIN = 3


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    site: str
    region: str
    setting: str
    label: int
    triage: int
    endoscopy_path: Path | None
    pathology_path: Path | None
    ehr_path: Path | None

    @property
    def mask(self) -> tuple[bool, bool, bool]:
        return (
            self.endoscopy_path is not None,
            self.pathology_path is not None,
            self.ehr_path is not None,
        )


class EncodedBatch(NamedTuple):
    tokens: torch.Tensor
    mask: torch.Tensor
    labels: torch.Tensor
    triage: torch.Tensor
    case_ids: tuple[str, ...]


class ModelOutput(NamedTuple):
    logits: torch.Tensor
    probabilities: torch.Tensor
    fused: torch.Tensor
    attention: torch.Tensor


@dataclass(frozen=True)
class RouteResult:
    action: Action
    expected_cost: float
    prediction_set: tuple[int, ...]
    probabilities: tuple[float, ...]
