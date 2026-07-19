from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from sklearn.metrics import roc_auc_score

from care_mm.analysis.discrimination import BinaryMetrics, binary_metrics


class CohortArm(str, Enum):
    PROSPECTIVE = "prospective"
    RETROSPECTIVE = "retrospective"


class CareSetting(str, Enum):
    TERTIARY = "tertiary"
    PROVINCIAL = "provincial"
    COMMUNITY = "community"


class ModalityProfile(str, Enum):
    ENDOSCOPY = "endoscopy"
    PATHOLOGY = "pathology"
    EHR = "ehr"
    ENDOSCOPY_EHR = "endoscopy_ehr"
    ENDOSCOPY_PATHOLOGY = "endoscopy_pathology"
    PATHOLOGY_EHR = "pathology_ehr"
    TRIMODAL = "trimodal"


@dataclass(frozen=True)
class PredictionFrame:
    case_ids: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    sites: np.ndarray
    regions: np.ndarray
    settings: np.ndarray
    arms: np.ndarray
    masks: np.ndarray
    triage_labels: np.ndarray
    triage_actions: np.ndarray
    early_operable: np.ndarray

    def validate(self) -> None:
        arrays = (
            self.case_ids,
            self.labels,
            self.scores,
            self.sites,
            self.regions,
            self.settings,
            self.arms,
            self.masks,
            self.triage_labels,
            self.triage_actions,
            self.early_operable,
        )
        shapes = {np.asarray(item).shape for item in arrays}
        if len(shapes) != 1:
            raise ValueError("prediction frame columns must align")
        if self.labels.ndim != 1:
            raise ValueError("prediction frame columns must be vectors")
        if len(np.unique(self.case_ids)) != len(self.case_ids):
            raise ValueError("prediction frame case identifiers must be unique")
        if np.any((self.labels < 0) | (self.labels > 1)):
            raise ValueError("diagnosis labels must be binary")
        if np.any((self.scores < 0) | (self.scores > 1)):
            raise ValueError("diagnosis scores must be probabilities")

    def subset(self, selector: np.ndarray) -> "PredictionFrame":
        if selector.dtype != bool or selector.shape != self.labels.shape:
            raise ValueError("prediction frame selector must be an aligned boolean vector")
        return PredictionFrame(
            self.case_ids[selector],
            self.labels[selector],
            self.scores[selector],
            self.sites[selector],
            self.regions[selector],
            self.settings[selector],
            self.arms[selector],
            self.masks[selector],
            self.triage_labels[selector],
            self.triage_actions[selector],
            self.early_operable[selector],
        )

    def by_site(self, site: str) -> "PredictionFrame":
        return self.subset(self.sites == site)

    def by_region(self, region: str) -> "PredictionFrame":
        return self.subset(self.regions == region)

    def by_setting(self, setting: CareSetting) -> "PredictionFrame":
        return self.subset(self.settings == setting.value)

    def by_arm(self, arm: CohortArm) -> "PredictionFrame":
        return self.subset(self.arms == arm.value)

    def by_mask(self, code: int) -> "PredictionFrame":
        return self.subset(self.masks == code)

    def diagnosis_metrics(self, threshold: float | None = None) -> BinaryMetrics:
        return binary_metrics(self.labels, self.scores, threshold)

    def triage_concordance(self, early_only: bool = False) -> float:
        selected = (
            self.early_operable.astype(bool)
            if early_only
            else np.ones(len(self.labels), dtype=bool)
        )
        if selected.sum() == 0:
            return float("nan")
        return float(np.mean(self.triage_labels[selected] == self.triage_actions[selected]))


@dataclass(frozen=True)
class ExperimentResult:
    name: str
    cohort: str
    count: int
    prevalence: float
    auroc: float
    sensitivity: float
    specificity: float
    triage_concordance: float
    early_operable_concordance: float


class ExperimentRegistry:
    def __init__(self) -> None:
        self._selectors: dict[str, Callable[[PredictionFrame], PredictionFrame]] = {}

    def register(self, name: str, selector: Callable[[PredictionFrame], PredictionFrame]) -> None:
        if name in self._selectors:
            raise KeyError(f"experiment already registered: {name}")
        self._selectors[name] = selector

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._selectors))

    def select(self, name: str, frame: PredictionFrame) -> PredictionFrame:
        return self._selectors[name](frame)

    def evaluate(self, name: str, frame: PredictionFrame) -> ExperimentResult:
        selected = self.select(name, frame)
        selected.validate()
        metrics = selected.diagnosis_metrics()
        return ExperimentResult(
            name,
            name,
            len(selected.labels),
            float(selected.labels.mean()),
            metrics.auroc,
            metrics.sensitivity,
            metrics.specificity,
            selected.triage_concordance(),
            selected.triage_concordance(early_only=True),
        )

    def evaluate_all(self, frame: PredictionFrame) -> tuple[ExperimentResult, ...]:
        return tuple(self.evaluate(name, frame) for name in self.names())


def default_registry() -> ExperimentRegistry:
    registry = ExperimentRegistry()
    registry.register("prospective_primary", lambda frame: frame.by_arm(CohortArm.PROSPECTIVE))
    registry.register("retrospective_support", lambda frame: frame.by_arm(CohortArm.RETROSPECTIVE))
    registry.register(
        "prospective_community",
        lambda frame: frame.by_arm(CohortArm.PROSPECTIVE).by_setting(CareSetting.COMMUNITY),
    )
    registry.register(
        "prospective_provincial",
        lambda frame: frame.by_arm(CohortArm.PROSPECTIVE).by_setting(CareSetting.PROVINCIAL),
    )
    registry.register(
        "prospective_tertiary",
        lambda frame: frame.by_arm(CohortArm.PROSPECTIVE).by_setting(CareSetting.TERTIARY),
    )
    registry.register(
        "community_endoscopy_only",
        lambda frame: (
            frame.by_arm(CohortArm.PROSPECTIVE).by_setting(CareSetting.COMMUNITY).by_mask(1)
        ),
    )
    registry.register("endoscopy_only", lambda frame: frame.by_mask(1))
    registry.register("pathology_only", lambda frame: frame.by_mask(2))
    registry.register("ehr_only", lambda frame: frame.by_mask(4))
    registry.register("endoscopy_pathology", lambda frame: frame.by_mask(3))
    registry.register("endoscopy_ehr", lambda frame: frame.by_mask(5))
    registry.register("pathology_ehr", lambda frame: frame.by_mask(6))
    registry.register("trimodal", lambda frame: frame.by_mask(7))
    return registry


@dataclass(frozen=True)
class PrespecifiedCriterion:
    name: str
    direction: str
    threshold: float
    observed: float
    passed: bool


def criterion(
    name: str, observed: float, threshold: float, direction: str
) -> PrespecifiedCriterion:
    if direction == "minimum":
        passed = observed >= threshold
    elif direction == "maximum":
        passed = observed <= threshold
    else:
        raise ValueError("criterion direction must be minimum or maximum")
    return PrespecifiedCriterion(name, direction, threshold, observed, passed)


def primary_criteria(
    auroc: float,
    triage_concordance: float,
    calibration_error: float,
    auroc_minimum: float,
    concordance_minimum: float,
    calibration_maximum: float,
) -> tuple[PrespecifiedCriterion, ...]:
    return (
        criterion("diagnostic_auroc", auroc, auroc_minimum, "minimum"),
        criterion(
            "early_operable_triage_concordance", triage_concordance, concordance_minimum, "minimum"
        ),
        criterion("expected_calibration_error", calibration_error, calibration_maximum, "maximum"),
    )


def leave_one_component_out(
    labels: np.ndarray,
    full_scores: np.ndarray,
    component_scores: dict[str, np.ndarray],
) -> dict[str, float]:
    full_auc = float(roc_auc_score(labels, full_scores))
    output = {}
    for name, scores in component_scores.items():
        if scores.shape != labels.shape:
            raise ValueError(f"component score shape mismatch: {name}")
        output[name] = float(roc_auc_score(labels, scores) - full_auc)
    return output


def component_synergy(
    labels: np.ndarray,
    full_scores: np.ndarray,
    without_first: np.ndarray,
    without_second: np.ndarray,
    without_both: np.ndarray,
) -> float:
    full = roc_auc_score(labels, full_scores)
    first_drop = full - roc_auc_score(labels, without_first)
    second_drop = full - roc_auc_score(labels, without_second)
    joint_drop = full - roc_auc_score(labels, without_both)
    return float(joint_drop - first_drop - second_drop)


def paired_site_metrics(frame: PredictionFrame) -> dict[str, BinaryMetrics]:
    output = {}
    for site in np.unique(frame.sites):
        selected = frame.by_site(str(site))
        if np.unique(selected.labels).size < 2:
            continue
        output[str(site)] = selected.diagnosis_metrics()
    return output


def aggregate_results(results: Iterable[ExperimentResult]) -> dict[str, float]:
    materialized = tuple(results)
    if not materialized:
        raise ValueError("experiment results cannot be empty")
    weights = np.asarray([item.count for item in materialized], dtype=np.float64)
    weights /= weights.sum()
    return {
        "auroc": float(np.sum(weights * [item.auroc for item in materialized])),
        "sensitivity": float(np.sum(weights * [item.sensitivity for item in materialized])),
        "specificity": float(np.sum(weights * [item.specificity for item in materialized])),
        "triage_concordance": float(
            np.sum(weights * [item.triage_concordance for item in materialized])
        ),
    }
