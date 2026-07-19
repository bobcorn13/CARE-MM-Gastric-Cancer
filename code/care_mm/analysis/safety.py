from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HarmProfile:
    false_positive_rate: float
    false_negative_rate: float
    false_alarms_per_patient_hour: float
    weighted_harm: float
    early_operable_false_negative_rate: float


def harm_profile(
    labels: np.ndarray,
    predictions: np.ndarray,
    observation_hours: np.ndarray,
    early_operable: np.ndarray,
    false_negative_cost: float = 10.0,
    false_positive_cost: float = 1.0,
) -> HarmProfile:
    arrays = (labels, predictions, observation_hours, early_operable)
    if len({np.asarray(item).shape for item in arrays}) != 1:
        raise ValueError("harm profile inputs must align")
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    observation_hours = np.asarray(observation_hours, dtype=np.float64)
    early_operable = np.asarray(early_operable, dtype=bool)
    false_positive = predictions & ~labels
    false_negative = ~predictions & labels
    negative_count = max(int((~labels).sum()), 1)
    positive_count = max(int(labels.sum()), 1)
    early_count = max(int((labels & early_operable).sum()), 1)
    weighted = (
        false_positive_cost * false_positive.sum() + false_negative_cost * false_negative.sum()
    ) / len(labels)
    return HarmProfile(
        false_positive_rate=float(false_positive.sum() / negative_count),
        false_negative_rate=float(false_negative.sum() / positive_count),
        false_alarms_per_patient_hour=float(false_positive.sum() / observation_hours.sum()),
        weighted_harm=float(weighted),
        early_operable_false_negative_rate=float(
            (false_negative & early_operable).sum() / early_count
        ),
    )


@dataclass(frozen=True)
class FailureMode:
    name: str
    count: int
    prevalence: float
    abstained: int
    abstention_rate: float
    errors: int
    error_rate: float


def failure_modes(
    mode_labels: np.ndarray,
    correct: np.ndarray,
    abstained: np.ndarray,
) -> tuple[FailureMode, ...]:
    mode_labels = np.asarray(mode_labels)
    correct = np.asarray(correct, dtype=bool)
    abstained = np.asarray(abstained, dtype=bool)
    if mode_labels.shape != correct.shape or mode_labels.shape != abstained.shape:
        raise ValueError("failure mode arrays must align")
    output = []
    for name in np.unique(mode_labels):
        selected = mode_labels == name
        count = int(selected.sum())
        deferred = int(abstained[selected].sum())
        errors = int((~correct[selected]).sum())
        output.append(
            FailureMode(
                str(name),
                count,
                count / len(mode_labels),
                deferred,
                deferred / count,
                errors,
                errors / count,
            )
        )
    return tuple(output)
