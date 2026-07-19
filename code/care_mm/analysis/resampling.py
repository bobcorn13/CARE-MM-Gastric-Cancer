from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

Statistic = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    level: float
    resamples: int
    valid_resamples: int


def patient_bootstrap(
    patient_ids: np.ndarray,
    statistic: Statistic,
    resamples: int = 10000,
    level: float = 0.95,
    seed: int = 2026,
) -> ConfidenceInterval:
    identifiers = np.asarray(patient_ids)
    if identifiers.ndim != 1 or identifiers.size == 0:
        raise ValueError("patient identifiers must be a non-empty vector")
    if len(np.unique(identifiers)) != len(identifiers):
        raise ValueError("patient bootstrap expects one row per patient")
    if not 0 < level < 1:
        raise ValueError("confidence level must lie between zero and one")
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        sampled = generator.integers(0, identifiers.size, identifiers.size)
        try:
            value = float(statistic(sampled))
        except ValueError:
            continue
        if np.isfinite(value):
            values.append(value)
    if not values:
        raise ValueError("no valid bootstrap resamples")
    alpha = 1 - level
    lower, upper = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    estimate = float(statistic(np.arange(identifiers.size)))
    return ConfidenceInterval(estimate, float(lower), float(upper), level, resamples, len(values))


def stratified_bootstrap_indices(
    strata: np.ndarray,
    resamples: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    strata = np.asarray(strata)
    if strata.ndim != 1:
        raise ValueError("strata must be a vector")
    generator = np.random.default_rng(seed)
    groups = [np.where(strata == value)[0] for value in np.unique(strata)]
    output = []
    for _ in range(resamples):
        pieces = [generator.choice(group, len(group), replace=True) for group in groups]
        output.append(np.concatenate(pieces))
    return tuple(output)


def bootstrap_metrics(
    patient_ids: np.ndarray,
    metrics: Mapping[str, Statistic],
    resamples: int = 10000,
    seed: int = 2026,
) -> dict[str, ConfidenceInterval]:
    return {
        name: patient_bootstrap(patient_ids, metric, resamples, seed=seed + index)
        for index, (name, metric) in enumerate(metrics.items())
    }


def paired_bootstrap_difference(
    patient_ids: np.ndarray,
    first: Statistic,
    second: Statistic,
    resamples: int = 10000,
    level: float = 0.95,
    seed: int = 2026,
) -> ConfidenceInterval:
    def difference(indices: np.ndarray) -> float:
        return first(indices) - second(indices)

    return patient_bootstrap(patient_ids, difference, resamples, level, seed)
