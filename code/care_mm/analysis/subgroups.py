from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SubgroupEstimate:
    dimension: str
    stratum: str
    count: int
    positives: int
    estimate: float


@dataclass(frozen=True)
class GapEstimate:
    dimension: str
    minimum_stratum: str
    maximum_stratum: str
    minimum: float
    maximum: float
    gap: float


def subgroup_estimates(
    labels: np.ndarray,
    scores: np.ndarray,
    dimensions: Mapping[str, np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[SubgroupEstimate, ...]:
    output = []
    for dimension, strata in dimensions.items():
        if len(strata) != len(labels):
            raise ValueError(f"subgroup dimension {dimension} has incorrect length")
        for value in np.unique(strata):
            selected = strata == value
            selected_labels = labels[selected]
            if np.unique(selected_labels).size < 2:
                estimate = float("nan")
            else:
                estimate = float(metric(selected_labels, scores[selected]))
            output.append(
                SubgroupEstimate(
                    dimension,
                    str(value),
                    int(selected.sum()),
                    int(selected_labels.sum()),
                    estimate,
                )
            )
    return tuple(output)


def subgroup_gaps(estimates: tuple[SubgroupEstimate, ...]) -> tuple[GapEstimate, ...]:
    dimensions = sorted({item.dimension for item in estimates})
    output = []
    for dimension in dimensions:
        selected = [
            item for item in estimates if item.dimension == dimension and np.isfinite(item.estimate)
        ]
        if len(selected) < 2:
            continue
        minimum = min(selected, key=lambda item: item.estimate)
        maximum = max(selected, key=lambda item: item.estimate)
        output.append(
            GapEstimate(
                dimension,
                minimum.stratum,
                maximum.stratum,
                minimum.estimate,
                maximum.estimate,
                maximum.estimate - minimum.estimate,
            )
        )
    return tuple(output)


def intersectional_groups(dimensions: Mapping[str, np.ndarray]) -> np.ndarray:
    if not dimensions:
        raise ValueError("at least one subgroup dimension is required")
    lengths = {len(values) for values in dimensions.values()}
    if len(lengths) != 1:
        raise ValueError("subgroup dimensions must have equal lengths")
    names = tuple(sorted(dimensions))
    return np.asarray(
        [
            "|".join(f"{name}={dimensions[name][row]}" for name in names)
            for row in range(next(iter(lengths)))
        ]
    )
