from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarginalValue:
    setting: str
    modality: str
    baseline: float
    augmented: float
    reference: float
    gap_closed: float


def gap_closed(baseline: float, augmented: float, reference: float) -> float:
    denominator = reference - baseline
    if denominator == 0:
        return float("nan")
    return (augmented - baseline) / denominator


def marginal_values(
    settings: tuple[str, ...],
    modalities: tuple[str, ...],
    baseline: np.ndarray,
    augmented: np.ndarray,
    reference: np.ndarray,
) -> tuple[MarginalValue, ...]:
    expected = (len(settings), len(modalities))
    if baseline.shape != expected or augmented.shape != expected or reference.shape != expected:
        raise ValueError("marginal value matrices have incorrect shape")
    output = []
    for row, setting in enumerate(settings):
        for column, modality in enumerate(modalities):
            output.append(
                MarginalValue(
                    setting,
                    modality,
                    float(baseline[row, column]),
                    float(augmented[row, column]),
                    float(reference[row, column]),
                    gap_closed(
                        float(baseline[row, column]),
                        float(augmented[row, column]),
                        float(reference[row, column]),
                    ),
                )
            )
    return tuple(output)


def inversion_statistic(values: tuple[MarginalValue, ...], scarce: str, rich: str) -> float:
    scarce_values = [item.gap_closed for item in values if item.setting == scarce]
    rich_values = [item.gap_closed for item in values if item.setting == rich]
    if not scarce_values or not rich_values:
        raise ValueError("both care settings must be represented")
    return float(np.mean(scarce_values) - np.mean(rich_values))


def permutation_p_value(
    first: np.ndarray,
    second: np.ndarray,
    permutations: int = 10000,
    seed: int = 2026,
) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("permutation samples must be vectors")
    observed = abs(first.mean() - second.mean())
    combined = np.concatenate((first, second))
    generator = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        shuffled = generator.permutation(combined)
        difference = abs(shuffled[: len(first)].mean() - shuffled[len(first) :].mean())
        exceed += int(difference >= observed)
    return (exceed + 1) / (permutations + 1)
