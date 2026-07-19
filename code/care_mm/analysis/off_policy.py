from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolicyEstimate:
    name: str
    estimate: float
    effective_sample_size: float
    minimum_weight: float
    maximum_weight: float


def validate_policy_inputs(
    outcomes: np.ndarray,
    observed_actions: np.ndarray,
    target_probabilities: np.ndarray,
    behavior_probabilities: np.ndarray,
) -> None:
    if outcomes.ndim != 1 or observed_actions.shape != outcomes.shape:
        raise ValueError("outcomes and observed actions must be aligned vectors")
    if target_probabilities.shape != behavior_probabilities.shape:
        raise ValueError("target and behavior policy matrices must align")
    if target_probabilities.shape[0] != outcomes.size:
        raise ValueError("policy rows must match outcomes")
    if np.any(behavior_probabilities <= 0):
        raise ValueError("behavior probabilities must be strictly positive")
    if not np.allclose(target_probabilities.sum(axis=1), 1.0):
        raise ValueError("target probabilities must sum to one")
    if not np.allclose(behavior_probabilities.sum(axis=1), 1.0):
        raise ValueError("behavior probabilities must sum to one")


def importance_weights(
    observed_actions: np.ndarray,
    target_probabilities: np.ndarray,
    behavior_probabilities: np.ndarray,
    clip: float | None = None,
) -> np.ndarray:
    rows = np.arange(len(observed_actions))
    numerator = target_probabilities[rows, observed_actions]
    denominator = behavior_probabilities[rows, observed_actions]
    weights = numerator / denominator
    return np.minimum(weights, clip) if clip is not None else weights


def effective_sample_size(weights: np.ndarray) -> float:
    denominator = np.square(weights).sum()
    if denominator == 0:
        return 0.0
    return float(np.square(weights.sum()) / denominator)


def inverse_probability_weighting(
    outcomes: np.ndarray,
    observed_actions: np.ndarray,
    target_probabilities: np.ndarray,
    behavior_probabilities: np.ndarray,
    clip: float | None = None,
) -> PolicyEstimate:
    validate_policy_inputs(outcomes, observed_actions, target_probabilities, behavior_probabilities)
    weights = importance_weights(
        observed_actions,
        target_probabilities,
        behavior_probabilities,
        clip,
    )
    estimate = float(np.mean(weights * outcomes))
    return PolicyEstimate(
        "inverse_probability_weighting",
        estimate,
        effective_sample_size(weights),
        float(weights.min()),
        float(weights.max()),
    )


def self_normalized_importance_sampling(
    outcomes: np.ndarray,
    observed_actions: np.ndarray,
    target_probabilities: np.ndarray,
    behavior_probabilities: np.ndarray,
    clip: float | None = None,
) -> PolicyEstimate:
    validate_policy_inputs(outcomes, observed_actions, target_probabilities, behavior_probabilities)
    weights = importance_weights(
        observed_actions,
        target_probabilities,
        behavior_probabilities,
        clip,
    )
    denominator = weights.sum()
    estimate = float(np.sum(weights * outcomes) / denominator) if denominator else float("nan")
    return PolicyEstimate(
        "self_normalized_importance_sampling",
        estimate,
        effective_sample_size(weights),
        float(weights.min()),
        float(weights.max()),
    )


def direct_method(
    target_probabilities: np.ndarray,
    outcome_predictions: np.ndarray,
) -> PolicyEstimate:
    if target_probabilities.shape != outcome_predictions.shape:
        raise ValueError("target policy and outcome predictions must align")
    values = np.sum(target_probabilities * outcome_predictions, axis=1)
    return PolicyEstimate("direct_method", float(values.mean()), float(len(values)), 1.0, 1.0)


def doubly_robust(
    outcomes: np.ndarray,
    observed_actions: np.ndarray,
    target_probabilities: np.ndarray,
    behavior_probabilities: np.ndarray,
    outcome_predictions: np.ndarray,
    clip: float | None = None,
) -> PolicyEstimate:
    validate_policy_inputs(outcomes, observed_actions, target_probabilities, behavior_probabilities)
    if outcome_predictions.shape != target_probabilities.shape:
        raise ValueError("outcome predictions and target probabilities must align")
    weights = importance_weights(
        observed_actions,
        target_probabilities,
        behavior_probabilities,
        clip,
    )
    rows = np.arange(len(outcomes))
    direct = np.sum(target_probabilities * outcome_predictions, axis=1)
    residual = outcomes - outcome_predictions[rows, observed_actions]
    estimate = float(np.mean(direct + weights * residual))
    return PolicyEstimate(
        "doubly_robust",
        estimate,
        effective_sample_size(weights),
        float(weights.min()),
        float(weights.max()),
    )


def bootstrap_policy_estimate(
    estimator: callable,
    size: int,
    resamples: int = 10000,
    seed: int = 2026,
) -> tuple[float, float, float]:
    generator = np.random.default_rng(seed)
    values = []
    full = np.arange(size)
    estimate = float(estimator(full))
    for _ in range(resamples):
        indices = generator.integers(0, size, size)
        value = float(estimator(indices))
        if np.isfinite(value):
            values.append(value)
    lower, upper = np.quantile(values, [0.025, 0.975])
    return estimate, float(lower), float(upper)
