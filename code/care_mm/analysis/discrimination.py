from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn import metrics


@dataclass(frozen=True)
class BinaryMetrics:
    auroc: float
    average_precision: float
    sensitivity: float
    specificity: float
    positive_predictive_value: float
    negative_predictive_value: float
    accuracy: float
    balanced_accuracy: float
    f1: float
    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


def optimal_youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    false_positive, true_positive, thresholds = metrics.roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    index = np.argmax((true_positive - false_positive)[finite])
    return float(thresholds[finite][index])


def fixed_sensitivity_threshold(labels: np.ndarray, scores: np.ndarray, target: float) -> float:
    if not 0 < target <= 1:
        raise ValueError("target sensitivity must lie in (0, 1]")
    false_positive, true_positive, thresholds = metrics.roc_curve(labels, scores)
    candidates = np.where(true_positive >= target)[0]
    if candidates.size == 0:
        return float("inf")
    best = candidates[np.argmin(false_positive[candidates])]
    return float(thresholds[best])


def binary_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float | None = None
) -> BinaryMetrics:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("labels and scores must be aligned vectors")
    if np.unique(labels).size != 2:
        raise ValueError("binary metrics require both classes")
    selected_threshold = (
        optimal_youden_threshold(labels, scores) if threshold is None else threshold
    )
    predictions = scores >= selected_threshold
    true_negative, false_positive, false_negative, true_positive = metrics.confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    sensitivity = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    ppv = true_positive / max(true_positive + false_positive, 1)
    npv = true_negative / max(true_negative + false_negative, 1)
    accuracy = (true_positive + true_negative) / labels.size
    balanced = (sensitivity + specificity) / 2
    f1 = 2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1)
    return BinaryMetrics(
        auroc=float(metrics.roc_auc_score(labels, scores)),
        average_precision=float(metrics.average_precision_score(labels, scores)),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        positive_predictive_value=float(ppv),
        negative_predictive_value=float(npv),
        accuracy=float(accuracy),
        balanced_accuracy=float(balanced),
        f1=float(f1),
        threshold=float(selected_threshold),
        true_positive=int(true_positive),
        false_positive=int(false_positive),
        true_negative=int(true_negative),
        false_negative=int(false_negative),
    )


def _midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.zeros(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1) + 1
        start = end
    output = np.empty(len(values), dtype=np.float64)
    output[order] = ranks
    return output


def _fast_delong(predictions: np.ndarray, positive_count: int) -> tuple[np.ndarray, np.ndarray]:
    classifiers, examples = predictions.shape
    negative_count = examples - positive_count
    positive = predictions[:, :positive_count]
    negative = predictions[:, positive_count:]
    positive_ranks = np.empty((classifiers, positive_count))
    negative_ranks = np.empty((classifiers, negative_count))
    combined_ranks = np.empty((classifiers, examples))
    for classifier in range(classifiers):
        positive_ranks[classifier] = _midrank(positive[classifier])
        negative_ranks[classifier] = _midrank(negative[classifier])
        combined_ranks[classifier] = _midrank(predictions[classifier])
    aucs = combined_ranks[:, :positive_count].sum(axis=1) / positive_count / negative_count - (
        positive_count + 1.0
    ) / (2.0 * negative_count)
    positive_components = (combined_ranks[:, :positive_count] - positive_ranks) / negative_count
    negative_components = (
        1.0 - (combined_ranks[:, positive_count:] - negative_ranks) / positive_count
    )
    positive_covariance = np.atleast_2d(np.cov(positive_components))
    negative_covariance = np.atleast_2d(np.cov(negative_components))
    covariance = positive_covariance / positive_count + negative_covariance / negative_count
    return aucs, covariance


@dataclass(frozen=True)
class DeLongResult:
    auc_first: float
    auc_second: float
    difference: float
    standard_error: float
    z_score: float
    p_value: float


def delong_test(labels: np.ndarray, first: np.ndarray, second: np.ndarray) -> DeLongResult:
    labels = np.asarray(labels, dtype=np.int64)
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if labels.shape != first.shape or labels.shape != second.shape:
        raise ValueError("DeLong inputs must be aligned")
    order = np.argsort(-labels)
    positive_count = int(labels.sum())
    predictions = np.vstack((first[order], second[order]))
    aucs, covariance = _fast_delong(predictions, positive_count)
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast.T)
    standard_error = float(np.sqrt(max(variance, np.finfo(float).eps)))
    difference = float(aucs[0] - aucs[1])
    z_score = difference / standard_error
    p_value = float(2 * stats.norm.sf(abs(z_score)))
    return DeLongResult(
        float(aucs[0]), float(aucs[1]), difference, standard_error, z_score, p_value
    )


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a vector within [0, 1]")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = (count - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted


def concordance(labels: np.ndarray, actions: np.ndarray) -> float:
    labels = np.asarray(labels)
    actions = np.asarray(actions)
    if labels.shape != actions.shape:
        raise ValueError("labels and actions must align")
    return float(np.mean(labels == actions))
