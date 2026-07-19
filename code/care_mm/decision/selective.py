from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SelectivePoint:
    coverage: float
    accuracy: float
    risk: float
    threshold: float
    retained: int


def selective_curve(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    points: int = 101,
) -> tuple[SelectivePoint, ...]:
    confidence, predictions = probabilities.max(dim=1)
    correctness = predictions.eq(labels).float()
    thresholds = torch.linspace(0.0, 1.0, points, device=probabilities.device)
    output = []
    for threshold in thresholds:
        retained = confidence >= threshold
        count = int(retained.sum())
        if count == 0:
            continue
        accuracy = float(correctness[retained].mean())
        output.append(
            SelectivePoint(
                coverage=count / labels.numel(),
                accuracy=accuracy,
                risk=1.0 - accuracy,
                threshold=float(threshold),
                retained=count,
            )
        )
    return tuple(output)


def area_under_risk_coverage(points: tuple[SelectivePoint, ...]) -> float:
    if len(points) < 2:
        return float("nan")
    ordered = sorted(points, key=lambda item: item.coverage)
    area = 0.0
    for left, right in zip(ordered[:-1], ordered[1:], strict=True):
        area += (right.coverage - left.coverage) * (right.risk + left.risk) / 2
    return area


def deferral_enrichment(
    deferred: torch.Tensor,
    difficult: torch.Tensor,
) -> float:
    if deferred.dtype != torch.bool or difficult.dtype != torch.bool:
        raise ValueError("deferral and difficulty flags must be boolean")
    if deferred.shape != difficult.shape:
        raise ValueError("deferral and difficulty flags must align")
    baseline = difficult.float().mean()
    if deferred.sum() == 0 or baseline == 0:
        return float("nan")
    return float(difficult[deferred].float().mean() / baseline)
