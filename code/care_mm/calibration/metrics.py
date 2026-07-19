from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    confidence: float
    accuracy: float
    gap: float


def expected_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    bins: int = 15,
) -> tuple[float, tuple[CalibrationBin, ...]]:
    if bins < 2:
        raise ValueError("at least two calibration bins are required")
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(labels).float()
    edges = torch.linspace(0.0, 1.0, bins + 1, device=probabilities.device)
    error = probabilities.new_tensor(0.0)
    reports = []
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        selected = (confidence > lower) & (confidence <= upper)
        count = int(selected.sum())
        if count == 0:
            reports.append(CalibrationBin(float(lower), float(upper), 0, 0.0, 0.0, 0.0))
            continue
        bin_confidence = confidence[selected].mean()
        bin_accuracy = correct[selected].mean()
        gap = torch.abs(bin_confidence - bin_accuracy)
        error = error + gap * selected.float().mean()
        reports.append(
            CalibrationBin(
                float(lower),
                float(upper),
                count,
                float(bin_confidence),
                float(bin_accuracy),
                float(gap),
            )
        )
    return float(error), tuple(reports)


def brier_score(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    one_hot = torch.zeros_like(probabilities).scatter_(1, labels.unsqueeze(1), 1.0)
    return float(torch.mean(torch.sum((probabilities - one_hot).square(), dim=1)))


def negative_log_likelihood(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    selected = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
    return float(-selected.log().mean())


def classwise_calibration_error(
    probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 15
) -> float:
    errors = []
    for class_index in range(probabilities.shape[1]):
        targets = labels.eq(class_index).long()
        two_class = torch.stack(
            (1 - probabilities[:, class_index], probabilities[:, class_index]), dim=1
        )
        error, _ = expected_calibration_error(two_class, targets, bins)
        errors.append(error)
    return sum(errors) / len(errors)
