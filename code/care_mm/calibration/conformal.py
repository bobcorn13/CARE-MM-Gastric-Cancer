from collections.abc import Iterable
from dataclasses import dataclass

import torch


def mask_code(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 2 or mask.shape[1] != 3:
        raise ValueError("mask must have shape batch by three")
    powers = torch.tensor([1, 2, 4], device=mask.device)
    return (mask.long() * powers).sum(dim=1)


def finite_sample_quantile(scores: torch.Tensor, miscoverage: float) -> torch.Tensor:
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("conformal scores must be a non-empty vector")
    if not 0 < miscoverage < 1:
        raise ValueError("miscoverage must lie between zero and one")
    rank = int(torch.ceil(torch.tensor((scores.numel() + 1) * (1 - miscoverage))).item())
    rank = min(max(rank, 1), scores.numel())
    return torch.kthvalue(scores, rank).values


@dataclass(frozen=True)
class CoverageReport:
    mask: int
    count: int
    covered: int
    coverage: float
    average_set_size: float
    singleton_rate: float


class MaskConditionalConformal:
    def __init__(self, miscoverage: float = 0.1) -> None:
        if not 0 < miscoverage < 1:
            raise ValueError("miscoverage must lie between zero and one")
        self.miscoverage = miscoverage
        self.thresholds: dict[int, float] = {}
        self.counts: dict[int, int] = {}

    def fit(self, probabilities: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor) -> None:
        if probabilities.ndim != 2:
            raise ValueError("probabilities must be a matrix")
        if labels.shape != (probabilities.shape[0],):
            raise ValueError("labels shape mismatch")
        codes = mask_code(masks)
        scores = 1.0 - probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
        self.thresholds.clear()
        self.counts.clear()
        for code in codes.unique().tolist():
            selected = scores[codes == code]
            self.thresholds[int(code)] = float(finite_sample_quantile(selected, self.miscoverage))
            self.counts[int(code)] = int(selected.numel())

    def prediction_sets(self, probabilities: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if not self.thresholds:
            raise RuntimeError("conformal calibrator has not been fitted")
        codes = mask_code(masks)
        thresholds = probabilities.new_empty(probabilities.shape[0])
        for row, code in enumerate(codes.tolist()):
            if code not in self.thresholds:
                raise KeyError(f"unseen availability mask {code}")
            thresholds[row] = self.thresholds[code]
        return (1.0 - probabilities) <= thresholds.unsqueeze(1)

    def sets_as_tuples(
        self, probabilities: torch.Tensor, masks: torch.Tensor
    ) -> tuple[tuple[int, ...], ...]:
        indicators = self.prediction_sets(probabilities, masks)
        return tuple(tuple(torch.where(row)[0].tolist()) for row in indicators)

    def coverage(
        self, probabilities: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor
    ) -> tuple[CoverageReport, ...]:
        sets = self.prediction_sets(probabilities, masks)
        codes = mask_code(masks)
        covered = sets.gather(1, labels.unsqueeze(1)).squeeze(1)
        reports = []
        for code in codes.unique().tolist():
            selected = codes == code
            subset = sets[selected]
            count = int(selected.sum())
            hit = int(covered[selected].sum())
            sizes = subset.sum(dim=1).float()
            reports.append(
                CoverageReport(
                    mask=int(code),
                    count=count,
                    covered=hit,
                    coverage=hit / count,
                    average_set_size=float(sizes.mean()),
                    singleton_rate=float((sizes == 1).float().mean()),
                )
            )
        return tuple(reports)

    def state_dict(self) -> dict[str, object]:
        return {
            "miscoverage": self.miscoverage,
            "thresholds": self.thresholds,
            "counts": self.counts,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.miscoverage = float(state["miscoverage"])
        thresholds = state["thresholds"]
        counts = state["counts"]
        if not isinstance(thresholds, dict) or not isinstance(counts, dict):
            raise TypeError("invalid conformal state")
        self.thresholds = {int(key): float(value) for key, value in thresholds.items()}
        self.counts = {int(key): int(value) for key, value in counts.items()}

    def supported_masks(self) -> tuple[int, ...]:
        return tuple(sorted(self.thresholds))

    def merge(self, calibrators: Iterable["MaskConditionalConformal"]) -> None:
        for calibrator in calibrators:
            if calibrator.miscoverage != self.miscoverage:
                raise ValueError("cannot merge calibrators with different miscoverage")
            for code, threshold in calibrator.thresholds.items():
                if code in self.thresholds:
                    raise ValueError(f"duplicate mask threshold {code}")
                self.thresholds[code] = threshold
                self.counts[code] = calibrator.counts[code]
