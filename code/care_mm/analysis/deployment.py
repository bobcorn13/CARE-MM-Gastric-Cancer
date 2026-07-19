from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class DeploymentEvent:
    case_id: str
    received_at: datetime
    completed_at: datetime
    viewed: bool
    recommendation: int
    observed_action: int
    abstained: bool
    mask_code: int

    @property
    def turnaround_seconds(self) -> float:
        return (self.completed_at - self.received_at).total_seconds()


@dataclass(frozen=True)
class DeploymentSummary:
    cases: int
    median_turnaround_seconds: float
    percentile_90_turnaround_seconds: float
    view_rate: float
    concordance: float
    abstention_rate: float


def summarize_deployment(events: tuple[DeploymentEvent, ...]) -> DeploymentSummary:
    if not events:
        raise ValueError("deployment events cannot be empty")
    turnaround = np.asarray([item.turnaround_seconds for item in events])
    viewed = np.asarray([item.viewed for item in events])
    recommendation = np.asarray([item.recommendation for item in events])
    observed = np.asarray([item.observed_action for item in events])
    abstained = np.asarray([item.abstained for item in events])
    return DeploymentSummary(
        cases=len(events),
        median_turnaround_seconds=float(np.median(turnaround)),
        percentile_90_turnaround_seconds=float(np.quantile(turnaround, 0.9)),
        view_rate=float(viewed.mean()),
        concordance=float((recommendation == observed).mean()),
        abstention_rate=float(abstained.mean()),
    )


@dataclass(frozen=True)
class CoverageTripwire:
    mask_code: int
    window_count: int
    errors: int
    observed_miscoverage: float
    standard_error: float
    threshold: float
    triggered: bool


def coverage_tripwire(
    covered: np.ndarray,
    mask_codes: np.ndarray,
    target_miscoverage: float = 0.1,
    standard_errors: float = 2.0,
) -> tuple[CoverageTripwire, ...]:
    covered = np.asarray(covered, dtype=bool)
    mask_codes = np.asarray(mask_codes, dtype=np.int64)
    if covered.shape != mask_codes.shape:
        raise ValueError("coverage and mask codes must align")
    output = []
    for code in np.unique(mask_codes):
        selected = mask_codes == code
        count = int(selected.sum())
        errors = int((~covered[selected]).sum())
        observed = errors / count
        error = np.sqrt(target_miscoverage * (1 - target_miscoverage) / count)
        threshold = target_miscoverage + standard_errors * error
        output.append(
            CoverageTripwire(
                int(code),
                count,
                errors,
                float(observed),
                float(error),
                float(threshold),
                bool(observed > threshold),
            )
        )
    return tuple(output)


@dataclass(frozen=True)
class VendorShift:
    reference_vendor: str
    comparison_vendor: str
    reference_metric: float
    comparison_metric: float
    absolute_drop: float
    relative_drop: float


def vendor_shift(
    vendors: np.ndarray,
    values: np.ndarray,
    reference_vendor: str,
) -> tuple[VendorShift, ...]:
    vendors = np.asarray(vendors)
    values = np.asarray(values, dtype=np.float64)
    if vendors.shape != values.shape:
        raise ValueError("vendor and metric vectors must align")
    reference = float(values[vendors == reference_vendor].mean())
    output = []
    for vendor in np.unique(vendors):
        if vendor == reference_vendor:
            continue
        comparison = float(values[vendors == vendor].mean())
        drop = reference - comparison
        output.append(
            VendorShift(
                reference_vendor,
                str(vendor),
                reference,
                comparison,
                drop,
                drop / reference if reference else float("nan"),
            )
        )
    return tuple(output)
