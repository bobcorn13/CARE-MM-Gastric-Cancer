import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from care_mm.analysis.discrimination import binary_metrics, delong_test, holm_adjust
from care_mm.analysis.reporting import ResultBook, comparison_row
from care_mm.analysis.resampling import patient_bootstrap
from care_mm.analysis.subgroups import subgroup_estimates, subgroup_gaps
from care_mm.calibration.metrics import brier_score, expected_calibration_error


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="care-mm-evaluate")
    value.add_argument("--predictions", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--resamples", type=int, default=10000)
    value.add_argument("--seed", type=int, default=2026)
    return value


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"case_id", "label", "score", "site", "setting", "mask"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"prediction archive is missing arrays: {sorted(missing)}")
        return {name: payload[name] for name in payload.files}


def evaluate(payload: dict[str, np.ndarray], resamples: int, seed: int) -> ResultBook:
    labels = payload["label"].astype(np.int64)
    scores = payload["score"].astype(np.float64)
    cases = payload["case_id"]
    metrics = binary_metrics(labels, scores)
    interval = patient_bootstrap(
        cases,
        lambda indices: float(roc_auc_score(labels[indices], scores[indices])),
        resamples,
        seed=seed,
    )
    probabilities = torch.from_numpy(np.stack((1 - scores, scores), axis=1)).float()
    label_tensor = torch.from_numpy(labels)
    ece, bins = expected_calibration_error(probabilities, label_tensor)
    brier = brier_score(probabilities, label_tensor)
    book = ResultBook()
    book.add_row(
        "primary_comparison",
        comparison_row(
            "CARE-MM",
            "endoscopy+WSI+EHR",
            metrics.auroc,
            interval.lower,
            interval.upper,
            metrics.sensitivity,
            metrics.specificity,
        ),
    )
    book.set_metadata("ece", ece)
    book.set_metadata("brier", brier)
    book.set_metadata("bootstrap_resamples", resamples)
    for item in bins:
        book.add_row(
            "calibration_bins",
            {
                "lower": item.lower,
                "upper": item.upper,
                "count": item.count,
                "confidence": item.confidence,
                "accuracy": item.accuracy,
                "gap": item.gap,
            },
        )
    dimensions = {"site": payload["site"], "setting": payload["setting"], "mask": payload["mask"]}
    estimates = subgroup_estimates(labels, scores, dimensions, roc_auc_score)
    for item in estimates:
        book.add_row("subgroups", vars(item))
    for item in subgroup_gaps(estimates):
        book.add_row("subgroup_gaps", vars(item))
    if "baseline_score" in payload:
        comparison = delong_test(labels, scores, payload["baseline_score"])
        adjusted = holm_adjust(np.asarray([comparison.p_value]))
        book.set_metadata("delong", vars(comparison))
        book.set_metadata("holm_adjusted_p", float(adjusted[0]))
    return book


def main() -> None:
    arguments = parser().parse_args()
    book = evaluate(load_predictions(arguments.predictions), arguments.resamples, arguments.seed)
    book.save(arguments.output)


if __name__ == "__main__":
    main()
