import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class AvailabilityProfile:
    setting: str
    mask: tuple[bool, bool, bool]
    probability: float


class AvailabilityCurriculum:
    def __init__(
        self, profiles: Sequence[AvailabilityProfile], setting_weights: Mapping[str, float]
    ) -> None:
        self._profiles = tuple(profiles)
        self._setting_weights = dict(setting_weights)
        self._settings = tuple(sorted(self._setting_weights))
        self._validate()

    @classmethod
    def from_csv(cls, path: str | Path) -> "AvailabilityCurriculum":
        profiles: list[AvailabilityProfile] = []
        setting_weights: dict[str, float] = {}
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                setting = row["setting"]
                mask = (
                    bool(int(row["endoscopy"])),
                    bool(int(row["pathology"])),
                    bool(int(row["ehr"])),
                )
                profiles.append(AvailabilityProfile(setting, mask, float(row["probability"])))
                setting_weights[setting] = float(row["setting_weight"])
        return cls(profiles, setting_weights)

    @classmethod
    def estimate(
        cls, rows: Iterable[tuple[str, tuple[bool, bool, bool]]]
    ) -> "AvailabilityCurriculum":
        mask_counts: dict[str, dict[tuple[bool, bool, bool], int]] = {}
        setting_counts: dict[str, int] = {}
        total = 0
        for setting, mask in rows:
            total += 1
            setting_counts[setting] = setting_counts.get(setting, 0) + 1
            bucket = mask_counts.setdefault(setting, {})
            bucket[mask] = bucket.get(mask, 0) + 1
        profiles = []
        for setting, masks in mask_counts.items():
            denominator = setting_counts[setting]
            profiles.extend(
                AvailabilityProfile(setting, mask, count / denominator)
                for mask, count in masks.items()
            )
        weights = {setting: count / total for setting, count in setting_counts.items()}
        return cls(profiles, weights)

    def _validate(self) -> None:
        if not self._profiles:
            raise ValueError("availability profiles cannot be empty")
        if abs(sum(self._setting_weights.values()) - 1.0) > 1e-6:
            raise ValueError("setting weights must sum to one")
        for setting in self._settings:
            total = sum(item.probability for item in self._profiles if item.setting == setting)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"mask probabilities for {setting} must sum to one")
        if any(not any(item.mask) for item in self._profiles):
            raise ValueError("empty modality masks are invalid")

    def sample(
        self, count: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        setting_probabilities = torch.tensor(
            [self._setting_weights[item] for item in self._settings]
        )
        setting_indices = torch.multinomial(
            setting_probabilities, count, replacement=True, generator=generator
        )
        masks = torch.zeros((count, 3), dtype=torch.bool)
        chosen_settings: list[str] = []
        for row, setting_index in enumerate(setting_indices.tolist()):
            setting = self._settings[setting_index]
            candidates = [item for item in self._profiles if item.setting == setting]
            probabilities = torch.tensor([item.probability for item in candidates])
            selected = torch.multinomial(probabilities, 1, generator=generator).item()
            masks[row] = torch.tensor(candidates[selected].mask)
            chosen_settings.append(setting)
        return masks, tuple(chosen_settings)

    def probability(self, setting: str, mask: tuple[bool, bool, bool]) -> float:
        for item in self._profiles:
            if item.setting == setting and item.mask == mask:
                return self._setting_weights[setting] * item.probability
        return 0.0

    def profiles(self) -> tuple[AvailabilityProfile, ...]:
        return self._profiles
