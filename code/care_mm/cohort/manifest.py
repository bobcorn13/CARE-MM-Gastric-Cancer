import csv
import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from pathlib import Path

from care_mm.types import CaseRecord


class ManifestError(ValueError):
    pass


class CohortManifest(Sequence[CaseRecord]):
    required = frozenset({"case_id", "site", "region", "setting", "label", "triage"})

    def __init__(self, records: Sequence[CaseRecord]) -> None:
        self._records = tuple(records)
        self._validate()

    @classmethod
    def from_csv(cls, path: str | Path, data_root: str | Path | None = None) -> "CohortManifest":
        source = Path(path)
        root = Path(data_root) if data_root is not None else source.parent
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            missing = cls.required - fields
            if missing:
                raise ManifestError(f"missing fields: {sorted(missing)}")
            records = [cls._parse_row(row, root) for row in reader]
        return cls(records)

    @staticmethod
    def _optional_path(value: str | None, root: Path) -> Path | None:
        if value is None or value.strip() == "":
            return None
        path = Path(value)
        return path if path.is_absolute() else root / path

    @classmethod
    def _parse_row(cls, row: dict[str, str], root: Path) -> CaseRecord:
        return CaseRecord(
            case_id=row["case_id"],
            site=row["site"],
            region=row["region"],
            setting=row["setting"],
            label=int(row["label"]),
            triage=int(row["triage"]),
            endoscopy_path=cls._optional_path(row.get("endoscopy_path"), root),
            pathology_path=cls._optional_path(row.get("pathology_path"), root),
            ehr_path=cls._optional_path(row.get("ehr_path"), root),
        )

    def _validate(self) -> None:
        identifiers = [record.case_id for record in self._records]
        if len(identifiers) != len(set(identifiers)):
            raise ManifestError("case identifiers must be unique")
        for record in self._records:
            if record.label not in (0, 1):
                raise ManifestError(f"invalid diagnosis label for {record.case_id}")
            if record.triage not in (0, 1, 2):
                raise ManifestError(f"invalid triage label for {record.case_id}")
            if not any(record.mask):
                raise ManifestError(f"case has no available modality: {record.case_id}")

    def patient_split(
        self, fractions: tuple[float, float, float], seed: int
    ) -> tuple["CohortManifest", ...]:
        if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-8:
            raise ManifestError("split fractions must contain three values summing to one")
        ranked = sorted(
            self._records,
            key=lambda item: hashlib.sha256(f"{seed}:{item.case_id}".encode()).digest(),
        )
        first = int(len(ranked) * fractions[0])
        second = first + int(len(ranked) * fractions[1])
        return (
            CohortManifest(ranked[:first]),
            CohortManifest(ranked[first:second]),
            CohortManifest(ranked[second:]),
        )

    def subset(
        self, sites: set[str] | None = None, settings: set[str] | None = None
    ) -> "CohortManifest":
        selected = [
            record
            for record in self._records
            if (sites is None or record.site in sites)
            and (settings is None or record.setting in settings)
        ]
        return CohortManifest(selected)

    def settings(self) -> tuple[str, ...]:
        return tuple(sorted({record.setting for record in self._records}))

    def sites(self) -> tuple[str, ...]:
        return tuple(sorted({record.site for record in self._records}))

    def prevalence(self) -> float:
        if not self._records:
            return float("nan")
        return sum(record.label for record in self._records) / len(self._records)

    def modality_counts(self) -> dict[tuple[bool, bool, bool], int]:
        counts: dict[tuple[bool, bool, bool], int] = {}
        for record in self._records:
            counts[record.mask] = counts.get(record.mask, 0) + 1
        return counts

    def digest(self) -> str:
        serializable = []
        for record in self._records:
            item = asdict(record)
            for field in ("endoscopy_path", "pathology_path", "ehr_path"):
                value = item[field]
                item[field] = None if value is None else str(value)
            serializable.append(item)
        payload = json.dumps(serializable, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> CaseRecord:
        return self._records[index]

    def __iter__(self) -> Iterator[CaseRecord]:
        return iter(self._records)
