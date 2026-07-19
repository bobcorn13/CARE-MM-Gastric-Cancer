import argparse
import csv
from pathlib import Path

from care_mm.cohort.availability import AvailabilityCurriculum
from care_mm.cohort.manifest import CohortManifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="care-mm-prepare")
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--availability-output", type=Path, required=True)
    value.add_argument("--data-root", type=Path)
    return value


def write_availability(curriculum: AvailabilityCurriculum, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "setting",
                "endoscopy",
                "pathology",
                "ehr",
                "probability",
                "setting_weight",
            ),
        )
        writer.writeheader()
        for item in curriculum.profiles():
            joint = sum(
                curriculum.probability(item.setting, other.mask)
                for other in curriculum.profiles()
                if other.setting == item.setting
            )
            writer.writerow(
                {
                    "setting": item.setting,
                    "endoscopy": int(item.mask[0]),
                    "pathology": int(item.mask[1]),
                    "ehr": int(item.mask[2]),
                    "probability": item.probability,
                    "setting_weight": joint,
                }
            )


def main() -> None:
    arguments = parser().parse_args()
    manifest = CohortManifest.from_csv(arguments.manifest, arguments.data_root)
    curriculum = AvailabilityCurriculum.estimate((item.setting, item.mask) for item in manifest)
    write_availability(curriculum, arguments.availability_output)


if __name__ == "__main__":
    main()
