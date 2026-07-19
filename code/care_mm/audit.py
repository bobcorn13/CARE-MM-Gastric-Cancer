import hashlib
import json
import platform
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class ArtifactDigest:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class RuntimeIdentity:
    python: str
    platform: str
    torch: str
    cuda: str | None
    cudnn: int | None
    numpy: str
    device_count: int
    devices: tuple[str, ...]


@dataclass(frozen=True)
class AuditRecord:
    runtime: RuntimeIdentity
    artifacts: tuple[ArtifactDigest, ...]
    configuration_digest: str
    manifest_digest: str
    seed: int
    command: tuple[str, ...]


def file_digest(path: str | Path, block_size: int = 1024 * 1024) -> ArtifactDigest:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return ArtifactDigest(source.name, source.stat().st_size, digest.hexdigest())


def bytes_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def mapping_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return bytes_digest(encoded)


def runtime_identity() -> RuntimeIdentity:
    devices = tuple(torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count()))
    return RuntimeIdentity(
        platform.python_version(),
        platform.platform(),
        torch.__version__,
        torch.version.cuda,
        torch.backends.cudnn.version(),
        np.__version__,
        torch.cuda.device_count(),
        devices,
    )


def discover_artifacts(root: str | Path, suffixes: tuple[str, ...]) -> tuple[ArtifactDigest, ...]:
    directory = Path(root)
    paths = sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in suffixes
    )
    return tuple(file_digest(path) for path in paths)


def create_audit_record(
    artifact_paths: Iterable[str | Path],
    configuration: dict[str, object],
    manifest_digest: str,
    seed: int,
    command: tuple[str, ...],
) -> AuditRecord:
    artifacts = tuple(file_digest(path) for path in artifact_paths)
    return AuditRecord(
        runtime_identity(),
        artifacts,
        mapping_digest(configuration),
        manifest_digest,
        seed,
        command,
    )


def save_audit_record(record: AuditRecord, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(asdict(record), handle, indent=2, sort_keys=True)
    temporary.replace(destination)


def load_audit_record(path: str | Path) -> AuditRecord:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    runtime = RuntimeIdentity(**payload["runtime"])
    artifacts = tuple(ArtifactDigest(**item) for item in payload["artifacts"])
    return AuditRecord(
        runtime,
        artifacts,
        payload["configuration_digest"],
        payload["manifest_digest"],
        int(payload["seed"]),
        tuple(payload["command"]),
    )


def verify_artifacts(record: AuditRecord, root: str | Path) -> dict[str, bool]:
    directory = Path(root)
    output = {}
    for expected in record.artifacts:
        candidate = directory / expected.path
        if not candidate.exists():
            output[expected.path] = False
            continue
        observed = file_digest(candidate)
        output[expected.path] = (
            observed.size == expected.size and observed.sha256 == expected.sha256
        )
    return output


def git_identity(root: str | Path) -> dict[str, str | bool]:
    directory = Path(root)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=directory,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=directory,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"revision": revision, "dirty": bool(status)}


@dataclass(frozen=True)
class DatasetManifestEntry:
    relative_path: str
    size: int
    sha256: str
    modality: str
    cases: int


def dataset_manifest(
    paths: Iterable[str | Path],
    root: str | Path,
    modality: str,
    cases: int,
) -> tuple[DatasetManifestEntry, ...]:
    base = Path(root)
    output = []
    for path in sorted(Path(item) for item in paths):
        digest = file_digest(path)
        output.append(
            DatasetManifestEntry(
                str(path.relative_to(base)),
                digest.size,
                digest.sha256,
                modality,
                cases,
            )
        )
    return tuple(output)


def validate_dataset_manifest(
    entries: tuple[DatasetManifestEntry, ...],
    root: str | Path,
) -> tuple[str, ...]:
    base = Path(root)
    failures = []
    for entry in entries:
        path = base / entry.relative_path
        if not path.exists():
            failures.append(f"missing:{entry.relative_path}")
            continue
        digest = file_digest(path)
        if digest.size != entry.size:
            failures.append(f"size:{entry.relative_path}")
        elif digest.sha256 != entry.sha256:
            failures.append(f"sha256:{entry.relative_path}")
    return tuple(failures)


def environment_variables(names: tuple[str, ...]) -> dict[str, str | None]:
    import os

    return {name: os.environ.get(name) for name in names}


def parameter_summary(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    buffers = sum(buffer.numel() for buffer in model.buffers())
    return {"total": total, "trainable": trainable, "frozen": total - trainable, "buffers": buffers}


def tensor_summary(value: torch.Tensor) -> dict[str, object]:
    finite = torch.isfinite(value)
    selected = value[finite]
    return {
        "shape": tuple(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "finite_fraction": float(finite.float().mean()),
        "minimum": float(selected.min()) if selected.numel() else None,
        "maximum": float(selected.max()) if selected.numel() else None,
        "mean": float(selected.float().mean()) if selected.numel() else None,
        "standard_deviation": float(selected.float().std()) if selected.numel() > 1 else None,
    }
