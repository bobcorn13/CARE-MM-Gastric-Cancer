from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ImageQuality:
    brightness: float
    contrast: float
    sharpness: float
    saturation: float
    dark_fraction: float
    bright_fraction: float
    accepted: bool


def image_quality(image: torch.Tensor) -> ImageQuality:
    if image.ndim != 3 or image.shape[0] not in (1, 3):
        raise ValueError("image must be channel-first grayscale or RGB")
    values = image.float()
    if values.max() > 1:
        values = values / 255.0
    grayscale = values.mean(dim=0)
    horizontal = grayscale[:, 1:] - grayscale[:, :-1]
    vertical = grayscale[1:, :] - grayscale[:-1, :]
    sharpness = float(horizontal.var() + vertical.var())
    brightness = float(grayscale.mean())
    contrast = float(grayscale.std())
    saturation = float((values.max(dim=0).values - values.min(dim=0).values).mean())
    dark = float((grayscale < 0.05).float().mean())
    bright = float((grayscale > 0.95).float().mean())
    accepted = 0.08 <= brightness <= 0.92 and contrast >= 0.04 and sharpness >= 0.001
    return ImageQuality(brightness, contrast, sharpness, saturation, dark, bright, accepted)


@dataclass(frozen=True)
class SlideQuality:
    tiles: int
    tissue_tiles: int
    tissue_fraction: float
    focus_score: float
    stain_intensity: float
    accepted: bool


def slide_quality(tile_quality: tuple[ImageQuality, ...], tissue_flags: np.ndarray) -> SlideQuality:
    if len(tile_quality) != len(tissue_flags):
        raise ValueError("tile quality and tissue flags must align")
    tissue_flags = np.asarray(tissue_flags, dtype=bool)
    tissue_tiles = int(tissue_flags.sum())
    focus_values = [
        item.sharpness for item, tissue in zip(tile_quality, tissue_flags, strict=True) if tissue
    ]
    stain_values = [
        item.saturation for item, tissue in zip(tile_quality, tissue_flags, strict=True) if tissue
    ]
    focus = float(np.median(focus_values)) if focus_values else 0.0
    stain = float(np.median(stain_values)) if stain_values else 0.0
    fraction = tissue_tiles / max(len(tile_quality), 1)
    accepted = tissue_tiles >= 16 and focus >= 0.001 and stain >= 0.02
    return SlideQuality(len(tile_quality), tissue_tiles, fraction, focus, stain, accepted)


@dataclass(frozen=True)
class SequenceQuality:
    frames: int
    accepted_frames: int
    accepted_fraction: float
    median_brightness: float
    median_sharpness: float
    temporal_difference: float
    accepted: bool


def sequence_quality(images: torch.Tensor) -> SequenceQuality:
    if images.ndim != 4:
        raise ValueError("endoscopy sequence must be frame, channel, height and width")
    qualities = tuple(image_quality(image) for image in images)
    accepted_count = sum(item.accepted for item in qualities)
    grayscale = images.float().mean(dim=1)
    temporal = float((grayscale[1:] - grayscale[:-1]).abs().mean()) if len(images) > 1 else 0.0
    fraction = accepted_count / len(qualities)
    return SequenceQuality(
        len(qualities),
        accepted_count,
        fraction,
        float(np.median([item.brightness for item in qualities])),
        float(np.median([item.sharpness for item in qualities])),
        temporal,
        bool(fraction >= 0.5 and temporal > 0),
    )
