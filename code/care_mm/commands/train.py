import argparse
import logging
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from care_mm.cohort.availability import AvailabilityCurriculum
from care_mm.cohort.dataset import FeatureBatch, MultimodalFeatureDataset, collate_features
from care_mm.cohort.manifest import CohortManifest
from care_mm.configuration import ExperimentConfiguration, load_configuration
from care_mm.fusion.system import CareMM, CareMMInputs
from care_mm.learning.checkpoint import save_training_state, set_seed
from care_mm.learning.distributed import (
    finalize_distributed,
    initialize_distributed,
    wrap_distributed,
)
from care_mm.learning.engine import TrainingEngine


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="care-mm-train")
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--endoscopy-width", type=int, required=True)
    value.add_argument("--pathology-width", type=int, required=True)
    value.add_argument("--ehr-numerical", type=int, required=True)
    value.add_argument("--ehr-cardinalities", type=int, nargs="*", default=[])
    value.add_argument("--workers", type=int, default=4)
    return value


def build_model(configuration: ExperimentConfiguration, arguments: argparse.Namespace) -> CareMM:
    model = CareMM(
        arguments.endoscopy_width,
        arguments.pathology_width,
        arguments.ehr_numerical,
        tuple(arguments.ehr_cardinalities),
        width=configuration.model.width,
        depth=configuration.model.depth,
        heads=configuration.model.heads,
        classes=configuration.model.classes,
        dropout=configuration.model.dropout,
    )
    model.freeze_encoders()
    return model


def loss_function(model: torch.nn.Module, batch: FeatureBatch) -> torch.Tensor:
    output = model(
        CareMMInputs(
            batch.endoscopy,
            batch.endoscopy_padding,
            batch.pathology,
            batch.pathology_padding,
            batch.ehr_numerical,
            batch.ehr_categorical,
            batch.ehr_missing,
            batch.available,
        )
    )
    weights = output.logits.new_tensor([1.0, 10.0])
    return F.cross_entropy(output.logits, batch.labels, weight=weights)


def run(configuration: ExperimentConfiguration, arguments: argparse.Namespace) -> None:
    configuration.validate()
    context = initialize_distributed()
    set_seed(configuration.seed + context.rank)
    manifest = CohortManifest.from_csv(configuration.data.manifest)
    curriculum = AvailabilityCurriculum.from_csv(configuration.data.availability)
    dataset = MultimodalFeatureDataset(manifest)
    batch_size = configuration.training.batch_size
    if batch_size is None:
        raise ValueError("batch size must be supplied because the paper does not report it")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=arguments.workers,
        collate_fn=collate_features,
        pin_memory=context.device.type == "cuda",
    )
    model = wrap_distributed(build_model(configuration, arguments), context)
    weight_decay = configuration.training.weight_decay
    if weight_decay is None:
        raise ValueError("weight decay must be supplied because the paper does not report it")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=configuration.training.learning_rate,
        weight_decay=weight_decay,
    )
    precision = configuration.training.precision
    scaler = torch.cuda.amp.GradScaler() if precision == "fp16" else None
    engine = TrainingEngine(
        model,
        optimizer,
        loss_function,
        context.device,
        scaler=scaler,
        gradient_clip=configuration.training.gradient_clipping,
    )
    output = configuration.output.directory
    if context.primary:
        output.mkdir(parents=True, exist_ok=True)
    for epoch in range(configuration.training.epochs):
        masks, _ = curriculum.sample(len(dataset))
        if masks.shape[0] != len(dataset):
            raise RuntimeError("availability curriculum sampling failed")
        result = engine.train_epoch(loader, epoch)
        if context.primary:
            save_training_state(
                output / "latest.pt",
                model,
                optimizer,
                None,
                scaler,
                epoch,
                engine.state.global_step,
                configuration.seed,
                {"loss": result.mean_loss, "manifest": manifest.digest()},
            )
    finalize_distributed()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arguments = parser().parse_args()
    run(load_configuration(arguments.config), arguments)


if __name__ == "__main__":
    main()
