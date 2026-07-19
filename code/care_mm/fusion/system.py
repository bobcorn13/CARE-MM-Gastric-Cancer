from dataclasses import dataclass

import torch
from torch import nn

from care_mm.fusion.gate import FusionHead
from care_mm.representation.ehr import FTTransformerEncoder
from care_mm.representation.endoscopy import EndoscopySequenceEncoder
from care_mm.representation.pathology import PathologyMILEncoder
from care_mm.types import ModelOutput


@dataclass(frozen=True)
class CareMMInputs:
    endoscopy: torch.Tensor | None
    endoscopy_padding: torch.Tensor | None
    pathology: torch.Tensor | None
    pathology_padding: torch.Tensor | None
    ehr_numerical: torch.Tensor | None
    ehr_categorical: torch.Tensor | None
    ehr_missing: torch.Tensor | None
    available: torch.Tensor


class CareMM(nn.Module):
    def __init__(
        self,
        endoscopy_width: int,
        pathology_width: int,
        ehr_numerical: int,
        ehr_cardinalities: tuple[int, ...],
        width: int = 512,
        depth: int = 4,
        heads: int = 8,
        classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.width = width
        self.endoscopy = EndoscopySequenceEncoder(
            endoscopy_width, width, heads=heads, dropout=dropout
        )
        self.pathology = PathologyMILEncoder(pathology_width, width, heads=heads, dropout=dropout)
        self.ehr = FTTransformerEncoder(
            ehr_numerical,
            ehr_cardinalities,
            width,
            depth=depth,
            heads=heads,
            dropout=dropout,
        )
        self.fusion = FusionHead(width, depth, heads, classes, dropout)

    def _empty(self, batch: int, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros((batch, self.width))

    def forward(self, inputs: CareMMInputs) -> ModelOutput:
        available = inputs.available
        if available.ndim != 2 or available.shape[1] != 3:
            raise ValueError("availability must have shape batch by three")
        batch = available.shape[0]
        reference = next(self.parameters())
        embeddings: list[torch.Tensor] = []
        if inputs.endoscopy is not None:
            if inputs.endoscopy_padding is None:
                raise ValueError("endoscopy padding mask is required")
            endoscopy, _ = self.endoscopy(inputs.endoscopy, inputs.endoscopy_padding)
            embeddings.append(endoscopy)
        else:
            embeddings.append(self._empty(batch, reference))
        if inputs.pathology is not None:
            if inputs.pathology_padding is None:
                raise ValueError("pathology padding mask is required")
            pathology, _ = self.pathology(inputs.pathology, inputs.pathology_padding)
            embeddings.append(pathology)
        else:
            embeddings.append(self._empty(batch, reference))
        if inputs.ehr_numerical is not None or inputs.ehr_categorical is not None:
            embeddings.append(
                self.ehr(inputs.ehr_numerical, inputs.ehr_categorical, inputs.ehr_missing)
            )
        else:
            embeddings.append(self._empty(batch, reference))
        tokens = torch.stack(embeddings, dim=1)
        logits, fused, attention = self.fusion(tokens, available)
        probabilities = torch.softmax(logits, dim=-1)
        return ModelOutput(logits, probabilities, fused, attention)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def freeze_encoders(self) -> None:
        for encoder in (self.endoscopy, self.pathology, self.ehr):
            encoder.requires_grad_(False)
            encoder.eval()

    def unfreeze_encoders(self) -> None:
        for encoder in (self.endoscopy, self.pathology, self.ehr):
            encoder.requires_grad_(True)
            encoder.train()
