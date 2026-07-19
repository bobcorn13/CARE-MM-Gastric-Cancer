from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    diagnosis: torch.Tensor
    triage: torch.Tensor
    entropy: torch.Tensor


class CostSensitiveObjective(nn.Module):
    def __init__(
        self,
        diagnosis_weights: tuple[float, float] = (1.0, 10.0),
        triage_weight: float = 1.0,
        entropy_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("diagnosis_weights", torch.tensor(diagnosis_weights))
        self.triage_weight = triage_weight
        self.entropy_weight = entropy_weight

    def forward(
        self,
        diagnosis_logits: torch.Tensor,
        diagnosis_labels: torch.Tensor,
        attention: torch.Tensor,
        triage_logits: torch.Tensor | None = None,
        triage_labels: torch.Tensor | None = None,
    ) -> LossBreakdown:
        diagnosis = F.cross_entropy(
            diagnosis_logits,
            diagnosis_labels,
            weight=self.diagnosis_weights,
        )
        if triage_logits is None or triage_labels is None:
            triage = diagnosis.new_zeros(())
        else:
            triage = F.cross_entropy(triage_logits, triage_labels)
        stable_attention = attention.clamp_min(1e-12)
        entropy = -(stable_attention * stable_attention.log()).sum(dim=-1).mean()
        total = diagnosis + self.triage_weight * triage + self.entropy_weight * entropy
        return LossBreakdown(total, diagnosis, triage, entropy)


class FocalCostObjective(nn.Module):
    def __init__(self, gamma: float = 2.0, false_negative_weight: float = 10.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weights", torch.tensor([1.0, false_negative_weight]))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probabilities = F.log_softmax(logits, dim=-1)
        probabilities = log_probabilities.exp()
        selected_log = log_probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
        selected_probability = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
        class_weights = self.weights[labels]
        return (-class_weights * (1 - selected_probability).pow(self.gamma) * selected_log).mean()


def modality_consistency_loss(
    full_logits: torch.Tensor,
    partial_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    teacher = torch.softmax(full_logits.detach() / temperature, dim=-1)
    student = torch.log_softmax(partial_logits / temperature, dim=-1)
    return F.kl_div(student, teacher, reduction="batchmean") * temperature**2
