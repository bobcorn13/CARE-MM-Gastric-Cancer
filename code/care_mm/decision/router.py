from dataclasses import dataclass

import torch

from care_mm.types import Action, RouteResult


@dataclass(frozen=True)
class ActionBoundary:
    action: Action
    minimum_probability: float
    maximum_probability: float


class CostSensitiveRouter:
    def __init__(self, costs: torch.Tensor, action_names: tuple[Action, ...]) -> None:
        if costs.ndim != 2:
            raise ValueError("cost matrix must have action and outcome dimensions")
        if costs.shape[0] != len(action_names):
            raise ValueError("action count and cost rows differ")
        if torch.any(costs < 0):
            raise ValueError("costs must be non-negative")
        self.costs = costs.float()
        self.action_names = action_names

    @classmethod
    def diagnosis_router(
        cls, false_negative_cost: float = 10.0, false_positive_cost: float = 1.0
    ) -> "CostSensitiveRouter":
        costs = torch.tensor(
            [
                [0.0, false_negative_cost],
                [false_positive_cost, 0.0],
            ]
        )
        return cls(costs, (Action.BIOPSY, Action.SURGICAL_REFERRAL))

    def expected_costs(self, probabilities: torch.Tensor) -> torch.Tensor:
        if probabilities.shape[-1] != self.costs.shape[1]:
            raise ValueError("probability class count and cost outcomes differ")
        costs = self.costs.to(probabilities.device)
        return probabilities @ costs.transpose(0, 1)

    def route(
        self,
        probabilities: torch.Tensor,
        prediction_sets: torch.Tensor,
    ) -> tuple[RouteResult, ...]:
        if probabilities.ndim != 2 or prediction_sets.shape != probabilities.shape:
            raise ValueError("probabilities and prediction sets must be aligned matrices")
        expected = self.expected_costs(probabilities)
        results = []
        for row in range(probabilities.shape[0]):
            members = tuple(torch.where(prediction_sets[row])[0].tolist())
            probability_values = tuple(float(item) for item in probabilities[row])
            if len(members) != 1:
                results.append(
                    RouteResult(Action.ABSTAIN, float("nan"), members, probability_values)
                )
                continue
            index = int(expected[row].argmin())
            results.append(
                RouteResult(
                    self.action_names[index],
                    float(expected[row, index]),
                    members,
                    probability_values,
                )
            )
        return tuple(results)

    def decision_boundaries(self, resolution: int = 10001) -> tuple[ActionBoundary, ...]:
        positive = torch.linspace(0.0, 1.0, resolution)
        probabilities = torch.stack((1.0 - positive, positive), dim=1)
        choices = self.expected_costs(probabilities).argmin(dim=1)
        boundaries = []
        for index, action in enumerate(self.action_names):
            selected = torch.where(choices == index)[0]
            if selected.numel() == 0:
                continue
            boundaries.append(
                ActionBoundary(
                    action,
                    float(positive[selected.min()]),
                    float(positive[selected.max()]),
                )
            )
        return tuple(boundaries)


class TriageCostMatrix:
    def __init__(self, matrix: torch.Tensor) -> None:
        if matrix.shape != (3, 3):
            raise ValueError("triage cost matrix must be three by three")
        if torch.any(torch.diag(matrix) != 0):
            raise ValueError("correct triage actions must have zero cost")
        self.matrix = matrix.float()

    def router(self) -> CostSensitiveRouter:
        return CostSensitiveRouter(
            self.matrix,
            (Action.BIOPSY, Action.ENDOSCOPIC_RESECTION, Action.SURGICAL_REFERRAL),
        )

    def harm(self, actions: torch.Tensor, outcomes: torch.Tensor) -> torch.Tensor:
        if actions.shape != outcomes.shape:
            raise ValueError("actions and outcomes must have identical shape")
        return self.matrix.to(actions.device)[actions, outcomes]
