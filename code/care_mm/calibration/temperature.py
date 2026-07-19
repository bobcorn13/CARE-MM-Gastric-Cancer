import torch
from torch import nn
from torch.nn import functional as F


class TemperatureScaler(nn.Module):
    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        if initial_temperature <= 0:
            raise ValueError("temperature must be positive")
        self.log_temperature = nn.Parameter(torch.tensor(initial_temperature).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(1e-3, 1e3)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self(logits), dim=-1)

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        maximum_iterations: int = 100,
        tolerance: float = 1e-7,
    ) -> float:
        if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
            raise ValueError("calibration logits and labels have incompatible shapes")
        logits = logits.detach()
        labels = labels.detach()
        optimizer = torch.optim.LBFGS(
            [self.log_temperature],
            lr=0.1,
            max_iter=maximum_iterations,
            tolerance_grad=tolerance,
            tolerance_change=tolerance,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(self(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(F.cross_entropy(self(logits), labels).detach())


class VectorScaler(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(classes))
        self.bias = nn.Parameter(torch.zeros(classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * self.scale + self.bias

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, iterations: int = 200) -> float:
        optimizer = torch.optim.LBFGS([self.scale, self.bias], max_iter=iterations)

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(self(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(F.cross_entropy(self(logits), labels).detach())
