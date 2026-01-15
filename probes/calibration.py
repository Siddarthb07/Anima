import torch
import torch.nn as nn


class PlattScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.a * logits + self.b)

    def fit(self, raw_scores: torch.Tensor, targets: torch.Tensor):
        optimizer = torch.optim.LBFGS(self.parameters(), lr=0.01, max_iter=100)
        criterion = nn.BCELoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(raw_scores.reshape(-1, 1)).squeeze(-1), targets.reshape(-1))
            loss.backward()
            return loss

        for _ in range(50):
            optimizer.step(closure)
