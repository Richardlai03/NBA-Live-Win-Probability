import torch
import torch.nn as nn

FEATURE_NAMES = [
    "score_diff",
    "time_remaining_s",
    "lead_leverage",
    "period",
]

N_FEATURES = len(FEATURE_NAMES)


class WinProbNet(nn.Module):
    def __init__(self, n_features: int = N_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> float:
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x)).item()
