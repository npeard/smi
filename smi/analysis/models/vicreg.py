import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class VicRegLoss(nn.Module):
    def __init__(
        self, sim_coeff: float = 25.0, std_coeff: float = 25.0, cov_coeff: float = 1.0
    ):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def off_diagonal(self, x: Tensor) -> Tensor:
        # Returns a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute VICReg loss between two feature tensors.

        Args:
            x: Projector output from channel 1, shape [batch, dim].
            y: Projector output from channel 2, shape [batch, dim].

        Returns:
            Scalar VICReg loss.
        """
        batch_size = x.shape[0]
        num_features = x.shape[1]

        # 1. Invariance Loss (MSE)
        repr_loss = F.mse_loss(x, y)

        # 2. Variance Loss (Hinge)
        # Force std dev to be close to 1.
        # clamp(min=1e-4) avoids division by zero
        std_x = torch.sqrt(x.var(dim=0) + 1e-4)
        std_y = torch.sqrt(y.var(dim=0) + 1e-4)

        # We want std >= 1. The relu (hinge) punishes std < 1.
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        # 3. Covariance Loss (Decorrelation)
        # Center the features (subtract mean)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        # Calculate Covariance Matrix
        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)

        # Penalize off-diagonal elements
        cov_loss = (
            self.off_diagonal(cov_x).pow(2).sum() / num_features
            + self.off_diagonal(cov_y).pow(2).sum() / num_features
        )

        return (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
