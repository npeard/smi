#!/usr/bin/env python3
"""Pure-PyTorch Mamba selective state-space model.

Implements the Mamba architecture from:
    Gu, A. & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with
    Selective State Spaces. arXiv:2312.00752.

This implementation uses the parallel associative-scan formulation so it
runs on any device (CPU or CUDA) without the custom CUDA kernels required
by the original ``mamba-ssm`` package.
"""

import logging
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Selective scan (parallel prefix-sum / associative scan)
# ---------------------------------------------------------------------------


def selective_scan(
    u: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor
) -> Tensor:
    """Parallel selective scan (ZOH discretisation).

    Args:
        u:     [batch, d_model, seq_len]   input
        delta: [batch, d_model, seq_len]   input-dependent step sizes (after softplus)
        A:     [d_model, d_state]          state matrix (stored as log for stability)
        B:     [batch, d_state, seq_len]   input-dependent B projection
        C:     [batch, d_state, seq_len]   input-dependent C projection
        D:     [d_model]                   skip connection weight

    Returns:
        y: [batch, d_model, seq_len]
    """
    batch, d_model, seq_len = u.shape
    d_state = A.shape[1]

    # ZOH discretisation: A_bar = exp(delta * A), B_bar = delta * B (simplified Euler)
    # delta: [batch, d_model, seq_len] -> unsqueeze for state dim
    # A:     [d_model, d_state]
    delta_A = torch.exp(
        delta.unsqueeze(2) * A.unsqueeze(0).unsqueeze(-1)
    )  # [batch, d_model, d_state, seq_len]

    # B: [batch, d_state, seq_len], delta: [batch, d_model, seq_len]
    delta_B_u = (
        delta.unsqueeze(2) * B.unsqueeze(1) * u.unsqueeze(2)
    )  # [batch, d_model, d_state, seq_len]

    # Sequential scan over time -- O(L) but simple and correct
    # For moderate seq_len this is the bottleneck; a parallel scan would be
    # O(L log L) but requires more complex code.
    h = torch.zeros(batch, d_model, d_state, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(seq_len):
        h = delta_A[..., t] * h + delta_B_u[..., t]
        # y_t = C_t * h_t  summed over state dim
        y_t = (C[:, :, t].unsqueeze(1) * h).sum(dim=2)  # [batch, d_model]
        ys.append(y_t)

    y = torch.stack(ys, dim=2)  # [batch, d_model, seq_len]
    return y + D.unsqueeze(0).unsqueeze(-1) * u


# ---------------------------------------------------------------------------
# Mamba block
# ---------------------------------------------------------------------------


class MambaBlock(nn.Module):
    """Single Mamba layer with selective SSM and gated MLP.

    The block follows the original paper's structure::

        x -> norm -> [expand] -> conv1d -> SiLU -> SSM -> gate -> contract -> residual
             norm -> [expand]  (z gate) -------------------------^
    """

    def __init__(
        self, d_model: int, d_state: int, d_conv: int, expand: int, use_layer_norm: bool
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state

        # Input normalisation
        if use_layer_norm:
            self.norm = nn.LayerNorm(d_model)
        else:
            self.norm = nn.Identity()

        # Expand projection: produces both the SSM branch and the gate
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Short depthwise conv before SSM (causal: pad left only)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,  # left-only causal padding, chomp later
            bias=True,
        )
        self._d_conv = d_conv  # saved to chomp during forward

        # SSM input-dependent projections
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        # d_state*2 = B + C dims, d_inner = delta dims before log

        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # A: log-parameterised diagonal state matrix, initialised as in the paper
        A = (
            torch.arange(1, d_state + 1, dtype=torch.float32)
            .unsqueeze(0)
            .expand(self.d_inner, -1)
        )
        self.A_log = nn.Parameter(torch.log(A))

        # D skip connection weight
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output contraction
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: [batch, d_model, seq_len]  (channels-first, matching project convention)

        Returns:
            [batch, d_model, seq_len]
        """
        residual = x
        _batch, _d_model, seq_len = x.shape

        # LayerNorm operates on the last dim -> transpose
        x = x.permute(0, 2, 1)  # [batch, seq_len, d_model]
        x = self.norm(x)

        # Expand: [batch, seq_len, d_inner * 2]
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)  # each [batch, seq_len, d_inner]

        # Causal depthwise conv on the SSM branch
        x_branch = x_branch.permute(0, 2, 1)  # [batch, d_inner, seq_len]
        x_branch = self.conv1d(x_branch)[..., :seq_len]  # chomp future
        x_branch = functional.silu(x_branch)

        # SSM projections: B, C, delta from x_branch
        # x_branch: [batch, d_inner, seq_len] -> permute for linear
        x_branch_t = x_branch.permute(0, 2, 1)  # [batch, seq_len, d_inner]
        bcd = self.x_proj(x_branch_t)  # [batch, seq_len, d_state*2 + d_inner]
        B, C, delta = bcd.split([self.d_state, self.d_state, self.d_inner], dim=-1)

        delta = functional.softplus(self.dt_proj(delta))  # [batch, seq_len, d_inner]

        # Reshape to channels-first for scan
        B = B.permute(0, 2, 1)  # [batch, d_state, seq_len]
        C = C.permute(0, 2, 1)  # [batch, d_state, seq_len]
        delta = delta.permute(0, 2, 1)  # [batch, d_inner, seq_len]

        A = -torch.exp(self.A_log)  # [d_inner, d_state] -- negative for stability

        y = selective_scan(
            x_branch, delta, A, B, C, self.D
        )  # [batch, d_inner, seq_len]

        # Gate with z branch (SiLU gating)
        z = functional.silu(z)  # [batch, seq_len, d_inner]
        y = y.permute(0, 2, 1) * z  # [batch, seq_len, d_inner]

        # Contract back to d_model
        y = self.out_proj(y)  # [batch, seq_len, d_model]
        y = y.permute(0, 2, 1)  # [batch, d_model, seq_len]

        return y + residual


# ---------------------------------------------------------------------------
# Full Mamba model
# ---------------------------------------------------------------------------


@dataclass
class MambaConfig:
    """Configuration for stacked Mamba SSM model.

    Args:
        sequence_length: Length of input sequence (time dimension)
        in_channels: Number of input channels (e.g., 3 for interferometer data)
        d_model: Internal feature dimension for Mamba blocks
        d_state: SSM state dimension (N in the paper; controls memory capacity)
        d_conv: Kernel size of the causal depthwise conv inside each block
        expand: Expansion factor for the inner dimension (d_inner = d_model * expand)
        num_layers: Number of stacked Mamba blocks
        use_layer_norm: Whether to apply LayerNorm at the start of each block
    """

    sequence_length: int
    in_channels: int
    d_model: int
    d_state: int
    d_conv: int
    expand: int
    num_layers: int
    use_layer_norm: bool

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError(f'num_layers must be >= 1, got {self.num_layers}')
        if self.d_state < 1:
            raise ValueError(f'd_state must be >= 1, got {self.d_state}')
        if self.expand < 1:
            raise ValueError(f'expand must be >= 1, got {self.expand}')


class Mamba(nn.Module):
    """Stacked Mamba selective state-space model.

    Takes multi-channel input of shape [batch, in_channels, seq_len] and
    produces output of shape [batch, 1, seq_len], matching the interface
    expected by LitModule.

    Architecture:
        lifting  ->  N x MambaBlock  ->  projection
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config

        # Lift input channels to d_model
        self.lifting = nn.Conv1d(config.in_channels, config.d_model, 1)

        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand,
                    use_layer_norm=config.use_layer_norm,
                )
                for _ in range(config.num_layers)
            ]
        )

        # Final norm before projection
        if config.use_layer_norm:
            self.final_norm = nn.Sequential(
                nn.Conv1d(1, 1, 1)  # placeholder -- replaced below
            )
            # Use a proper channels-last LayerNorm via a small wrapper
            self.final_norm = _ChannelsLastLayerNorm(config.d_model)
        else:
            self.final_norm = nn.Identity()

        # Project to single output channel
        self.projection = nn.Conv1d(config.d_model, 1, 1)

        self._init_weights()

        logger.info(f'Number of parameters in Mamba: {self.total_params:,}')

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear | nn.Conv1d):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # dt_proj bias initialised as in paper (log-uniform between dt_min, dt_max)
        dt_min, dt_max = 0.001, 0.1
        for layer in self.layers:
            dt_init_std = layer.d_model**-0.5
            nn.init.uniform_(layer.dt_proj.weight, -dt_init_std, dt_init_std)
            dt = torch.exp(
                torch.rand(layer.d_inner) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                layer.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through Mamba.

        Args:
            x: [batch, in_channels, seq_len]

        Returns:
            [batch, 1, seq_len]
        """
        x = self.lifting(x)  # [batch, d_model, seq_len]
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)  # [batch, d_model, seq_len]
        return self.projection(x)  # [batch, 1, seq_len]

    @property
    @torch.jit.unused
    def total_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _ChannelsLastLayerNorm(nn.Module):
    """LayerNorm wrapper for channels-first tensors [batch, C, L]."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, d_model, seq_len]
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        return x.permute(0, 2, 1)
