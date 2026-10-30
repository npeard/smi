#!/usr/bin/env python3
"""Pure-PyTorch Mamba selective state-space model.

Implements the Mamba architecture from:
    Gu, A. & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with
    Selective State Spaces. arXiv:2312.00752.

This implementation uses a chunked scan in plain tensor ops, so it runs on
any device (CPU, CUDA or MPS) and scripts with TorchScript, without the
custom CUDA kernels required by the original ``mamba-ssm`` package.
"""

import logging
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from .base import SequenceModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Selective scan
# ---------------------------------------------------------------------------


def _block_scan(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Inclusive scan of h_t = a_t * h_{t-1} + b_t, h_{-1} = 0, along dim -2.

    The time axis is -2 (length T, small) so each step touches a contiguous
    [..., n] slab holding every block at once. Returns (P, H): the running
    product of a and the local h, both [..., T, n].
    """
    a_steps = a.unbind(-2)
    b_steps = b.unbind(-2)
    p = a_steps[0]
    h = b_steps[0]
    ps = [p]
    hs = [h]
    for t in range(1, len(a_steps)):
        h = torch.addcmul(b_steps[t], a_steps[t], h)
        p = p * a_steps[t]
        ps.append(p)
        hs.append(h)
    return torch.stack(ps, -2), torch.stack(hs, -2)


def diagonal_linear_scan(a: Tensor, b: Tensor) -> Tensor:
    """Solve h_t = a_t * h_{t-1} + b_t (h_{-1} = 0) along the last dim.

    A chunked scan replacing the O(L) per-sample loop, whose one small kernel
    launch per sample dominated MPS step time. Each level scans blocks of
    ``block`` samples sequentially, every block at once, and hands the
    block-end values (a block's total decay and local state) up as a
    ``block``-times shorter recurrence; the way down adds each block's
    carry-in, h_local + P * h_prev_block_end. Every operation is a product of
    the a_t or an a*h + b update, as in the sequential loop, so nothing is
    divided by a decay or exponentiated out of range: a cumulative-sum form
    (exp(S_t) * cumsum(exp(-S_s) b_s)) would overflow float32 once a block's
    total log-decay passes about -88. Causal, since padding is appended
    after the sequence as identity steps (a = 1, b = 0).

    Memory: autograd keeps the padded a, b and both levels' (P, H), about four
    [batch, d_inner, d_state, seq_len] tensors per layer.
    """
    # Block length. The step count is about block * log_block(seq_len): 32
    # gives 96 sequential steps at 4096 samples instead of 4096. On MPS 8, 16
    # and 32 tied for training at 4096 samples and 32 was fastest for
    # evaluation at 16384. A local, not a module constant: TorchScript cannot
    # read a Python global int.
    block = 32
    # Way up: per level, the (P, H) needed on the way down and its length.
    levels: list[tuple[Tensor, Tensor, int]] = []
    while a.shape[-1] > block:
        length = a.shape[-1]
        pad = (-length) % block
        if pad > 0:
            a = functional.pad(a, (0, pad), value=1.0)
            b = functional.pad(b, (0, pad), value=0.0)
        # contiguous(): MPS runs each step's elementwise op on a strided
        # [..., n] slice about 2x slower than on a contiguous one.
        p, h = _block_scan(
            a.unflatten(-1, (-1, block)).transpose(-1, -2).contiguous(),
            b.unflatten(-1, (-1, block)).transpose(-1, -2).contiguous(),
        )
        levels.append((p, h, length))
        a = p[..., -1, :]
        b = h[..., -1, :]
    h = _block_scan(a.unsqueeze(-1), b.unsqueeze(-1))[1].squeeze(-1)
    # Way down: h now holds the true state at each block end of the level.
    for i in range(len(levels) - 1, -1, -1):
        p, h_local, length = levels[i]
        carry = functional.pad(h[..., :-1], (1, 0)).unsqueeze(-2)
        h = torch.addcmul(h_local, p, carry).transpose(-1, -2).flatten(-2)
        h = h[..., :length]
    return h


def selective_scan(
    u: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor
) -> Tensor:
    """Selective scan (ZOH discretisation).

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
    # ZOH discretisation: A_bar = exp(delta * A), B_bar = delta * B (simplified Euler)
    delta_A = torch.exp(
        delta.unsqueeze(2) * A.unsqueeze(0).unsqueeze(-1)
    )  # [batch, d_model, d_state, seq_len]
    delta_B_u = (
        delta.unsqueeze(2) * B.unsqueeze(1) * u.unsqueeze(2)
    )  # [batch, d_model, d_state, seq_len]
    h = diagonal_linear_scan(delta_A, delta_B_u)
    # y_t = C_t * h_t summed over the state dim
    y = (C.unsqueeze(1) * h).sum(dim=2)  # [batch, d_model, seq_len]
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


class Mamba(SequenceModel):
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
        for module in self.layers:
            layer: MambaBlock = module  # ty: ignore[invalid-assignment]  # ModuleList iterates as Module; layers are all MambaBlock
            dt_init_std = layer.d_model**-0.5
            nn.init.uniform_(layer.dt_proj.weight, -dt_init_std, dt_init_std)
            d_inner: int = layer.d_inner
            dt = torch.exp(
                torch.rand((d_inner,)) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                # dt_proj is an nn.Linear with bias=True, so .bias is a
                # Tensor; the annotation lets ty confirm .copy_ is callable.
                bias: Tensor = layer.dt_proj.bias
                bias.copy_(inv_dt)

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
    def causal(self) -> bool:
        """The scan and its depthwise conv are left-to-right only."""
        return True

    @property
    @torch.jit.unused
    def receptive_field(self) -> int | None:
        """Recurrent state has no finite lookback."""
        return None

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
