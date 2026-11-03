"""Conditional flow matching: a sampler of p(x1 | cond) from a velocity field.

Rectified (linear) paths x_t = (1 - t) x_0 + t x_1 with x_0 ~ N(0, I) and
target velocity x_1 - x_0; sampling integrates the learned field from t = 0
to 1 with fixed midpoint steps.

Hand-rolled because neither PyPI package fits: Meta's ``flow_matching``
(1.0.10, py3-none-any) installs a top-level ``tests`` package into
site-packages that shadows this repo's ``tests`` and breaks the suite's
``tests._sims`` plugin; ``torchcfm`` (1.0.7) pulls pot, torchdyn,
scikit-learn, pandas and absl-py for what is here a few lines.

The velocity network is any registered sequence architecture built with
``in_channels = x_channels + cond_channels + 2 * n_time_features`` and
``out_channels = x_channels``: the noisy target, the conditioning signals and
a Fourier embedding of t are concatenated as channels, so the backbone needs no
time-conditioning layer of its own.

Not TorchScript-packaged: the sampler is an ODE loop over the network, used
for analysis, not deployment.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional


class FlowMatching(nn.Module):
    """Velocity network plus the flow-matching loss and ODE sampler.

    Tensors are ``[B, C, L]``; ``x1`` has ``x_channels`` channels.
    """

    def __init__(self, velocity: nn.Module, x_channels: int, n_time_features: int):
        """Wrap ``velocity``, which maps the concatenated channels to ``x_channels``.

        Args:
            velocity: Network ``[B, x + cond + 2 F, L] -> [B, x, L]``.
            x_channels: Channels of the sampled quantity.
            n_time_features: F, the sin/cos frequency pairs embedding t.
        """
        super().__init__()
        self.velocity = velocity
        self.x_channels = x_channels
        # Frequencies pi * 2^k: the lowest is half a period over t in [0, 1],
        # so the embedding is one-to-one; higher ones resolve nearby times.
        self.register_buffer(
            'frequencies', math.pi * 2.0 ** torch.arange(n_time_features).float()
        )

    def field(self, x_t: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        """Velocity at ``x_t``; ``t`` is ``[B]``."""
        angle = t[:, None] * self.frequencies  # [B, F]
        emb = torch.cat([angle.sin(), angle.cos()], dim=1)
        emb = emb[:, :, None].expand(-1, -1, x_t.shape[-1])
        return self.velocity(torch.cat([x_t, cond, emb], dim=1))

    def loss(
        self, x1: Tensor, cond: Tensor, generator: torch.Generator | None
    ) -> Tensor:
        """Mean squared error of the field against x_1 - x_0 at t ~ U(0, 1).

        Args:
            x1: ``[B, x_channels, L]`` targets.
            cond: ``[B, cond_channels, L]`` conditioning signals.
            generator: Generator for x_0 and t, on ``x1``'s device, or None
                for the global RNG. Validation passes a fixed one: x_0 and t
                are the loss's own noise, and drawing them from the global
                stream would make the best-checkpoint pick follow it.
        """
        x0 = torch.randn(
            x1.shape, generator=generator, device=x1.device, dtype=x1.dtype
        )
        t = torch.rand(
            x1.shape[0], generator=generator, device=x1.device, dtype=x1.dtype
        )
        tb = t[:, None, None]
        x_t = (1 - tb) * x0 + tb * x1
        return functional.mse_loss(self.field(x_t, t, cond), x1 - x0)

    @torch.no_grad()
    def sample(
        self,
        cond: Tensor,
        n_samples: int,
        n_steps: int,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Draw ``n_samples`` per conditioning shot by midpoint integration.

        Args:
            cond: ``[B, cond_channels, L]``.
            n_samples: S, draws per shot.
            n_steps: Midpoint steps over t in [0, 1] (two field calls each).
            generator: Generator for the initial noise, on ``cond``'s device,
                or None for the global RNG.

        Returns:
            ``[S, B, x_channels, L]``.
        """
        batch, _, length = cond.shape
        x = torch.randn(
            (n_samples * batch, self.x_channels, length),
            generator=generator,
            device=cond.device,
            dtype=cond.dtype,
        )
        cond = cond.repeat(n_samples, 1, 1)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
            half = x + 0.5 * dt * self.field(x, t, cond)
            x = x + dt * self.field(half, t + 0.5 * dt, cond)
        return x.reshape(n_samples, batch, self.x_channels, length)
