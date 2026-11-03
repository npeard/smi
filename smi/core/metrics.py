"""Evaluation metrics in stored physical units.

The training loop logs losses, which are not comparable to the analytic
baseline's numbers. These helpers produce the same quantities the baseline
artifact reports: velocity RMSE in um/ms and displacement RMSE in um,
summarized per shot with the same six keys ``baseline_results.json`` uses.
"""

import math

import numpy as np
import torch

from smi.core.kinematics import MS_PER_S, integrate_velocity

#: The six summary keys, in the order ``baseline_results.json`` defines them.
SUMMARY_KEYS = ('mean', 'median', 'p25', 'p75', 'min', 'max')


def velocity_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root-mean-square velocity error in um/ms.

    Both arguments are already in stored units (um/ms), so no rescale happens
    here -- a loss-only rescale at this boundary is what previously made the
    training log and the reported baseline differ by 1000x.

    Args:
        pred: Predicted velocity, any shape.
        target: Ground-truth velocity, broadcastable to ``pred``'s shape.

    Returns:
        Scalar tensor on ``pred``'s device holding the RMSE over all elements.
    """
    return torch.sqrt(torch.mean((pred - target) ** 2))


def per_shot_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """RMSE over the last (time) axis: ``[batch, length]`` -> ``[batch]``."""
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=-1))


def sign_invariant_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-shot RMSE after the best global sign: min over s in {+1, -1}.

    A Michelson interferometer cannot tell the direction of motion, so a
    prediction is correct up to one sign per shot. The sign is chosen per shot,
    not per sample, so a decoder that flips sign mid-shot is still penalized.

    Args:
        pred: Predicted velocity, ``[batch, length]``.
        target: Ground-truth velocity, same shape.

    Returns:
        ``[batch]`` RMSE in the units of the inputs.
    """
    # Shares best_sign with the training loss so validation and evaluation can
    # never disagree on which sign a shot gets.
    return per_shot_rmse(best_sign(pred, target) * pred, target)


def best_sign(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-shot global sign s in {+1, -1} minimizing ``||s * pred - target||``.

    Args:
        pred: Predicted velocity, ``[batch, length]``.
        target: Ground-truth velocity, same shape.

    Returns:
        ``[batch, 1]`` signs, detached: the choice carries no gradient, the
        chosen branch does.
    """
    # sign(pred . target); ties go to +1.
    dot = (pred * target).sum(dim=-1, keepdim=True).detach()
    return torch.where(dot >= 0, 1.0, -1.0).to(pred.dtype)


def apply_best_sign(
    velocity_hat: torch.Tensor,
    displacement_hat: torch.Tensor,
    trained_hat: torch.Tensor,
    trained_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flip each shot's velocity and displacement prediction by one shared sign.

    The sign is chosen on the quantity the model is trained on (velocity or
    displacement, passed as ``trained_hat``/``trained_target``); the other is
    its integral or derivative, so it must follow the same choice rather than
    pick its own. For a displacement target the differentiated velocity is
    dominated by amplified ripple, so a sign chosen on it would be noise. Both
    training loss (``LitModule``) calls this, so the
    choice lives in one place.

    Args:
        velocity_hat: Predicted velocity, ``[batch, length]``.
        displacement_hat: Predicted displacement, same shape.
        trained_hat: Whichever of the two the model outputs.
        trained_target: Its ground truth, same shape.

    Returns:
        ``(sign * velocity_hat, sign * displacement_hat)``.
    """
    sign = best_sign(trained_hat, trained_target)
    return sign * velocity_hat, sign * displacement_hat


def displacement_from_velocity(velocity: torch.Tensor, dt_ms: float) -> torch.Tensor:
    """Integrate a velocity waveform to displacement in um.

    Displacement obtained this way is *derived*: it is the cumulative integral
    of a predicted velocity, not a directly fitted displacement. The analytic
    baseline fits displacement directly, so the two displacement numbers are
    not measured the same way and a table placing them side by side must say
    so.

    Integration is delegated to :func:`smi.core.kinematics.integrate_velocity` rather
    than re-deriving the rule here. That method accepts only 1-D or 2-D input,
    while every model in this project emits ``[batch, 1, length]``; the
    channel axis is squeezed for the call and restored afterwards.

    Args:
        velocity: Velocity in um/ms, shaped ``[batch, length]`` or
            ``[batch, 1, length]``.
        dt_ms: Sample spacing in milliseconds.

    Returns:
        Displacement in um, with the same shape as ``velocity``, starting at
        zero for each waveform.

    Raises:
        ValueError: If ``velocity`` has a channel axis wider than one, or more
            than three dimensions.
    """
    sample_rate = MS_PER_S / dt_ms

    if velocity.ndim == 3:
        if velocity.shape[1] != 1:
            raise ValueError(
                'displacement_from_velocity expects a single channel, got '
                f'{velocity.shape[1]}'
            )
        flat = velocity.squeeze(1)
    elif velocity.ndim <= 2:
        flat = velocity
    else:
        raise ValueError(f'Unsupported velocity shape: {tuple(velocity.shape)}')

    displacement = integrate_velocity(flat, sample_rate)
    assert isinstance(displacement, torch.Tensor)

    return displacement.unsqueeze(1) if velocity.ndim == 3 else displacement


def summarize_per_shot(values: np.ndarray) -> dict[str, float]:
    """Summarize a per-shot metric with the baseline artifact's six keys.

    Args:
        values: One metric value per shot.

    Returns:
        Mapping with keys ``mean``, ``median``, ``p25``, ``p75``, ``min`` and
        ``max``. The key set matches ``baseline_results.json`` exactly so the
        report can tabulate analytic and learned numbers without translating
        between two vocabularies.
    """
    return {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'p25': float(np.percentile(values, 25)),
        'p75': float(np.percentile(values, 75)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
    }


def band_coverage(draws: torch.Tensor, truth: torch.Tensor, level: float) -> float:
    """Fraction of ``truth`` inside the central ``level`` band of ``draws``.

    Args:
        draws: ``[S, ...]`` samples of the predictive distribution.
        truth: ``[...]``, the realized values.
        level: Band probability, e.g. 0.9 for the 5-95% quantiles.
    """
    tail = (1 - level) / 2
    q = torch.quantile(draws, torch.tensor([tail, 1 - tail], dtype=draws.dtype), dim=0)
    return float(((truth >= q[0]) & (truth <= q[1])).double().mean())


def crps_samples(draws: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Fair sample CRPS, ``E|X - y| - E|X - X'| / 2``, elementwise over ``truth``.

    The pair mean runs over the ``S (S - 1)`` distinct pairs: dividing by
    ``S^2`` counts the zero self-pairs and biases the score high by about
    ``E|X - X'| / (2 S)``, which at 64 draws is comparable to the gaps being
    compared. The pair term uses the sorted-sample identity
    ``sum_ij |x_i - x_j| = 2 sum_i (2 i - S + 1) x_(i)`` (0-based), which is
    O(S log S) rather than O(S^2) memory.
    """
    s = draws.shape[0]
    spread = (draws - truth).abs().mean(dim=0)
    ordered = draws.sort(dim=0).values
    weight = (2 * torch.arange(s, dtype=draws.dtype) - s + 1).reshape(
        -1, *([1] * truth.dim())
    )
    pair = 2 * (weight * ordered).sum(dim=0) / (s * (s - 1))
    return spread - pair / 2


def crps_gaussian(
    mean: torch.Tensor, std: torch.Tensor, truth: torch.Tensor
) -> torch.Tensor:
    """Closed-form CRPS of ``N(mean, std^2)`` at ``truth``."""
    z = (truth - mean) / std
    normal = torch.distributions.Normal(0.0, 1.0)
    return std * (
        z * (2 * normal.cdf(z) - 1)
        + 2 * normal.log_prob(z).exp()
        - 1 / math.sqrt(math.pi)
    )
