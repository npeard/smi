"""Evaluation metrics in stored physical units.

The training loop logs losses, which are not comparable to the analytic
baseline's numbers. These helpers produce the same quantities the baseline
artifact reports: velocity RMSE in um/ms and displacement RMSE in um,
summarized per shot with the same six keys ``baseline_results.json`` uses.
"""

import numpy as np
import torch

from smi.synthetic.coil_driver import MS_PER_S, CoilDriver

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


def displacement_from_velocity(velocity: torch.Tensor, dt_ms: float) -> torch.Tensor:
    """Integrate a velocity waveform to displacement in um.

    Displacement obtained this way is *derived*: it is the cumulative integral
    of a predicted velocity, not a directly fitted displacement. The analytic
    baseline fits displacement directly, so the two displacement numbers are
    not measured the same way and a table placing them side by side must say
    so.

    Integration is delegated to :meth:`CoilDriver.integrate_velocity` rather
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

    displacement = CoilDriver.integrate_velocity(flat, sample_rate)
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
