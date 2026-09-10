#!/usr/bin/env python3
"""Non-deep-learning baselines for displacement/velocity estimation.

This module answers the "establishing baselines" question: before claiming a
neural network is doing something useful, how well does classical, parametric
interferometry do on the same data?

Two distinct measurements live here, and it matters which is which.

**Method 1 -- forward-model residual** (:func:`forward_model_residual`).
Question answered: *does the Michelson forward model explain this data at
all?* We take the displacement implied by the known speaker drive voltage
(via :class:`~smi.synthetic.coil_driver.CoilDriver`), push it through
``cos(4*pi/lambda * d(t) + phi)``, and fit only the per-shot phase offset and
amplitude. Because the model is linear in its ``cos``/``sin`` components once
the displacement is fixed, this is a closed-form least-squares -- there is no
iterative optimizer and no local minimum to worry about. The reported number
is a residual RMSE on the photodiode signal, in volts. This is the honest
floor: no inverse method can beat the forward model's own ability to describe
the measurement.

**Method 2 -- inverse fit** (:func:`inverse_fit`).
Question answered: *how well can a classical estimator recover displacement
and velocity from the photodiode signals alone?* Only the three PD channels
are used; the drive voltage is held out and used purely as ground truth for
scoring. Displacement is recovered by multi-wavelength decoding across the
635 / 675 / 515 nm channels, then differentiated for velocity. The reported
numbers are displacement RMSE in microns and velocity RMSE in microns/second
-- the same units the network is scored in. **This is the number that belongs
beside the network's RMSE.**

The pairing is the point. A bad method-2 number on its own is ambiguous
between "the inverse algorithm is bad" and "the data does not obey the
assumed physics". Method 1 disambiguates: if method 1 also fits badly, the
model does not describe the data and no inverse -- classical or learned --
can do better than that ceiling.

Under Gaussian residuals the per-shot MSE *is* the negative log-likelihood up
to an additive constant, so "compute the likelihood of the data under the
Michelson model" and "fit each shot and report MSE" are the same measurement.
That is why the simple thing here is also the rigorous thing.

Unit conventions follow the rest of the package: wavelength and displacement
in microns, velocity in microns/second, time in seconds, photodiode signals
in volts.

Run as a script to regenerate the results artifact::

    pixi run -e dev python -m smi.analysis.baseline_fit --n-shots 200
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from scipy.signal import savgol_filter

from smi.synthetic.coil_driver import CoilDriver

logger = logging.getLogger(__name__)

# Photodiode channel -> laser wavelength in microns. RP1_CH1 is the speaker
# drive voltage and is not a photodiode.
PD_WAVELENGTHS_UM: dict[str, float] = {
    'RP1_CH2': 0.635,
    'RP2_CH1': 0.675,
    'RP2_CH2': 0.515,
}
DRIVE_KEY = 'RP1_CH1'

# Default location of the acquired datasets, relative to this file.
DATA_DIR = Path(__file__).resolve().parent / 'data'
DEFAULT_DATASETS: dict[str, str] = {
    'free-space': 'free-space-synchro_10k.h5',
    'mmfiber': 'mmfiber-synchro_10k.h5',
}
DEFAULT_RESULTS_PATH = DATA_DIR / 'baseline_results.json'


def _as_float64_zero_mean(signal: np.ndarray) -> np.ndarray:
    """Cast to float64 and remove the DC offset.

    The forward model produces a DC-free interference term (see
    ``MichelsonInterferometer.get_interferometer_output``), so the measured
    signal must be DC-removed before it can be compared to one.

    Args:
        signal: Signal samples of any float dtype.

    Returns:
        Zero-mean float64 copy of the input.
    """
    out = np.asarray(signal, dtype=np.float64)
    return out - out.mean()


# ---------------------------------------------------------------------------
# Method 1: forward-model residual
# ---------------------------------------------------------------------------


@dataclass
class PhaseAmplitudeFit:
    """Result of the closed-form phase/amplitude fit for one channel.

    Attributes:
        amplitude: Fitted fringe amplitude (V). Non-negative by construction.
        phase: Fitted phase offset (radians), wrapped to (-pi, pi].
        offset: Fitted constant offset (V); nominally zero for DC-removed data.
        rmse: Root-mean-square residual between fit and data (V).
        r_squared: Fraction of the signal variance explained by the fit.
    """

    amplitude: float
    phase: float
    offset: float
    rmse: float
    r_squared: float


def fit_phase_amplitude(signal: np.ndarray, theta: np.ndarray) -> PhaseAmplitudeFit:
    """Fit ``A * cos(theta + phi) + c`` to a signal in closed form.

    The model is nonlinear in ``phi`` but linear in the pair
    ``(A*cos(phi), -A*sin(phi))``, because::

        A * cos(theta + phi) = [A cos(phi)] * cos(theta) - [A sin(phi)] * sin(theta)

    So fitting reduces to an ordinary least-squares solve on the design matrix
    ``[cos(theta), sin(theta), 1]``, and the amplitude and phase are read back
    off the two coefficients. There is no iterative optimizer here and no
    local minimum: the least-squares solution is the global one.

    Args:
        signal: Measured samples (V), shape ``(n,)``.
        theta: Known model phase argument (radians), shape ``(n,)``. For the
            Michelson model this is ``4*pi/wavelength * displacement``.

    Returns:
        The fitted amplitude, phase, offset and residual statistics.

    Raises:
        ValueError: If the inputs do not have matching one-dimensional shapes.
    """
    signal = np.asarray(signal, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    if signal.ndim != 1 or theta.ndim != 1 or signal.shape != theta.shape:
        raise ValueError(
            f'signal and theta must be 1-D with matching shape, '
            f'got {signal.shape} and {theta.shape}'
        )

    design = np.stack([np.cos(theta), np.sin(theta), np.ones_like(theta)], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, signal, rcond=None)
    a_cos, b_sin, offset = (float(c) for c in coeffs)

    # signal ~ a*cos(theta) + b*sin(theta) = A*cos(theta + phi)
    # with A = hypot(a, b) and phi = atan2(-b, a).
    amplitude = float(np.hypot(a_cos, b_sin))
    phase = float(np.arctan2(-b_sin, a_cos))

    residual = signal - design @ coeffs
    rmse = float(np.sqrt(np.mean(residual**2)))
    variance = float(np.var(signal))
    r_squared = float(1.0 - np.var(residual) / variance) if variance > 0 else 0.0

    return PhaseAmplitudeFit(
        amplitude=amplitude, phase=phase, offset=offset, rmse=rmse, r_squared=r_squared
    )


def forward_model_residual(
    pd_signal: np.ndarray, displacement: np.ndarray, wavelength_um: float
) -> PhaseAmplitudeFit:
    """Method 1: fit the Michelson forward model to one channel of one shot.

    The displacement is *given* (derived from the speaker drive voltage), so
    the only free parameters are the per-shot fringe amplitude and phase
    offset. Whatever residual is left over is the part of the photodiode
    signal that the Michelson model plus the coil-driver calibration cannot
    explain -- detector noise, calibration error, fiber-induced distortion,
    speckle, or physics the model omits.

    Args:
        pd_signal: Photodiode samples for one shot (V), shape ``(n,)``.
        displacement: Mirror displacement for the same shot (microns), shape
            ``(n,)``. Note this is the mirror displacement, not the optical
            path difference, which is twice as large.
        wavelength_um: Laser wavelength for this channel (microns).

    Returns:
        The closed-form fit, whose ``rmse`` is the method-1 residual in volts
        and whose ``r_squared`` is the explained variance fraction.
    """
    signal = _as_float64_zero_mean(pd_signal)
    theta = 4.0 * np.pi / wavelength_um * np.asarray(displacement, dtype=np.float64)
    return fit_phase_amplitude(signal, theta)


# ---------------------------------------------------------------------------
# Method 2: inverse fit by multi-wavelength decoding
# ---------------------------------------------------------------------------


def _viterbi(
    cost: torch.Tensor, grid_step: float, max_step: int, smooth_weight: float
) -> torch.Tensor:
    """Minimum-cost monotone-in-time path through a displacement grid.

    Solves, over a batch of shots simultaneously::

        argmin_{j_0..j_{n-1}}  sum_i cost[b, j_i, i]
                             + smooth_weight * sum_i ((j_i - j_{i-1}) * dg)^2

    subject to ``|j_i - j_{i-1}| <= max_step``. The band constraint encodes a
    maximum plausible mirror speed, and the quadratic penalty makes the
    trajectory prefer smooth motion -- which is what breaks the
    ``cos`` ambiguity that defeats single-channel phase unwrapping when the
    velocity changes sign.

    Args:
        cost: Per-sample data cost, shape ``(batch, n_grid, n_time)``.
        grid_step: Spacing between adjacent displacement grid points (microns).
        max_step: Maximum grid points the path may move per time sample.
        smooth_weight: Weight on the squared-displacement-change penalty.

    Returns:
        Grid indices of the optimal path, shape ``(batch, n_time)``, dtype long.
    """
    batch, _n_grid, n_time = cost.shape
    device = cost.device
    offsets = torch.arange(-max_step, max_step + 1, device=device, dtype=cost.dtype)
    penalty = smooth_weight * (offsets * grid_step) ** 2

    running = cost[:, :, 0].clone()
    backpointer = torch.empty(
        (batch, cost.shape[1], n_time), dtype=torch.int16, device=device
    )
    for i in range(1, n_time):
        padded = torch.nn.functional.pad(
            running, (max_step, max_step), value=float('inf')
        )
        # window[b, j, k] == running[b, j + k - max_step]
        window = padded.unfold(1, 2 * max_step + 1, 1)
        best, arg = (window + penalty).min(dim=2)
        running = best + cost[:, :, i]
        backpointer[:, :, i] = (arg - max_step).to(torch.int16)

    node = running.argmin(dim=1)
    path = torch.empty((batch, n_time), dtype=torch.long, device=device)
    path[:, -1] = node
    rows = torch.arange(batch, device=device)
    for i in range(n_time - 1, 0, -1):
        node = node + backpointer[rows, node, i].long()
        path[:, i - 1] = node
    return path


def _channel_costs(
    signals: torch.Tensor,
    grid: torch.Tensor,
    wavelengths: torch.Tensor,
    phases: torch.Tensor,
) -> torch.Tensor:
    """Squared-error data cost of every grid displacement at every time sample.

    Args:
        signals: Amplitude-normalized PD signals, shape ``(batch, n_ch, n_time)``.
        grid: Candidate displacements (microns), shape ``(n_grid,)``.
        wavelengths: Channel wavelengths (microns), shape ``(n_ch,)``.
        phases: Per-shot per-channel phase offsets, shape ``(batch, n_ch)``.

    Returns:
        Cost tensor of shape ``(batch, n_grid, n_time)``.
    """
    batch, n_channels, n_time = signals.shape
    cost = torch.zeros(
        (batch, grid.numel(), n_time), dtype=signals.dtype, device=signals.device
    )
    # Accumulate channel by channel rather than materializing the full
    # (batch, n_ch, n_grid, n_time) difference, which is the peak-memory term.
    for k in range(n_channels):
        # model[b, g] = cos(4*pi/lambda_k * grid_g + phi_bk)
        model = torch.cos(
            4.0 * torch.pi / wavelengths[k] * grid[None, :] + phases[:, k, None]
        )
        cost += (model[:, :, None] - signals[:, k, None, :]) ** 2
    return cost


def _normalize_fringe_amplitude(signals: np.ndarray) -> np.ndarray:
    """Scale each channel so its fringes span roughly [-1, 1].

    A pure sinusoid has standard deviation ``A / sqrt(2)``, so dividing by
    ``sqrt(2) * std`` puts a clean fringe train on unit amplitude, which is
    what the ``cos`` template in the decoder assumes.

    Args:
        signals: PD signals, shape ``(..., n_ch, n_time)``.

    Returns:
        Zero-mean, unit-fringe-amplitude signals of the same shape.
    """
    out = np.asarray(signals, dtype=np.float64)
    out = out - out.mean(axis=-1, keepdims=True)
    scale = np.sqrt(2.0) * out.std(axis=-1, keepdims=True)
    scale = np.where(scale > 0, scale, 1.0)
    return out / scale


_Decoded = tuple[torch.Tensor, torch.Tensor]


def _phase_candidates(
    resolution: int, n_channels: int, device: torch.device
) -> torch.Tensor:
    """Enumerate the relative-phase grid searched by :func:`inverse_fit`.

    Only two relative phases are enumerated. An overall phase shift common to
    all channels is degenerate with a shift of the displacement origin, and
    that origin is itself unobservable from DC-removed fringes, so searching
    a third phase would only re-explore the same solutions.

    Args:
        resolution: Grid points per relative-phase dimension.
        n_channels: Number of photodiode channels.
        device: Device to build the tensor on.

    Returns:
        Candidate phase offsets, shape ``(resolution ** 2, n_channels)``, with
        channel 0 pinned to zero.
    """
    axis = torch.linspace(
        0.0, 2.0 * torch.pi, resolution + 1, dtype=torch.float32, device=device
    )[:-1]
    first, second = torch.meshgrid(axis, axis, indexing='ij')
    candidates = torch.zeros(
        (axis.numel() ** 2, n_channels), dtype=torch.float32, device=device
    )
    if n_channels > 1:
        candidates[:, 1] = first.reshape(-1)
    if n_channels > 2:
        candidates[:, 2] = second.reshape(-1)
    return candidates


def _decode_batched(
    signals: torch.Tensor,
    phases: torch.Tensor,
    *,
    grid: torch.Tensor,
    wavelengths: torch.Tensor,
    grid_step: float,
    max_step: int,
    smooth_weight: float,
    chunk_elements: int,
) -> _Decoded:
    """Viterbi-decode a batch and return its total path cost and path.

    The cost tensor is ``(rows, n_grid, n_time)`` float32 and the decoder
    transiently needs several multiples of it, so rows are processed in
    memory-bounded chunks. Rows never interact, so chunking cannot change the
    result -- it only bounds peak memory.

    Args:
        signals: Amplitude-normalized signals, ``(rows, n_ch, n_time)``.
        phases: Per-row phase offsets, ``(rows, n_ch)``.
        grid: Candidate displacements (microns).
        wavelengths: Channel wavelengths (microns).
        grid_step: Spacing between grid points (microns).
        max_step: Maximum grid points traversed per time sample.
        smooth_weight: Weight on the squared-step penalty.
        chunk_elements: Soft cap on ``rows * n_grid * n_time`` per call.

    Returns:
        Tuple of total path cost per row and grid-index path per row.
    """
    per_row = max(1, grid.numel() * signals.shape[2])
    chunk = max(1, int(chunk_elements // per_row))
    costs: list[torch.Tensor] = []
    paths: list[torch.Tensor] = []
    for begin in range(0, signals.shape[0], chunk):
        stop = begin + chunk
        cost = _channel_costs(
            signals[begin:stop], grid, wavelengths, phases[begin:stop]
        )
        path = _viterbi(cost, grid_step, max_step, smooth_weight)
        data = cost.gather(1, path[:, None, :]).squeeze(1).sum(dim=1)
        step = (path[:, 1:] - path[:, :-1]).to(cost.dtype) * grid_step
        costs.append(data + smooth_weight * (step**2).sum(dim=1))
        paths.append(path)
        del cost
    return torch.cat(costs), torch.cat(paths)


@dataclass
class InverseFitResult:
    """Displacement and velocity recovered from photodiode signals alone.

    Attributes:
        displacement: Estimated displacement (microns), shape ``(batch, n_time)``,
            zero-mean (the absolute offset is unobservable from fringes).
        velocity: Estimated velocity (microns/second), same shape.
        phases: Per-shot per-channel phase offsets chosen by the search
            (radians), shape ``(batch, n_channels)``.
    """

    displacement: np.ndarray
    velocity: np.ndarray
    phases: np.ndarray


def inverse_fit(
    pd_signals: np.ndarray,
    sample_rate: float,
    wavelengths_um: np.ndarray | list[float],
    displacement_limit_um: float,
    grid_points: int = 1201,
    max_speed_um_per_s: float = 4000.0,
    smooth_scale: float = 0.5,
    phase_grid_points: int = 16,
    phase_search_samples: int = 1024,
    phase_refine_candidates: int = 8,
    velocity_smooth_window: int = 251,
    decode_chunk_elements: int = 200_000_000,
    device: str | torch.device = 'cpu',
) -> InverseFitResult:
    """Method 2: recover displacement from the photodiode signals alone.

    The drive voltage is deliberately *not* used. Each wavelength wraps its
    fringe phase at a different rate, so the three channels taken together
    pin down the displacement modulo a much longer synthetic beat period than
    any single channel could. Concretely, for each candidate displacement on a
    grid we score how well all three ``cos`` templates match the measured
    samples, then pick the time trajectory minimizing total data cost plus a
    smoothness penalty (Viterbi). The smoothness term is essential: it is what
    lets the decoder walk through velocity sign reversals, which is exactly
    where naive per-channel Hilbert phase unwrapping fails.

    The per-channel phase offsets are unknown per shot. They are found by a
    coarse grid search on the two *relative* phases (the third is degenerate
    with the unobservable displacement offset), scored by the Viterbi path
    cost. Using the smoothness-constrained cost rather than the per-sample
    minimum is what makes this search work; the per-sample minimum is happy to
    let the trajectory jump around and so cannot rank phases. The search runs
    on a short prefix to shortlist candidates, then rescores the shortlist on
    the full shot, because a short prefix ranks phases too noisily to trust
    outright.

    Velocity is obtained by central-difference differentiation of the decoded
    displacement.

    Args:
        pd_signals: Photodiode signals, shape ``(batch, n_channels, n_time)``
            or ``(n_channels, n_time)`` for a single shot.
        sample_rate: Acquisition sample rate (Hz).
        wavelengths_um: Channel wavelengths (microns), length ``n_channels``.
        displacement_limit_um: Half-width of the displacement search grid
            (microns); the grid spans ``[-limit, +limit]``.
        grid_points: Number of displacement grid points.
        max_speed_um_per_s: Largest mirror speed the decoder will allow
            (microns/second). Sets the Viterbi band width.
        smooth_scale: Dimensionless weight on the smoothness penalty relative
            to the maximum allowed per-sample step.
        phase_grid_points: Grid resolution per relative-phase dimension. The
            coarse search evaluates ``phase_grid_points ** 2`` candidates.
        phase_search_samples: Number of leading time samples used to shortlist
            phase candidates. Keeps the coarse search cheap.
        phase_refine_candidates: How many shortlisted candidates are rescored
            on the full shot before picking a winner.
        velocity_smooth_window: Savitzky-Golay window (samples) used for the
            derivative. At 488 kHz, 251 samples is 0.5 ms -- far shorter than
            a period of the 1 kHz upper drive frequency, so it suppresses
            grid-quantization noise without attenuating the signal.
        decode_chunk_elements: Soft cap on ``rows * grid_points * n_time``
            per Viterbi call, to bound peak memory. Does not change results.
        device: Torch device for the decode. ``'cuda:0'`` is much faster for
            batches.

    Returns:
        The decoded displacement, velocity and per-shot phases.

    Raises:
        ValueError: If the signal array is not 2-D or 3-D, or the number of
            channels does not match the wavelength list.
    """
    signals = np.asarray(pd_signals, dtype=np.float64)
    if signals.ndim == 2:
        signals = signals[None, ...]
    if signals.ndim != 3:
        raise ValueError(f'pd_signals must be 2-D or 3-D, got shape {signals.shape}')
    wavelengths = np.asarray(wavelengths_um, dtype=np.float64)
    if signals.shape[1] != wavelengths.size:
        raise ValueError(
            f'got {signals.shape[1]} channels but {wavelengths.size} wavelengths'
        )

    batch, n_channels, n_time = signals.shape
    normalized = _normalize_fringe_amplitude(signals)

    torch_device = torch.device(device)
    sig_t = torch.tensor(normalized, dtype=torch.float32, device=torch_device)
    wl_t = torch.tensor(wavelengths, dtype=torch.float32, device=torch_device)
    grid = torch.linspace(
        -displacement_limit_um,
        displacement_limit_um,
        grid_points,
        dtype=torch.float32,
        device=torch_device,
    )
    grid_step = float(grid[1] - grid[0])
    max_step = max(1, int(np.ceil(max_speed_um_per_s / sample_rate / grid_step)))
    smooth_weight = smooth_scale / (grid_step * max_step) ** 2

    def decode(signals_t: torch.Tensor, phases: torch.Tensor) -> _Decoded:
        return _decode_batched(
            signals_t,
            phases,
            grid=grid,
            wavelengths=wl_t,
            grid_step=grid_step,
            max_step=max_step,
            smooth_weight=smooth_weight,
            chunk_elements=decode_chunk_elements,
        )

    # --- coarse phase search on a short prefix ----------------------------
    prefix = min(phase_search_samples, n_time)
    candidates = _phase_candidates(phase_grid_points, n_channels, torch_device)
    n_cand = candidates.shape[0]

    # (batch * n_cand, ...): every shot paired with every phase candidate, so
    # the whole search is a handful of kernel launches, not one per candidate.
    prefix_signals = sig_t[:, :, :prefix].repeat_interleave(n_cand, dim=0)
    coarse_cost, _ = decode(prefix_signals, candidates.repeat(batch, 1))
    coarse_cost = coarse_cost.view(batch, n_cand)

    # --- rescore the best candidates on the full shot ---------------------
    # The prefix is short enough that its ranking is noisy, so the prefix is
    # used only to shortlist. The winner is decided by full-shot cost, which
    # is the quantity we actually care about minimizing.
    keep = min(phase_refine_candidates, n_cand)
    shortlist = coarse_cost.topk(keep, dim=1, largest=False).indices  # (batch, keep)
    full_signals = sig_t.repeat_interleave(keep, dim=0)
    full_phases = candidates[shortlist.reshape(-1)]
    full_cost, full_path = decode(full_signals, full_phases)
    full_cost = full_cost.view(batch, keep)
    winner = full_cost.argmin(dim=1)

    rows = torch.arange(batch, device=torch_device)
    flat = rows * keep + winner
    path = full_path[flat]
    best_phases = full_phases[flat]

    displacement = grid[path]
    displacement = displacement - displacement.mean(dim=1, keepdim=True)

    disp_np = displacement.detach().cpu().numpy().astype(np.float64)

    # The decoded displacement is quantized to the grid, so a bare finite
    # difference at 488 kHz differentiates a staircase and buries the answer
    # in quantization noise. A Savitzky-Golay derivative fits a local cubic
    # and differentiates that instead. The window is long in samples but
    # short in time (see velocity_smooth_window), and the speaker is driven
    # only to 1 kHz, so this removes quantization noise rather than signal.
    window = min(velocity_smooth_window, n_time - (1 - n_time % 2))
    if window >= 5:
        velocity_np = savgol_filter(
            disp_np, window, polyorder=3, deriv=1, delta=1.0 / sample_rate, axis=1
        )
    else:
        velocity_np = np.gradient(disp_np, 1.0 / sample_rate, axis=1)

    return InverseFitResult(
        displacement=disp_np,
        velocity=velocity_np,
        phases=best_phases.detach().cpu().numpy().astype(np.float64),
    )


# ---------------------------------------------------------------------------
# Evaluation over real acquired data
# ---------------------------------------------------------------------------


@dataclass
class DatasetBaseline:
    """Aggregated baseline numbers for one acquired dataset.

    Attributes:
        dataset: Short dataset label, e.g. ``'free-space'``.
        file: File name the numbers came from.
        n_shots: Number of shots evaluated.
        n_samples_per_shot: Time samples per shot.
        sample_rate_hz: Acquisition sample rate (Hz).
        method1: Per-channel forward-model residual statistics (volts).
        method2: Inverse-fit displacement/velocity error statistics.
        signal_scale: Reference scales used to make the RMSE numbers readable.
        runtime_s: Wall-clock seconds spent evaluating this dataset.
    """

    dataset: str
    file: str
    n_shots: int
    n_samples_per_shot: int
    sample_rate_hz: float
    method1: dict[str, Any] = field(default_factory=dict)
    method2: dict[str, Any] = field(default_factory=dict)
    signal_scale: dict[str, Any] = field(default_factory=dict)
    runtime_s: float = 0.0


def _summary(values: np.ndarray) -> dict[str, float]:
    """Mean/median/quartile summary of a 1-D array of per-shot scores."""
    values = np.asarray(values, dtype=np.float64)
    return {
        'mean': float(values.mean()),
        'median': float(np.median(values)),
        'p25': float(np.percentile(values, 25)),
        'p75': float(np.percentile(values, 75)),
        'min': float(values.min()),
        'max': float(values.max()),
    }


def evaluate_dataset(
    file_path: Path,
    label: str,
    n_shots: int = 200,
    batch_size: int = 16,
    device: str = 'cpu',
    grid_points: int = 1201,
    phase_grid_points: int = 16,
) -> DatasetBaseline:
    """Run both baseline methods over a subset of one acquired HDF5 file.

    Shots are read as a contiguous leading slice; the files are 0.66 GB per
    channel, so the whole array is never materialized.

    Args:
        file_path: Path to the acquisition HDF5 file.
        label: Short label recorded in the result.
        n_shots: Number of shots to evaluate.
        batch_size: Shots decoded per Viterbi batch. Larger amortizes GPU
            kernel launch overhead; memory scales with
            ``batch_size * grid_points * n_time``.
        device: Torch device for the method-2 decode.
        grid_points: Displacement grid resolution for method 2.
        phase_grid_points: Phase search resolution for method 2.

    Returns:
        Aggregated statistics for this dataset.
    """
    start = time.perf_counter()
    coil = CoilDriver()
    channels = list(PD_WAVELENGTHS_UM)
    wavelengths = np.array([PD_WAVELENGTHS_UM[c] for c in channels])

    with h5py.File(file_path, 'r') as handle:
        sample_rate = float(handle.attrs['sample_rate'])
        n_shots = min(n_shots, handle[DRIVE_KEY].shape[0])
        n_time = int(handle[DRIVE_KEY].shape[1])
        drive = np.asarray(handle[DRIVE_KEY][:n_shots], dtype=np.float64)
        pd_raw = np.stack(
            [np.asarray(handle[c][:n_shots], dtype=np.float64) for c in channels],
            axis=1,
        )

    # Ground truth from the drive voltage.
    true_disp = np.empty((n_shots, n_time))
    true_vel = np.empty((n_shots, n_time))
    for i in range(n_shots):
        true_disp[i], _, _ = coil.get_displacement(drive[i], sample_rate)
        true_vel[i], _, _ = coil.get_velocity(drive[i], sample_rate)
    true_disp_centered = true_disp - true_disp.mean(axis=1, keepdims=True)

    # --- Method 1 ---------------------------------------------------------
    m1: dict[str, Any] = {}
    for k, channel in enumerate(channels):
        rmses = np.empty(n_shots)
        r2s = np.empty(n_shots)
        rel = np.empty(n_shots)
        for i in range(n_shots):
            fit = forward_model_residual(pd_raw[i, k], true_disp[i], wavelengths[k])
            rmses[i] = fit.rmse
            r2s[i] = fit.r_squared
            signal_rms = float(np.std(pd_raw[i, k]))
            rel[i] = fit.rmse / signal_rms if signal_rms > 0 else np.nan
        m1[channel] = {
            'wavelength_um': float(wavelengths[k]),
            'residual_rmse_volts': _summary(rmses),
            'residual_rmse_fraction_of_signal_rms': _summary(rel),
            'r_squared': _summary(r2s),
        }

    # --- Method 2 ---------------------------------------------------------
    # The search grid must cover the true excursion with margin; use the
    # observed ground-truth range for this subset.
    limit = float(np.abs(true_disp_centered).max()) * 1.15
    max_speed = float(np.abs(true_vel).max()) * 1.25

    disp_rmse = np.empty(n_shots)
    vel_rmse = np.empty(n_shots)
    disp_corr = np.empty(n_shots)
    for begin in range(0, n_shots, batch_size):
        end = min(begin + batch_size, n_shots)
        result = inverse_fit(
            pd_raw[begin:end],
            sample_rate=sample_rate,
            wavelengths_um=wavelengths,
            displacement_limit_um=limit,
            grid_points=grid_points,
            max_speed_um_per_s=max_speed,
            phase_grid_points=phase_grid_points,
            device=device,
        )
        for j, i in enumerate(range(begin, end)):
            # Fringe sign is not observable: cos is even, so d(t) and -d(t)
            # produce identical signals. Score against whichever sign fits
            # better rather than penalizing an unobservable degree of freedom.
            candidates = [
                (result.displacement[j], result.velocity[j]),
                (-result.displacement[j], -result.velocity[j]),
            ]
            errors = [
                float(np.sqrt(np.mean((d - true_disp_centered[i]) ** 2)))
                for d, _ in candidates
            ]
            pick = int(np.argmin(errors))
            est_d, est_v = candidates[pick]
            disp_rmse[i] = errors[pick]
            vel_rmse[i] = float(np.sqrt(np.mean((est_v - true_vel[i]) ** 2)))
            denom = float(np.std(est_d) * np.std(true_disp_centered[i]))
            disp_corr[i] = (
                float(np.mean(est_d * true_disp_centered[i]) / denom)
                if denom > 0
                else 0.0
            )

    return DatasetBaseline(
        dataset=label,
        file=file_path.name,
        n_shots=int(n_shots),
        n_samples_per_shot=n_time,
        sample_rate_hz=sample_rate,
        method1=m1,
        method2={
            'displacement_rmse_um': _summary(disp_rmse),
            'velocity_rmse_um_per_s': _summary(vel_rmse),
            'displacement_correlation': _summary(disp_corr),
            'displacement_nrmse': _summary(
                disp_rmse / np.std(true_disp_centered, axis=1)
            ),
            'velocity_nrmse': _summary(vel_rmse / np.std(true_vel, axis=1)),
        },
        signal_scale={
            'true_displacement_rms_um': _summary(np.std(true_disp_centered, axis=1)),
            'true_velocity_rms_um_per_s': _summary(np.std(true_vel, axis=1)),
            'true_displacement_ptp_um': _summary(np.ptp(true_disp_centered, axis=1)),
        },
        runtime_s=time.perf_counter() - start,
    )


def run_baselines(
    n_shots: int = 200,
    device: str = 'cpu',
    batch_size: int = 16,
    output: Path | None = None,
    datasets: dict[str, str] | None = None,
    grid_points: int = 1201,
    phase_grid_points: int = 16,
) -> dict[str, Any]:
    """Evaluate both baselines on every configured dataset and write JSON.

    Args:
        n_shots: Shots per dataset.
        device: Torch device for the method-2 decode.
        batch_size: Shots per decode batch.
        output: Destination JSON path. Defaults to
            ``smi/analysis/data/baseline_results.json``.
        datasets: Mapping of label to file name under ``smi/analysis/data``.
        grid_points: Displacement grid resolution.
        phase_grid_points: Phase search resolution.

    Returns:
        The results dictionary that was written to disk.
    """
    datasets = datasets or DEFAULT_DATASETS
    output = output or DEFAULT_RESULTS_PATH

    results: dict[str, Any] = {
        'schema_version': 1,
        'config': {
            'n_shots': n_shots,
            'device': device,
            'batch_size': batch_size,
            'grid_points': grid_points,
            'phase_grid_points': phase_grid_points,
            'pd_wavelengths_um': dict(PD_WAVELENGTHS_UM),
        },
        'datasets': {},
    }
    for label, name in datasets.items():
        path = DATA_DIR / name
        if not path.exists():
            logger.warning('skipping %s: %s not found', label, path)
            continue
        logger.info('evaluating %s (%s)...', label, name)
        baseline = evaluate_dataset(
            path,
            label,
            n_shots=n_shots,
            batch_size=batch_size,
            device=device,
            grid_points=grid_points,
            phase_grid_points=phase_grid_points,
        )
        results['datasets'][label] = asdict(baseline)
        m2 = baseline.method2
        logger.info(
            '  %s: method-2 median displacement RMSE %.4f um, '
            'median velocity RMSE %.1f um/s (%.1f s)',
            label,
            m2['displacement_rmse_um']['median'],
            m2['velocity_rmse_um_per_s']['median'],
            baseline.runtime_s,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    logger.info('wrote %s', output)
    return results


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point for regenerating the baseline artifact."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--n-shots', type=int, default=200)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--grid-points', type=int, default=1201)
    parser.add_argument('--phase-grid-points', type=int, default=16)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    run_baselines(
        n_shots=args.n_shots,
        device=args.device,
        batch_size=args.batch_size,
        output=args.output,
        grid_points=args.grid_points,
        phase_grid_points=args.phase_grid_points,
    )


if __name__ == '__main__':
    main()
