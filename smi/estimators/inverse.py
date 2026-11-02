"""Method 2: displacement recovered from the photodiode signals alone.

Multi-wavelength decoding over a displacement grid by a batched Viterbi
search, then differentiation for velocity. Arrays in, arrays out; see
:mod:`smi.experiments.baselines` for how it is scored against the drive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.signal import savgol_filter

from smi.theory.coil_driver import MS_PER_S

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
        velocity: Estimated velocity (microns/ms), same shape.
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
    max_speed_um_per_ms: float = 4.0,
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
        max_speed_um_per_ms: Largest mirror speed the decoder will allow
            (microns per millisecond). Sets the Viterbi band width.
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
    # sample_rate is in Hz, so the speed must be per-second to divide into it
    # and yield microns per sample; max_speed is per-millisecond.
    max_speed_um_per_s = max_speed_um_per_ms * MS_PER_S
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
    # savgol_filter requires an odd window. The min() bounds the length
    # against short shots, but an even velocity_smooth_window would survive it
    # and raise inside scipy, so force parity on the result rather than only
    # on the bound.
    window = min(velocity_smooth_window, n_time - (1 - n_time % 2))
    window -= 1 - window % 2
    # The time step is in milliseconds, so differentiating displacement in um
    # yields um/ms -- the unit CoilDriver emits and the network is scored in.
    # A per-second step here would make the returned velocity, and hence the
    # RMSE against the ground truth, 1000x too large.
    dt_ms = MS_PER_S / sample_rate
    if window >= 5:
        velocity_np = savgol_filter(
            disp_np, window, polyorder=3, deriv=1, delta=dt_ms, axis=1
        )
    else:
        velocity_np = np.gradient(disp_np, dt_ms, axis=1)

    return InverseFitResult(
        displacement=disp_np,
        velocity=velocity_np,
        phases=best_phases.detach().cpu().numpy().astype(np.float64),
    )
