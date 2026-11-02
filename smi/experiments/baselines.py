#!/usr/bin/env python3
"""Non-deep-learning baselines for displacement/velocity estimation.

This module answers the "establishing baselines" question: before claiming a
neural network is doing something useful, how well does classical, parametric
interferometry do on the same data?

Two distinct measurements live here, and it matters which is which.

**Method 1 -- forward-model residual** (:func:`forward_model_residual`).
Question answered: *does the Michelson forward model explain this data at
all?* We take the displacement implied by the known speaker drive voltage
(via :class:`~smi.theory.coil_driver.CoilDriver`), push it through
``cos(4*pi/lambda * d(t) + phi)``, and fit only the per-shot phase offset and
amplitude. Because the model is linear in its ``cos``/``sin`` components once
the displacement is fixed, this is a closed-form least-squares -- there is no
iterative optimizer and no local minimum to worry about. The reported number
is a residual RMSE on the photodiode signal, in volts. This is the honest
floor: no inverse method can beat the forward model's own ability to describe
the measurement.

One global ``(cos, sin, 1)`` fit over the whole 33.6 ms record is however a
*rigid* estimator: real shots drift by 1-2.7 fringes within a record from
speckle and path drift, and a single phase cannot absorb that, so the global
R^2 understates the model's agreement with the data.
:func:`windowed_forward_model_residual` refits per window of W samples and
pools the variances into one comparable R^2 per shot. Both are reported. A
windowed R^2 must always be quoted beside :func:`scrambled_theta_control`,
which does the identical fit against a *different* shot's displacement at the
*same* W, and so measures the windowing's fitting capacity on its own.

W itself is fringe-matched (:func:`fringe_matched_window`): it is resolved per
shot per channel so every window spans about two fringes, because a fixed W
holds a third of a fringe on the quietest shots and a dozen on the loudest and
so does not mean the same thing across excursion bins. The artifact the
baseline report reads is generated this way, so one estimator backs every
table and figure in it.

**Method 2 -- inverse fit** (:func:`inverse_fit`).
Question answered: *how well can a classical estimator recover displacement
and velocity from the photodiode signals alone?* Only the three PD channels
are used; the drive voltage is held out and used purely as ground truth for
scoring. Displacement is recovered by multi-wavelength decoding across the
635 / 675 / 515 nm channels, then differentiated for velocity. The reported
numbers are displacement RMSE in microns and velocity RMSE in microns per
millisecond -- the same units the network is scored in. **This is the number
that belongs beside the network's RMSE.**

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
in microns, velocity in microns per millisecond (um/ms), time in seconds,
photodiode signals in volts. Velocity is um/ms because that is what
:class:`~smi.theory.coil_driver.CoilDriver` emits and what the network is
trained and scored on; sample rates remain in Hz.

The two methods deliberately run at different shot counts: method 1 is
milliseconds per shot and runs at N = 1000, while method 2's GPU Viterbi
decode is ~5.85 s per shot and stays at N = 200. Both are recorded in the
artifact's ``config`` block; never quote one N over the other's table.

Run as a script to regenerate the results artifact -- the defaults are the
settings it was produced with::

    pixi run -e dev python scripts/run_baselines.py
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from smi.core.metrics import summarize_per_shot
from smi.core.paths import DATA_DIR
from smi.data.schema import DRIVE_KEY, PD_WAVELENGTHS_UM, SAMPLE_RATE_ATTR
from smi.estimators.inverse import inverse_fit
from smi.estimators.phase_fit import (
    DEFAULT_FRINGES_PER_WINDOW,
    DEFAULT_WINDOW_SAMPLES,
    MAX_FRINGE_WINDOW,
    MIN_FRINGE_WINDOW,
    binned_by_amplitude,
    forward_model_residual,
    fringe_matched_window,
    scrambled_theta_control,
    windowed_forward_model_residual,
)
from smi.theory.coil_driver import CoilDriver

logger = logging.getLogger(__name__)


# Default location of the committed artifacts.


# ---------------------------------------------------------------------------
# Evaluation over real acquired data
# ---------------------------------------------------------------------------


@dataclass
class DatasetBaseline:
    """Aggregated baseline numbers for one acquired dataset.

    Attributes:
        dataset: Short dataset label, e.g. ``'free-space'``.
        file: File name the numbers came from.
        n_shots: Number of shots method 1 evaluated.
        n_shots_method2: Number of shots method 2 evaluated. Lower than
            ``n_shots`` on purpose -- see :func:`evaluate_dataset`.
        n_samples_per_shot: Time samples per shot.
        sample_rate_hz: Acquisition sample rate (Hz).
        method1: Per-channel forward-model residual statistics (volts).
        method2: Inverse-fit displacement/velocity error statistics.
        signal_scale: Reference scales used to make the RMSE numbers readable.
        decode_grid: Decode grid extent and speed band, derived per dataset at
            run time and therefore not recoverable from the static config.
        runtime_s: Wall-clock seconds spent evaluating this dataset.
    """

    dataset: str
    file: str
    n_shots: int
    n_shots_method2: int
    n_samples_per_shot: int
    sample_rate_hz: float
    method1: dict[str, Any] = field(default_factory=dict)
    method2: dict[str, Any] = field(default_factory=dict)
    signal_scale: dict[str, Any] = field(default_factory=dict)
    decode_grid: dict[str, float] = field(default_factory=dict)
    runtime_s: float = 0.0


# How the windowed measure picks its window length. 'fringe-matched' resolves
# W per shot per channel from that shot's own fringe rate, which is what the
# report's artifact uses so every table in it means one estimator. 'fixed'
# holds W at ``window_samples``, which is what the known-answer tests want:
# their fringe rate is synthetic, so matching a window to it measures nothing.
WINDOWING_MODES = frozenset({'fringe-matched', 'fixed'})


def _method1_channel(
    signals: np.ndarray,
    true_disp: np.ndarray,
    scramble: np.ndarray,
    wavelength_um: float,
    ptp_um: np.ndarray,
    windowing: str,
    window_samples: int | None,
) -> dict[str, Any]:
    """Method-1 statistics for one channel over every shot.

    Args:
        signals: Photodiode samples for this channel, shape ``(n_shots, n)``.
        true_disp: Drive-implied displacement per shot (um), same shape.
        scramble: Shot index each shot borrows a control displacement from.
        wavelength_um: This channel's laser wavelength (microns).
        ptp_um: Per-shot displacement peak-to-peak, for the excursion bins.
        windowing: ``'fringe-matched'`` or ``'fixed'``.
        window_samples: Window length used when ``windowing='fixed'``.

    Returns:
        The per-channel block written into the artifact.
    """
    n_shots, n_time = signals.shape
    rmses = np.empty(n_shots)
    r2s = np.empty(n_shots)
    rel = np.empty(n_shots)
    windowed_r2s = np.empty(n_shots)
    control_r2s = np.empty(n_shots)
    resolved_w = np.empty(n_shots, dtype=np.int64)
    for i in range(n_shots):
        fit = forward_model_residual(signals[i], true_disp[i], wavelength_um)
        rmses[i] = fit.rmse
        r2s[i] = fit.r_squared
        signal_rms = float(np.std(signals[i]))
        rel[i] = fit.rmse / signal_rms if signal_rms > 0 else np.nan
        # One W per shot per channel, resolved from *this* shot's own
        # displacement and this channel's wavelength: a shot's fringe rate
        # differs by 31% between 515 and 675 nm, so a per-shot W shared
        # across channels would be fringe-matched for none of them.
        shot_w = (
            fringe_matched_window(true_disp[i], wavelength_um)
            if windowing == 'fringe-matched'
            else window_samples
        )
        resolved_w[i] = shot_w if shot_w is not None else n_time
        windowed_r2s[i] = windowed_forward_model_residual(
            signals[i], true_disp[i], wavelength_um, window_samples=shot_w
        ).r_squared
        # The control gets the *same* W as the fit it controls, resolved from
        # this shot rather than from the displacement it borrows. Scoring a
        # fit against a control that had a different amount of fitting
        # freedom measures the freedom, not the agreement.
        control_r2s[i] = scrambled_theta_control(
            signals[i], true_disp[scramble[i]], wavelength_um, window_samples=shot_w
        ).r_squared

    return {
        'wavelength_um': wavelength_um,
        # What window the numbers below came from. A fringe-matched W is per
        # shot, so no single number states it: the median and the range are
        # what let a reader see the window without holding the per-shot array.
        'window_samples_resolved': {
            'median': float(np.median(resolved_w)),
            'min': int(resolved_w.min()),
            'max': int(resolved_w.max()),
            'p25': float(np.percentile(resolved_w, 25)),
            'p75': float(np.percentile(resolved_w, 75)),
        },
        'residual_rmse_volts': summarize_per_shot(rmses),
        'residual_rmse_fraction_of_signal_rms': summarize_per_shot(rel),
        'r_squared': summarize_per_shot(r2s),
        # The windowed refit measures model/data agreement and the global
        # single-window fit is the contrast that motivates windowing, so both
        # are reported side by side. The control is not optional -- short
        # windows can fit almost anything, so a windowed R^2 reported without
        # it is not evidence.
        'r_squared_windowed': summarize_per_shot(windowed_r2s),
        'r_squared_windowed_scrambled': summarize_per_shot(control_r2s),
        # Binned by drive excursion because the aggregate R^2 hides the trend
        # that matters: the forward model tracks small excursions better than
        # large ones. Persisted rather than left to an ad-hoc session, so the
        # claim can be checked against the artifact instead of taken on trust.
        'r_squared_by_drive_ptp': binned_by_amplitude(r2s, ptp_um),
        'r_squared_windowed_by_drive_ptp': binned_by_amplitude(windowed_r2s, ptp_um),
        'r_squared_windowed_scrambled_by_drive_ptp': binned_by_amplitude(
            control_r2s, ptp_um
        ),
    }


def evaluate_dataset(
    file_path: Path,
    label: str,
    n_shots: int = 1000,
    n_shots_method2: int = 200,
    batch_size: int = 16,
    device: str = 'cpu',
    grid_points: int = 1201,
    phase_grid_points: int = 16,
    window_samples: int | None = DEFAULT_WINDOW_SAMPLES,
    windowing: str = 'fringe-matched',
) -> DatasetBaseline:
    """Run both baseline methods over a subset of one acquired HDF5 file.

    Shots are read as a contiguous leading slice; the files are 0.66 GB per
    channel, so the whole array is never materialized.

    **The two methods run at different shot counts on purpose.** Method 1 is a
    handful of least-squares solves per shot, about a millisecond each, so it
    can afford N = 1000 -- and it needs it, because that is where the
    large-excursion claims live and the extreme bins are otherwise too thin to
    carry a trend. Method 2 is a sequential GPU Viterbi decode at ~5.85 s per
    shot; N = 1000 would cost ~3.3 hours across both acquisitions to put a
    tighter error bar on displacement and velocity RMSE numbers that nobody
    disputes. Both counts are recorded so a reader can tell which N produced
    which table.

    **The windowed measure is fringe-matched by default.** A fixed ``W``
    spans a third of a fringe on the quietest shots and a dozen on the
    loudest, so one window's worth of phase error is not comparable between
    them and the measure means something different in each excursion bin.
    :func:`fringe_matched_window` scales ``W`` by each shot's own fringe rate
    instead, so every window holds about the same number of fringes. Pass
    ``windowing='fixed'`` for the fixed-``W`` behaviour, which is what the
    known-answer tests want: their fringe rate is synthetic, so matching a
    window to it measures nothing.

    Args:
        file_path: Path to the acquisition HDF5 file.
        label: Short label recorded in the result.
        n_shots: Number of shots for method 1.
        n_shots_method2: Number of shots for method 2, capped at ``n_shots``.
        batch_size: Shots decoded per Viterbi batch. Larger amortizes GPU
            kernel launch overhead; memory scales with
            ``batch_size * grid_points * n_time``.
        device: Torch device for the method-2 decode.
        grid_points: Displacement grid resolution for method 2.
        phase_grid_points: Phase search resolution for method 2.
        window_samples: Window length for method 1's windowed refit, used
            only when ``windowing='fixed'``.
        windowing: ``'fringe-matched'`` to resolve ``W`` per shot per channel
            from that shot's own fringe rate, or ``'fixed'`` to use
            ``window_samples`` for every shot.

    Returns:
        Aggregated statistics for this dataset.

    Raises:
        ValueError: If fewer than two shots are available, which leaves the
            scrambled-theta control no other shot to borrow a displacement
            from, or if ``windowing`` is not one of the two accepted values.
    """
    if windowing not in WINDOWING_MODES:
        raise ValueError(
            f'windowing must be one of {sorted(WINDOWING_MODES)}, got {windowing!r}'
        )
    start = time.perf_counter()
    coil = CoilDriver()
    channels = list(PD_WAVELENGTHS_UM)
    wavelengths = np.array([PD_WAVELENGTHS_UM[c] for c in channels])

    with h5py.File(file_path, 'r') as handle:
        sample_rate = float(handle.attrs[SAMPLE_RATE_ATTR])
        n_shots = min(n_shots, handle[DRIVE_KEY].shape[0])
        n_shots_method2 = min(n_shots_method2, n_shots)
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
        # One shared spectrum: get_displacement and get_velocity each call
        # get_displacement_spectrum internally, so calling both computes the
        # same FFT twice.
        true_vel[i], true_disp[i] = coil.get_velocity_and_displacement(
            drive[i], sample_rate
        )
    true_disp_centered = true_disp - true_disp.mean(axis=1, keepdims=True)

    # --- Method 1 ---------------------------------------------------------
    # The scrambled-theta control pairs each shot's signal with the next
    # shot's displacement (the last wraps to the first). A fixed rotation
    # rather than a random pairing keeps the artifact reproducible, and a
    # neighbouring shot comes from the same acquisition conditions, which is
    # what makes it the right control: it differs from the real pairing only
    # in that the displacement no longer belongs to this signal.
    #
    # With a single shot the rotation maps it onto itself, so the "control"
    # would be the real fit under a different name -- a low number that looks
    # like evidence of no overfitting and is nothing of the kind. Refuse
    # rather than emit it.
    if n_shots < 2:
        raise ValueError(
            f'the scrambled-theta control needs at least 2 shots to borrow a '
            f'displacement from, got {n_shots}'
        )
    scramble = np.roll(np.arange(n_shots), -1)

    ptp_um = np.ptp(true_disp_centered, axis=1)
    m1: dict[str, Any] = {
        channel: _method1_channel(
            pd_raw[:, k],
            true_disp,
            scramble,
            float(wavelengths[k]),
            ptp_um,
            windowing,
            window_samples,
        )
        for k, channel in enumerate(channels)
    }

    # --- Method 2 ---------------------------------------------------------
    # The search grid must cover the true excursion with margin; use the
    # observed ground-truth range for this subset.
    # Derived from the shots method 2 actually scores, not from method 1's
    # larger subset: a grid wider than this subset needs would waste decode
    # time, and a speed band set by shots outside the subset could not be
    # reproduced from the numbers the artifact records.
    m2_disp = true_disp_centered[:n_shots_method2]
    m2_vel = true_vel[:n_shots_method2]
    limit = float(np.abs(m2_disp).max()) * 1.15
    max_speed = float(np.abs(m2_vel).max()) * 1.25

    disp_rmse = np.empty(n_shots_method2)
    vel_rmse = np.empty(n_shots_method2)
    disp_corr = np.empty(n_shots_method2)
    for begin in range(0, n_shots_method2, batch_size):
        end = min(begin + batch_size, n_shots_method2)
        result = inverse_fit(
            pd_raw[begin:end],
            sample_rate=sample_rate,
            wavelengths_um=wavelengths,
            displacement_limit_um=limit,
            grid_points=grid_points,
            max_speed_um_per_ms=max_speed,
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
        n_shots_method2=int(n_shots_method2),
        n_samples_per_shot=n_time,
        sample_rate_hz=sample_rate,
        method1=m1,
        method2={
            'displacement_rmse_um': summarize_per_shot(disp_rmse),
            'velocity_rmse_um_per_ms': summarize_per_shot(vel_rmse),
            'displacement_correlation': summarize_per_shot(disp_corr),
            'displacement_nrmse': summarize_per_shot(
                disp_rmse / np.std(m2_disp, axis=1)
            ),
            'velocity_nrmse': summarize_per_shot(vel_rmse / np.std(m2_vel, axis=1)),
        },
        signal_scale={
            # Over method 1's N, which is the larger subset, so these scales
            # describe the cohort the R^2 tables are computed on.
            'true_displacement_rms_um': summarize_per_shot(
                np.std(true_disp_centered, axis=1)
            ),
            'true_velocity_rms_um_per_ms': summarize_per_shot(np.std(true_vel, axis=1)),
            'true_displacement_ptp_um': summarize_per_shot(
                np.ptp(true_disp_centered, axis=1)
            ),
        },
        # Derived from this subset's ground truth at run time, so they vary
        # with --n-shots and with the file. They set the decode grid's extent
        # and the Viterbi step band, which means a run cannot be reproduced
        # from the static config block alone -- record them per dataset.
        decode_grid={'displacement_limit_um': limit, 'max_speed_um_per_ms': max_speed},
        runtime_s=time.perf_counter() - start,
    )


def run_baselines(
    n_shots: int,
    n_shots_method2: int,
    device: str,
    batch_size: int,
    output: Path,
    datasets: dict[str, str],
    grid_points: int,
    phase_grid_points: int,
    window_samples: int | None,
    windowing: str,
) -> dict[str, Any]:
    """Evaluate both baselines on every configured dataset and write JSON.

    Args:
        n_shots: Shots per dataset for method 1.
        n_shots_method2: Shots per dataset for method 2, which is ~5.85 s per
            shot and so runs on a smaller subset (see
            :func:`evaluate_dataset`).
        device: Torch device for the method-2 decode.
        batch_size: Shots per decode batch.
        output: Destination JSON path.
        datasets: Mapping of label to file name under ``data/``.
        grid_points: Displacement grid resolution.
        phase_grid_points: Phase search resolution.
        window_samples: Window length for method 1's windowed refit, used
            only when ``windowing='fixed'``.
        windowing: ``'fringe-matched'`` or ``'fixed'``; see
            :func:`evaluate_dataset`.

    Returns:
        The results dictionary that was written to disk.
    """
    results: dict[str, Any] = {
        # 3: method 1 gained the windowed refit and its scrambled-theta
        # control, the excursion bins split 2+ into 2-3 and 3+, and the two
        # methods no longer share one shot count.
        # 4: the windowed measure and its control are fringe-matched per shot
        # per channel, so `window_samples` no longer describes them and each
        # channel carries a `window_samples_resolved` distribution instead.
        'schema_version': 4,
        'config': {
            'n_shots': n_shots,
            'n_shots_method2': n_shots_method2,
            # What the windowed measure is. A reader who sees only
            # `window_samples` would take the windowed columns for a
            # fixed-window measure; under 'fringe-matched' that field is
            # inactive and the per-channel `window_samples_resolved` block is
            # what states the window the numbers came from.
            'windowing': windowing,
            'window_samples': window_samples if windowing == 'fixed' else None,
            'fringes_per_window': (
                DEFAULT_FRINGES_PER_WINDOW if windowing == 'fringe-matched' else None
            ),
            'window_clip_samples': (
                [MIN_FRINGE_WINDOW, MAX_FRINGE_WINDOW]
                if windowing == 'fringe-matched'
                else None
            ),
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
            n_shots_method2=n_shots_method2,
            batch_size=batch_size,
            device=device,
            grid_points=grid_points,
            phase_grid_points=phase_grid_points,
            window_samples=window_samples,
            windowing=windowing,
        )
        results['datasets'][label] = asdict(baseline)
        m2 = baseline.method2
        for channel, stats in baseline.method1.items():
            # Log the windowed number with its control attached, so even the
            # console output cannot quote one without the other.
            logger.info(
                '  %s %s: method-1 median R^2 global %.3f, windowed %.3f '
                '(scrambled control %.3f) at median W = %.0f',
                label,
                channel,
                stats['r_squared']['median'],
                stats['r_squared_windowed']['median'],
                stats['r_squared_windowed_scrambled']['median'],
                stats['window_samples_resolved']['median'],
            )
        logger.info(
            '  %s: method-2 median displacement RMSE %.4f um, '
            'median velocity RMSE %.4f um/ms (%.1f s)',
            label,
            m2['displacement_rmse_um']['median'],
            m2['velocity_rmse_um_per_ms']['median'],
            baseline.runtime_s,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    logger.info('wrote %s', output)
    return results
