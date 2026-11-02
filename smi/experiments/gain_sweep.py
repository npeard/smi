"""Run the coil-driver gain sweep over the acquired datasets and write the artifact.

The estimator and its findings live in :mod:`smi.estimators.gain_sweep`; this
module only reads the HDF5 files, runs it, and writes
``gain_sweep_results.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from smi.data.schema import DRIVE_KEY, PD_WAVELENGTHS_UM, SAMPLE_RATE_ATTR
from smi.estimators.gain_sweep import (
    LEGACY_K,
    LIVE_K,
    DelayCurve,
    GainSweep,
    gain_grid,
    per_shot_best_gain,
    sweep_gain,
    unit_gain_displacement,
)
from smi.estimators.phase_fit import DEFAULT_WINDOW_SAMPLES

logger = logging.getLogger(__name__)


# Sweep range and step. The range brackets the live 14.553 generously on both
# sides and reaches the 2024 calibration's 33.42, so that value is refuted by
# the same curve rather than by a separate run. A 0.25 step is 1.7% of the
# live gain, so it resolves a peak to well inside the 3.2-6.5% the two
# acquisitions disagree by -- a coarser step could not have shown that
# disagreement is real rather than quantization.
DEFAULT_GAIN_LOW = 10.0
DEFAULT_GAIN_HIGH = 35.0
DEFAULT_GAIN_STEP = 0.25


def sweep_dataset(
    file_path: Path,
    label: str,
    n_shots: int = 1000,
    gains: np.ndarray | None = None,
    window_samples: int | None = DEFAULT_WINDOW_SAMPLES,
    delay_max_shift: int = 8,
    include_per_shot: bool = True,
) -> GainSweep:
    """Sweep the gain over a leading slice of one acquisition HDF5 file.

    N = 1000 by default, which is worth the cost: N = 200 left the extreme
    excursion bins at n = 14-72, too thin to carry the trend that separates a
    scale error from unmodeled drift, while N = 1000 gives n = 84-310 per bin.
    Measured cost with every option on (101 gains, windowed measure, delay
    check, per-shot fit) is ~9.3 min per acquisition on CPU; the global fit
    alone is ~3 min. Method 2's N = 200 in :mod:`smi.experiments.baselines` is
    a different budget for a ~5.85 s/shot GPU decode and must not be
    conflated with this one.

    Args:
        file_path: Path to the acquisition HDF5 file.
        label: Short label recorded in the result.
        n_shots: Shots to score, capped at the file's shot count.
        gains: Candidate gains. Defaults to :func:`gain_grid` over
            ``[DEFAULT_GAIN_LOW, DEFAULT_GAIN_HIGH]``, which brackets both the
            live constant and the 2024 calibration's value.
        window_samples: Window length for the windowed measure, or ``None``
            for the global fit only.
        delay_max_shift: Half-width of the delay check in samples, swept at
            the winning gain. Pass 0 to skip it.
        include_per_shot: Also fit a separate optimal gain per shot, which
            bounds what any scalar correction could achieve. Roughly doubles
            the global-fit work; see :func:`per_shot_best_gain`.

    Returns:
        The per-channel gain curves for this acquisition, plus the aliasing
        and time-delay checks that rule those two candidates out, and the
        per-shot upper bound when requested.
    """
    if gains is None:
        gains = gain_grid(DEFAULT_GAIN_LOW, DEFAULT_GAIN_HIGH, DEFAULT_GAIN_STEP)

    channels = list(PD_WAVELENGTHS_UM)
    wavelengths = [PD_WAVELENGTHS_UM[c] for c in channels]

    with h5py.File(file_path, 'r') as handle:
        sample_rate = float(handle.attrs[SAMPLE_RATE_ATTR])
        n_shots = min(n_shots, handle[DRIVE_KEY].shape[0])
        drive = np.asarray(handle[DRIVE_KEY][:n_shots], dtype=np.float64)
        pd_raw = np.stack(
            [np.asarray(handle[c][:n_shots], dtype=np.float64) for c in channels],
            axis=1,
        )

    # One k=1 evaluation per shot, reused for every gain in the grid. This is
    # the whole reason a 101-gain sweep costs minutes rather than hours.
    unit = np.empty_like(drive)
    unit_vel = np.empty_like(drive)
    for i in range(n_shots):
        unit_vel[i], unit[i] = unit_gain_displacement(drive[i], sample_rate)

    sweep = sweep_gain(
        pd_raw,
        unit,
        wavelengths,
        gains,
        window_samples=window_samples,
        channel_names=channels,
        dataset=label,
        file=file_path.name,
        unit_velocity=unit_vel,
        sample_rate=sample_rate,
        delay_max_shift=delay_max_shift,
    )
    if include_per_shot:
        sweep.per_shot = per_shot_best_gain(
            pd_raw, unit, wavelengths, gains, channel_names=channels, dataset=label
        )
    return sweep


def run_gain_sweep(
    n_shots: int,
    gains: np.ndarray,
    window_samples: int | None,
    output: Path,
    datasets: dict[str, str],
    delay_max_shift: int,
    include_per_shot: bool,
    data_dir: Path,
) -> dict[str, Any]:
    """Sweep every configured acquisition and write the JSON artifact.

    Args:
        n_shots: Shots per dataset.
        gains: Candidate gains.
        window_samples: Window length for the windowed measure.
        output: Destination JSON path.
        datasets: Mapping of label to file name under ``data_dir``.
        delay_max_shift: Half-width of the delay check in samples.
        include_per_shot: Also fit an optimal gain per shot, bounding what any
            scalar correction could achieve.
        data_dir: Directory the dataset file names are relative to.

    Returns:
        The results dictionary that was written to disk.
    """
    results: dict[str, Any] = {
        # 1: first committed gain sweep. Bump when the schema changes shape,
        # as baseline_results.json does, so a stale reader fails loudly.
        'schema_version': 1,
        'config': {
            'n_shots': n_shots,
            'window_samples': window_samples,
            'gain_low': float(gains[0]),
            'gain_high': float(gains[-1]),
            'n_gains': int(np.asarray(gains).size),
            'live_k': LIVE_K,
            'legacy_k': LEGACY_K,
            'delay_max_shift': delay_max_shift,
            'include_per_shot': include_per_shot,
            'pd_wavelengths_um': dict(PD_WAVELENGTHS_UM),
        },
        'datasets': {},
    }
    for label, name in datasets.items():
        path = data_dir / name
        if not path.exists():
            logger.warning('skipping %s: %s not found', label, path)
            continue
        logger.info('sweeping gain on %s (%s)...', label, name)
        sweep = sweep_dataset(
            path,
            label,
            n_shots=n_shots,
            gains=gains,
            window_samples=window_samples,
            delay_max_shift=delay_max_shift,
            include_per_shot=include_per_shot,
        )
        results['datasets'][label] = sweep.to_dict()
        for channel in sweep.channels:
            curve = sweep.curves[channel]
            logger.info(
                '  %s %s (%.3f um): best k %.2f (ratio %.3f vs live %.3f), '
                'median R^2 %.3f at best vs %.3f at live, %.3f at legacy k=%.2f',
                label,
                channel,
                curve.wavelength_um,
                sweep.best_gain(channel),
                sweep.best_gain(channel) / LIVE_K,
                LIVE_K,
                sweep.best_r_squared(channel),
                sweep.r_squared_at(channel, LIVE_K),
                sweep.r_squared_at(channel, LEGACY_K),
                LEGACY_K,
            )
            if curve.peak_fringe_rate_hz is not None:
                logger.info(
                    '    ruled out: peak fringe rate %.1f kHz vs %.0f kHz '
                    'Nyquist (%.0fx headroom)',
                    curve.peak_fringe_rate_hz / 1e3,
                    (sweep.nyquist_hz or 0.0) / 1e3,
                    curve.fringe_rate_headroom or 0.0,
                )
            if curve.delay_median_r_squared is not None:
                assert curve.delay_shifts is not None
                pooled = DelayCurve(
                    shifts=np.asarray(curve.delay_shifts),
                    r_squared=np.asarray(curve.delay_median_r_squared),
                )
                logger.info(
                    '    delay: peak at %+d samples, %s (R^2 %.3f there vs '
                    '%.3f at zero)',
                    pooled.best_shift,
                    'INTERIOR -- a real delay'
                    if pooled.peak_is_interior
                    else 'not interior, so no delay',
                    float(pooled.r_squared.max()),
                    pooled.r_squared_at(0),
                )
            if sweep.per_shot is not None:
                # Logged next to the best-global-gain line, because the point
                # of this number is that it is an upper bound on that one.
                per_shot_gains = sweep.per_shot.best_gains(channel)
                logger.info(
                    '    caveat: per-shot gains median %.2f (IQR %.2f-%.2f), '
                    'median R^2 %.3f -- the ceiling for any scalar gain fix',
                    float(np.median(per_shot_gains)),
                    float(np.percentile(per_shot_gains, 25)),
                    float(np.percentile(per_shot_gains, 75)),
                    sweep.per_shot.median_r_squared(channel),
                )
        logger.info('  %s: %.1f s', label, sweep.runtime_s)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    logger.info('wrote %s', output)
    return results
