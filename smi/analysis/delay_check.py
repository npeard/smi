"""Measure any time shift between the velocity target and the photodiode signals.

The supervision target (``velocity``) is not recorded: it is derived from the
coil drive channel ``RP1_CH1`` through
:class:`~smi.synthetic.coil_driver.CoilDriver`, while the model's inputs are the
three measured photodiode channels. Those two travel different paths -- one a
numeric transfer function, the other analog optics and a second Red Pitaya -- so
a constant time offset between them is possible, and would corrupt every model
trained against the pair. This module measures that offset and reports it; it
deliberately applies no correction.

**What is correlated, and why it is not the raw photodiode trace.** A
self-mixing photodiode signal is a fringe pattern: it oscillates roughly as
``cos(4*pi*d/lambda)`` in the target's *displacement*, so it has no linear
relationship with velocity at all, and cross-correlating velocity against the
raw trace measures nothing regardless of the delay. What velocity does set is
the *instantaneous fringe rate*: faster motion means faster oscillation. A
smoothed ``|d/dt PD|`` is therefore monotone in ``|velocity|``, and the pair
``(|velocity|, fringe envelope)`` is the linearly comparable one. Both are
non-negative envelopes at twice the drive frequency, which is acceptable here:
a delay shifts both equally, so the lag is still the quantity recovered.

**Why the verdict is an interior peak and not an argmax.** ``np.argmax`` always
returns something. When two signals share no delay-localized structure, the
correlation curve is monotone across the search window and the argmax pins to
an edge -- which means *no peak was found*, not that a delay of ``max_lag``
samples exists. Reading such an edge argmax as a measured lag was a real error
made earlier on this branch. ``is_interior_peak`` is therefore true only when
the argmax lies strictly inside the window and the peak clears
:data:`MIN_PEAK_CORRELATION`; any other outcome means the measurement did not
resolve a delay and ``lag_samples`` must not be acted on.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from smi.synthetic.coil_driver import CoilDriver

logger = logging.getLogger(__name__)

# Nominal acquisition sample rate (Hz): 125 MS/s at decimation 256. Used only
# as a fallback; the real rate is read from the file's attrs.
DEFAULT_SAMPLE_RATE_HZ = 488281.25

# A normalized cross-correlation below this is not a peak worth trusting. The
# envelopes compared here are strongly shaped by the drive waveform, so a
# genuine alignment sits far above this; the threshold exists to reject noise.
MIN_PEAK_CORRELATION = 0.3

# Photodiode channels, in the order VelocityDataset feeds them to the model.
PD_CHANNELS: tuple[str, ...] = ('RP1_CH2', 'RP2_CH1', 'RP2_CH2')
DRIVE_CHANNEL = 'RP1_CH1'

DATA_DIR = Path(__file__).resolve().parent / 'data'
DATASETS: dict[str, str] = {
    'free-space': 'free-space-synchro_10k.h5',
    'mmfiber': 'mmfiber-synchro_10k.h5',
}


@dataclass
class LagResult:
    """Outcome of one cross-correlation lag estimate.

    Attributes:
        lag_samples: Argmax lag in samples. Positive means ``signal`` lags
            ``target`` -- it must be shifted earlier to align. Meaningless
            unless ``is_interior_peak`` is true.
        lag_us: ``lag_samples`` converted to microseconds.
        peak_correlation: Normalized correlation coefficient at the argmax.
        is_interior_peak: True only when the argmax is strictly inside the
            search window and clears :data:`MIN_PEAK_CORRELATION`.
    """

    lag_samples: int
    lag_us: float
    peak_correlation: float
    is_interior_peak: bool


def _standardize(x: np.ndarray) -> np.ndarray:
    """Return ``x`` as flat float64 with zero mean and unit standard deviation."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x - x.mean()
    std = x.std()
    if std == 0.0:
        return np.zeros_like(x)
    return x / std


def _correlation_window(
    target: np.ndarray, signal: np.ndarray, max_lag: int
) -> np.ndarray:
    """Normalized cross-correlation of two standardized arrays over +/-``max_lag``.

    Args:
        target: Standardized reference waveform.
        signal: Standardized comparison waveform, same length.
        max_lag: Half-width of the lag window, in samples.

    Returns:
        Array of length ``2 * max_lag + 1``, index ``k + max_lag`` holding the
        correlation at lag ``k``.
    """
    n = target.size
    # Full linear cross-correlation, normalized by n so lag 0 of a standardized
    # pair equals Pearson's r. Done through the FFT rather than np.correlate:
    # the direct form is O(n^2) and, at n = 16384 over hundreds of shots and
    # six channel/dataset pairs, does not finish in a usable time.
    size = int(2 ** np.ceil(np.log2(2 * n - 1)))
    spectrum = np.fft.rfft(signal, size) * np.conjugate(np.fft.rfft(target, size))
    # Index n-1 of the 'full' correlation is lag 0; np.roll puts it there.
    full = np.roll(np.fft.irfft(spectrum, size), n - 1)[: 2 * n - 1] / n
    return full[n - 1 - max_lag : n + max_lag]


def _verdict(curve: np.ndarray, max_lag: int, sample_rate_hz: float) -> LagResult:
    """Turn a correlation curve into a :class:`LagResult`.

    An argmax at either end of ``curve`` means the correlation never turned
    over inside the window, so no peak was found; see the module docstring.
    """
    best = int(np.argmax(curve))
    lag_samples = int(best - max_lag)
    peak = float(curve[best])
    interior = 0 < best < curve.size - 1 and peak >= MIN_PEAK_CORRELATION
    return LagResult(
        lag_samples=lag_samples,
        lag_us=lag_samples / sample_rate_hz * 1e6,
        peak_correlation=peak,
        is_interior_peak=interior,
    )


def estimate_lag(
    target: np.ndarray,
    signal: np.ndarray,
    max_lag: int,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
) -> LagResult:
    """Estimate the lag of ``signal`` relative to ``target``.

    Both inputs are standardized before correlating, so the reported peak is a
    correlation coefficient in ``[-1, 1]`` and is comparable across shots and
    channels of differing amplitude.

    Args:
        target: Reference 1-D waveform.
        signal: 1-D waveform of the same length, possibly time-shifted.
        max_lag: Half-width of the search window, in samples.
        sample_rate_hz: Sample rate used to convert samples to microseconds.

    Returns:
        A :class:`LagResult`. See the module docstring for why
        ``is_interior_peak``, not ``lag_samples``, is the verdict.

    Raises:
        ValueError: If the inputs differ in length, or ``max_lag`` is not
            positive and smaller than the signal length.
    """
    a = _standardize(target)
    b = _standardize(signal)
    if a.shape != b.shape:
        raise ValueError(
            f'target and signal must match in length: {a.shape} vs {b.shape}'
        )
    if not 0 < max_lag < a.size:
        raise ValueError(f'max_lag must be in (0, {a.size}), got {max_lag}')

    return _verdict(_correlation_window(a, b, max_lag), max_lag, sample_rate_hz)


def fringe_envelope(pd: np.ndarray, smooth_samples: int) -> np.ndarray:
    """Return a fringe-rate envelope: boxcar-smoothed ``|d/dt PD|``.

    Args:
        pd: Raw photodiode trace, shape ``(n_time,)``.
        smooth_samples: Boxcar width. Must span several fringe periods so the
            oscillation averages out and only its rate survives.

    Returns:
        A 1-D array of length ``n_time - smooth_samples``.
    """
    rate = np.abs(np.diff(np.asarray(pd, dtype=np.float64)))
    kernel = np.ones(smooth_samples) / smooth_samples
    return np.convolve(rate, kernel, mode='valid')


def speed_envelope(velocity: np.ndarray, smooth_samples: int) -> np.ndarray:
    """Smooth ``|velocity|`` exactly as :func:`fringe_envelope` smooths the PD.

    The leading sample is dropped so the two envelopes line up index-for-index
    with the ``np.diff`` taken inside :func:`fringe_envelope`; otherwise the
    differencing alone would show up as a half-sample bias in the lag.

    Args:
        velocity: Velocity waveform in um/ms, shape ``(n_time,)``.
        smooth_samples: Boxcar width, matching the photodiode side.

    Returns:
        A 1-D array of length ``n_time - smooth_samples``.
    """
    speed = np.abs(np.asarray(velocity, dtype=np.float64)[1:])
    kernel = np.ones(smooth_samples) / smooth_samples
    return np.convolve(speed, kernel, mode='valid')


def measure_dataset(
    h5_path: Path, n_shots: int = 200, max_lag: int = 512, smooth_samples: int = 64
) -> dict[str, LagResult]:
    """Measure the velocity/photodiode lag per channel over a sample of shots.

    Per-shot correlation curves are averaged before the argmax is taken, so a
    lag common to every shot survives while per-shot noise averages down.

    Args:
        h5_path: Acquisition HDF5 file.
        n_shots: Number of leading shots to average over.
        max_lag: Half-width of the lag search window, in samples.
        smooth_samples: Boxcar width for both envelopes.

    Returns:
        Mapping photodiode channel name -> :class:`LagResult`.
    """
    coil = CoilDriver()
    with h5py.File(h5_path, 'r') as handle:
        sample_rate = float(handle.attrs.get('sample_rate', DEFAULT_SAMPLE_RATE_HZ))
        n_shots = min(n_shots, int(handle[DRIVE_CHANNEL].shape[0]))
        drive = np.asarray(handle[DRIVE_CHANNEL][:n_shots], dtype=np.float64)
        pd_raw = {
            c: np.asarray(handle[c][:n_shots], dtype=np.float64) for c in PD_CHANNELS
        }

    speed = np.stack(
        [
            _standardize(
                speed_envelope(
                    coil.get_velocity_and_displacement(drive[i], sample_rate)[0],
                    smooth_samples,
                )
            )
            for i in range(n_shots)
        ]
    )

    results: dict[str, LagResult] = {}
    for channel, traces in pd_raw.items():
        curve = np.zeros(2 * max_lag + 1)
        for i in range(n_shots):
            envelope = _standardize(fringe_envelope(traces[i], smooth_samples))
            curve += _correlation_window(speed[i], envelope, max_lag)
        curve /= n_shots
        results[channel] = _verdict(curve, max_lag, sample_rate)
    return results


def main() -> None:
    """Report the measured lag for every dataset and photodiode channel."""
    parser = argparse.ArgumentParser(description='Velocity/photodiode delay check.')
    parser.add_argument('--n-shots', type=int, default=200)
    parser.add_argument('--max-lag', type=int, default=512)
    parser.add_argument('--smooth-samples', type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    columns = ('dataset', 'channel', 'lag', 'lag_us', 'peak')
    header = '{:<12} {:<9} {:>8} {:>10} {:>7}  verdict'.format(*columns)
    logger.info(header)
    logger.info('-' * len(header))
    for label, name in DATASETS.items():
        path = DATA_DIR / name
        if not path.exists():
            logger.warning('%-12s (missing: %s)', label, path)
            continue
        results = measure_dataset(
            path,
            n_shots=args.n_shots,
            max_lag=args.max_lag,
            smooth_samples=args.smooth_samples,
        )
        for channel, r in results.items():
            verdict = 'interior peak' if r.is_interior_peak else 'NO interior peak'
            logger.info(
                '%-12s %-9s %8d %10.2f %7.3f  %s',
                label,
                channel,
                r.lag_samples,
                r.lag_us,
                r.peak_correlation,
                verdict,
            )


if __name__ == '__main__':
    main()
