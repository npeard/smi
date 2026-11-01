"""Synthetic forward model: drive-voltage trajectory -> fringe -> detector."""

import functools
import hashlib
import json
import logging
import math
import statistics
from dataclasses import asdict, dataclass, fields

import torch

from smi.core.acquisition import BOARDS, STEMLAB_125_14, AcquisitionConfig
from smi.theory.coil_driver import CoilDriver
from smi.theory.waveform import Waveform

# Scale of the drive voltage relative to the rig's pre-compensated waveform
# (CoilDriver.sample(normalize_gain=True), driven by RedPitayaManager at
# generator amplitude 1.0, manager.py:286-290). The code chain alone does not
# reproduce the rig's velocity scale (it depends strongly on the Waveform
# range and is ~400x too large at the rig default 1-1000 Hz), so this is set
# so the MEDIAN per-shot velocity RMS over 256 seeded shots (Waveform
# 1-1000 Hz, generator seed 0; 152.8 um/ms at drive_gain=1) is 0.375 um/ms,
# the middle of the median true velocity RMS already published for the two
# acquisitions, 0.373 and 0.377 um/ms (report/report.typ sec 5.1; no
# acquisition file was opened). The gain is therefore partly a compensation
# for the generator-vs-acquisition time-base mismatch in the rig path (see
# sample_trajectory), not a purely physical analog gain.
NOMINAL_DRIVE_GAIN = 0.002454

# Evaluator limit for smi.estimators.inverse.inverse_fit(max_speed_um_per_ms=...):
# measured 99.9th percentile |v| at drive_gain = 4 x NOMINAL_DRIVE_GAIN (the
# top of the sweep) over the same 256 shots, times a margin of 1.5, rounded up.
# Measured 10.87 um/ms, x1.5 = 16.30, rounded up.
MAX_SPEED_UM_PER_MS = 17.0

# Bump on ANY change to simulate_batch numerics or the coil/trajectory path
# (sample_trajectory, CoilDriver, Waveform): SimConfig.hash() covers config
# values, not code, so a frozen test set written by older code would otherwise
# be reused under the same name. Written to frozen-set attrs and to records.
SIMULATOR_VERSION = 1


def sample_trajectory(
    batch_size: int,
    waveform: Waveform,
    coil: CoilDriver,
    drive_gain: float,
    sample_rate: float,
    device: torch.device,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw rig-style drive voltages and push them through the coil.

    Mirrors ``CoilDriver.sample(normalize_gain=True)``: the random waveform
    spectrum is pre-compensated by the velocity transfer function and scaled by
    a per-shot U(0, 1) AMPLITUDE draw (voltage, not power), then multiplied by
    ``drive_gain`` before the coil transfer function.

    Args:
        batch_size: Number of shots.
        waveform: Source of the random (velocity-shaped) spectrum.
        coil: Speaker transfer function.
        drive_gain: Voltage scale relative to the rig's pre-compensated
            waveform (``NOMINAL_DRIVE_GAIN`` is the calibrated value).
        sample_rate: Acquisition sample rate (Hz).
        device: Target device.
        generator: Optional generator living on ``device``.

    Returns:
        ``(voltage, velocity [um/ms], displacement [um])``, each ``[B, L]``.
    """
    raw = waveform.generate_batch_torch(batch_size, device, generator)
    # The rig pre-compensates on the Waveform's own time base (the generator
    # rate, via t in CoilDriver.sample), but the coil transfer is applied to the
    # acquired record at ``sample_rate``; the mismatch is part of the rig path.
    voltage = coil.precompensate_torch(
        raw, waveform.gen_sample_rate, float(waveform.valid_freqs[0])
    )
    amp = torch.rand(batch_size, 1, device=device, generator=generator) * drive_gain
    voltage = voltage * amp
    velocity, displacement = coil.get_velocity_and_displacement_torch(
        voltage, sample_rate
    )
    return voltage, velocity, displacement


logger = logging.getLogger(__name__)

# Blinded nominal configuration (spec sec. 3.1): every value comes from a
# hardware spec, the rig code, first principles or an already-published
# number; none from opening an acquisition. "Assumed" marks an unknown whose
# nominal is a stated guess and whose sweep (NOMINAL_SWEEPS) is deliberately
# wide.

# RedPitayaManager acquires at acq_dec = 256 (manager.py settings).
NOMINAL_ACQ_DECIMATION = 256
# RedPitayaManager's default drive band, start_freq / end_freq (manager.py).
NOMINAL_DRIVE_BAND_HZ = (1.0, 1000.0)
# Weak-feedback strength: plan nominal, inside the weak regime C < 1.
NOMINAL_FEEDBACK_C = 0.2
# Linewidth enhancement factor: literature-typical 3-5 for diode lasers;
# spec nominal 3.
NOMINAL_ALPHA = 3.0
# Shot length 16384 samples at fs = 125 MHz / 256: T = 33.55 ms. The published
# drift below was measured on this rig, so it names it rather than taking a
# board.
_SHOT_S = (
    STEMLAB_125_14.buffer_size
    * NOMINAL_ACQ_DECIMATION
    / STEMLAB_125_14.sample_rate_dec1_hz
)
# Published within-shot drift: 1 to 2.7 fringes per shot (report/report.typ
# sec 5.1), measured as the peak-to-peak of the windowed phase. One fringe is
# 2 pi rad of phase; a random walk of D rad/sqrt(s) has mean range
# sqrt(8 T / pi) D over a shot of T s, so N fringes -> 2 pi N / sqrt(8 T / pi).
# Nominal at the geometric middle, sqrt(1 * 2.7) = 1.64 fringes. The
# published figure comes through the windowed estimator, which reads somewhat
# low at the nominal D (finite windows smooth the walk) while this conversion uses the
# continuous Brownian range, so the nominal is a lower bound on the true D.
_FRINGES_TO_RAD_PER_SQRT_S = 2 * math.pi / math.sqrt(8 * _SHOT_S / math.pi)
NOMINAL_PHASE_DRIFT_FRINGES = math.sqrt(2.7)
NOMINAL_PHASE_DRIFT_RAD_PER_SQRT_S = (
    NOMINAL_PHASE_DRIFT_FRINGES * _FRINGES_TO_RAD_PER_SQRT_S
)


def rad_per_sqrt_s_to_fringes(rate: float) -> float:
    """Mean within-shot drift in fringes of a phase walk of ``rate`` rad/sqrt(s)."""
    return rate / _FRINGES_TO_RAD_PER_SQRT_S


# Sweep for each uncertain knob, as SimConfig field values (so a sweep point
# is dataclasses.replace(cfg, **{name: value})). Nominals sit inside each.
NOMINAL_SWEEPS: dict[str, tuple[object, ...]] = {
    # 1/4 to 4x the published velocity scale; 4x is MAX_SPEED's anchor.
    'drive_gain': tuple(NOMINAL_DRIVE_GAIN * k for k in (0.25, 0.5, 1.0, 2.0, 4.0)),
    'feedback_c': (0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 0.9),
    'alpha': (0.0, 1.5, 3.0, 4.0, 5.0, 7.0),
    'snr_db': (10.0, 20.0, 30.0, 40.0, 50.0),
    'shot_noise': (False, True),
    'visibility_range': ((0.1, 1.0), (0.3, 1.0), (0.5, 1.0), (0.7, 1.0), (0.9, 1.0)),
    'dc_offset_range': ((0.0, 0.0), (-0.1, 0.1), (-0.3, 0.3), (-1.0, 1.0)),
    'amplitude_drift_rel': (0.0, 0.02, 0.05, 0.1, 0.2, 0.5),
    # 0, half the published low end, low (1 fringe), mid, high (2.7), 2x high.
    'phase_drift_rad_per_sqrt_s': tuple(
        n * _FRINGES_TO_RAD_PER_SQRT_S
        for n in (0.0, 0.5, 1.0, NOMINAL_PHASE_DRIFT_FRINGES, 2.7, 5.4)
    ),
    'contrast_drift_rel': (0.0, 0.02, 0.05, 0.1, 0.2, 0.5),
    # Top below the board's boxcar -3 dB (216 kHz, see SimConfig.nominal).
    'pd_bandwidth_hz': (None, 200e3, 100e3, 30e3, 10e3, 3e3),
    'adc_bits': (None, 14, 12, 10, 8),
    # Input range over DC level: 1.25 (DC fills the range) to 20 (DC 50 mV).
    'adc_full_scale': (1.25, 2.0, 5.0, 10.0, 20.0),
}

# Bracketed Newton iterations for the SMI excess phase. Measured worst-case
# |residual| over 2e6 phases in [-500, 500] rad (float32, alpha in {0, 3, 5}):
# the float32 floor (<= 4.8e-7 rad) is reached after 5 iterations at C = 0.9
# and 7 at C = 0.99; 8 adds one of margin. Above C = 0.99 the bisection
# fallback still converges, only slower (6.7e-6 rad at C = 0.999).
SMI_NEWTON_ITERATIONS = 8


@dataclass(frozen=True)
class SimConfig:
    """Every knob of the synthetic forward model, with no defaults.

    Signals are in units of the nominal fringe: the detector intensity is
    ``a(t) * (1 + V(t) * F(phase))`` relative to its nominal DC level, and the
    output is that minus the nominal DC (1), so a noise-free unit-visibility
    Michelson channel is exactly ``cos(phase)``.

    Attributes:
        acquisition: The board the simulated rig is read out by (shot length
            and base clock). Not part of ``hash()``: the fields it implies for
            the nominal rig are ``sample_rate``, ``adc_bits`` and
            ``adc_full_scale``, which are hashed, and the rest is the fixed
            buffer size.
        wavelengths_um: One wavelength per channel (um).
        sample_rate: Acquisition rate (Hz); must match the Waveform grid.
        start_freq_hz: Lower edge of the drive spectrum (Hz).
        end_freq_hz: Upper edge of the drive spectrum (Hz).
        drive_gain: See ``NOMINAL_DRIVE_GAIN``.
        feedback_c: SMI feedback strength C; 0 is Michelson, 1 and above
            (moderate feedback, multivalued) is rejected.
        alpha: Linewidth enhancement factor; ignored when ``feedback_c == 0``.
        snr_db: Nominal-fringe power (0.5) over noise power; None is noiseless.
        shot_noise: Scale the noise std by sqrt(intensity) (same mean power);
            requires ``snr_db``.
        visibility_range: Per-shot, per-channel fringe visibility ~ U(lo, hi).
        dc_offset_range: Per-shot, per-channel additive DC level ~ U(lo, hi) in
            signal units, added to the optical intensity (so it passes through
            the PD low-pass); (0, 0) is off and draws nothing.
        amplitude_drift_rel: Within-shot relative drift of ``a``: a random
            walk with per-sample step ``scale / sqrt(length)``, i.e. its std
            reaches ``scale`` at the end of a shot of THIS length (the scale is
            relative-per-shot, not per-sample).
        phase_drift_rad_per_sqrt_s: Within-shot phase random walk, independent
            per channel (rad / sqrt(s)).
        contrast_drift_rel: Within-shot relative drift of ``V``, scaled per shot
            as ``amplitude_drift_rel``; ``V * (1 + walk)`` is clamped to [0, 1].
        pd_bandwidth_hz: -3 dB frequency of the causal one-pole photodiode
            low-pass; None is unlimited.
        adc_bits: Quantizer bits; None is unquantized.
        adc_full_scale: Quantizer range [-full_scale, +full_scale) in signal
            units; out-of-range samples clip to the end codes.
    """

    acquisition: AcquisitionConfig
    wavelengths_um: tuple[float, ...]
    sample_rate: float
    start_freq_hz: float
    end_freq_hz: float
    drive_gain: float
    feedback_c: float
    alpha: float
    snr_db: float | None
    shot_noise: bool
    visibility_range: tuple[float, float]
    dc_offset_range: tuple[float, float]
    amplitude_drift_rel: float
    phase_drift_rad_per_sqrt_s: float
    contrast_drift_rel: float
    pd_bandwidth_hz: float | None
    adc_bits: int | None
    adc_full_scale: float

    def _coerce(self) -> None:
        # YAML names a board; checkpoints and frozen sets store asdict() output.
        if isinstance(self.acquisition, str):
            object.__setattr__(self, 'acquisition', BOARDS[self.acquisition])
        elif isinstance(self.acquisition, dict):
            object.__setattr__(
                self, 'acquisition', AcquisitionConfig(**self.acquisition)
            )
        # YAML gives lists and ints; the frozen config's hash must not depend on
        # the spelling (json writes 1 vs 1.0), so coerce to tuples of floats.
        for name in ('wavelengths_um', 'visibility_range', 'dc_offset_range'):
            value = tuple(float(x) for x in getattr(self, name))
            object.__setattr__(self, name, value)
        for name in (
            'sample_rate',
            'start_freq_hz',
            'end_freq_hz',
            'drive_gain',
            'feedback_c',
            'alpha',
            'amplitude_drift_rel',
            'phase_drift_rad_per_sqrt_s',
            'contrast_drift_rel',
            'adc_full_scale',
        ):
            object.__setattr__(self, name, float(getattr(self, name)))
        for name in ('snr_db', 'pd_bandwidth_hz'):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, float(getattr(self, name)))

    def __post_init__(self) -> None:
        self._coerce()
        _check_feedback(self.feedback_c)
        lo, hi = self.visibility_range
        if not 0.0 <= lo <= hi <= 1.0:
            raise ValueError(
                f'visibility_range must satisfy 0 <= lo <= hi <= 1: {lo, hi}'
            )
        dc_lo, dc_hi = self.dc_offset_range
        if dc_lo > dc_hi:
            raise ValueError(f'dc_offset_range must satisfy lo <= hi: {dc_lo, dc_hi}')
        drifts = (
            self.amplitude_drift_rel,
            self.phase_drift_rad_per_sqrt_s,
            self.contrast_drift_rel,
        )
        if min(drifts) < 0:
            raise ValueError(f'drift scales must be >= 0, got {drifts}')
        if self.shot_noise and self.snr_db is None:
            raise ValueError('shot_noise shapes the noise set by snr_db; set snr_db')
        if self.pd_bandwidth_hz is not None and not (
            0 < self.pd_bandwidth_hz < self.sample_rate / 2
        ):
            raise ValueError(
                f'pd_bandwidth_hz must lie in (0, Nyquist): {self.pd_bandwidth_hz}'
            )
        if self.adc_bits is not None and self.adc_bits < 1:
            raise ValueError(f'adc_bits must be >= 1, got {self.adc_bits}')
        if self.adc_full_scale <= 0:
            raise ValueError(f'adc_full_scale must be > 0, got {self.adc_full_scale}')

    def identity_fields(self) -> dict[str, object]:
        """The fields ``hash()`` covers, as ``asdict`` spells them.

        ``acquisition`` is left out so the hashes recorded in the checked-in
        results stay valid: every one of them was taken on the STEMlab 125-14,
        whose clock and ADC already show up in ``sample_rate``, ``adc_bits``
        and ``adc_full_scale``. A config on a board with another buffer size
        would hash like its STEMlab twin, so any other board raises until
        ``acquisition`` is included here (and the recorded hashes regenerated).
        """
        if self.acquisition != STEMLAB_125_14:
            raise ValueError(
                'SimConfig.hash() leaves acquisition out, which is only safe for '
                f'STEMLAB_125_14; include it in identity_fields and regenerate the '
                f'recorded hashes before using {self.acquisition}'
            )
        return {k: v for k, v in asdict(self).items() if k != 'acquisition'}

    def hash(self) -> str:
        """Stable sha1 of ``identity_fields`` (keys frozen test sets)."""
        blob = json.dumps(self.identity_fields(), sort_keys=True)
        return hashlib.sha1(blob.encode(), usedforsecurity=False).hexdigest()

    @classmethod
    def nominal(
        cls,
        acquisition: AcquisitionConfig,
        wavelengths_um: tuple[float, ...],
        feedback_c: float,
        alpha: float,
    ) -> SimConfig:
        """Blinded nominal rig configuration; see NOMINAL_SWEEPS for ranges."""
        # The published drift is per shot of the STEMlab's length (_SHOT_S), and
        # the nominal drift in rad/sqrt(s) is derived from it with that length.
        # Another board's shot length would make that conversion, and the other
        # assumed values here, a different rig's; refuse rather than mix them.
        if acquisition != STEMLAB_125_14:
            raise ValueError(f'nominal is the STEMlab 125-14 rig, got {acquisition}')
        return cls(
            acquisition=acquisition,
            wavelengths_um=wavelengths_um,
            # Board clock over acq_dec = 256.
            sample_rate=acquisition.sample_rate_dec1_hz / NOMINAL_ACQ_DECIMATION,
            start_freq_hz=NOMINAL_DRIVE_BAND_HZ[0],
            end_freq_hz=NOMINAL_DRIVE_BAND_HZ[1],
            # Published median velocity RMS (see NOMINAL_DRIVE_GAIN).
            drive_gain=NOMINAL_DRIVE_GAIN,
            feedback_c=feedback_c,
            alpha=alpha,
            # Assumed: no detector noise spec is known for the rig.
            snr_db=30.0,
            # Assumed: electronic (additive) noise dominates.
            shot_noise=False,
            # Assumed: speckle and alignment visibility, unknown spread.
            visibility_range=(0.5, 1.0),
            # Assumed: ambient light / electronic offset of up to 10% of DC.
            dc_offset_range=(-0.1, 0.1),
            # Assumed: no published amplitude drift.
            amplitude_drift_rel=0.05,
            # Published 1-2.7 fringes per shot, geometric middle (see above).
            phase_drift_rad_per_sqrt_s=NOMINAL_PHASE_DRIFT_RAD_PER_SQRT_S,
            # Assumed: the report says contrast wanders but gives no number.
            contrast_drift_rel=0.05,
            # Assumed: the PD amplifier model is not recorded in the repo. The
            # board's decimation averaging (acq_set averaging=True: a 256-sample
            # boxcar at 125 MHz) is -3 dB at 0.443 fs = 216 kHz and costs 0.1%
            # at 10 kHz fringes, so it does not set this and only caps the sweep.
            pd_bandwidth_hz=100e3,
            adc_bits=acquisition.adc_bits,
            # LV input range +/-1 V (the board's) over an ASSUMED 0.5 V DC
            # level, in signal units. The step matches the rig; clipping does
            # not (the rig range is fixed in volts, this one centred on DC).
            adc_full_scale=acquisition.input_full_scale_v / 0.5,
        )

    @classmethod
    def from_calibration(
        cls,
        acquisition: AcquisitionConfig,
        calibration: dict,
        wavelengths_um: tuple[float, ...],
    ) -> SimConfig:
        """Config from ``smi.analysis.calibrate_sim.calibrate`` output.

        Per-channel estimates are pooled by the median across channels (the
        simulator's knobs are shared by all channels; ``visibility_range``
        pools each bound). A knob the data could not identify (None in every
        channel) takes ``SimConfig.nominal``'s value and is logged, so a
        fallback is never silent.
        """
        channels = list(calibration['channels'].values())
        nominal = cls.nominal(
            acquisition, wavelengths_um, NOMINAL_FEEDBACK_C, NOMINAL_ALPHA
        )
        values: dict[str, object] = {
            'sample_rate': calibration['sample_rate'],
            'drive_gain': calibration['drive_gain'],
            'adc_bits': calibration['adc_bits'],
        }
        fallback = []
        per_channel = [
            f.name
            for f in fields(cls)
            if f.name in channels[0] and f.name != 'adc_bits'
        ]
        for name in per_channel:
            found = [c[name] for c in channels if c[name] is not None]
            if not found:
                fallback.append(name)
            elif name == 'visibility_range':
                lo, hi = zip(*found, strict=True)
                values[name] = (statistics.median(lo), statistics.median(hi))
            else:
                values[name] = statistics.median(found)
        if fallback:
            logger.warning('not identifiable, nominal used: %s', ', '.join(fallback))
        known = {f.name: getattr(nominal, f.name) for f in fields(cls)}
        return cls(**{**known, **values})

    @classmethod
    def noise_free(
        cls,
        *,
        acquisition: AcquisitionConfig,
        wavelengths_um: tuple[float, ...],
        sample_rate: float,
        start_freq_hz: float,
        end_freq_hz: float,
        drive_gain: float,
        feedback_c: float,
        alpha: float,
    ) -> SimConfig:
        """Config with every detector knob at its exact-identity value."""
        return cls(
            acquisition=acquisition,
            wavelengths_um=wavelengths_um,
            sample_rate=sample_rate,
            start_freq_hz=start_freq_hz,
            end_freq_hz=end_freq_hz,
            drive_gain=drive_gain,
            feedback_c=feedback_c,
            alpha=alpha,
            snr_db=None,
            shot_noise=False,
            visibility_range=(1.0, 1.0),
            dc_offset_range=(0.0, 0.0),
            amplitude_drift_rel=0.0,
            phase_drift_rad_per_sqrt_s=0.0,
            contrast_drift_rel=0.0,
            pd_bandwidth_hz=None,
            adc_bits=None,
            # Unused while adc_bits is None; a unit fringe spans exactly +/-1.
            adc_full_scale=1.0,
        )


@dataclass
class SimBatch:
    """One simulated batch, all float32 on the requested device."""

    signals: torch.Tensor  # [B, C, L] detector output, not normalized
    voltage: torch.Tensor  # [B, L] drive voltage
    velocity: torch.Tensor  # [B, L] um/ms
    displacement: torch.Tensor  # [B, L] um
    optical_phase: torch.Tensor  # [B, C, L] rad: phi_0 (pre-feedback) + phase drift


def _check_feedback(feedback_c: float) -> None:
    if feedback_c >= 1.0:
        raise ValueError(
            f'feedback_c={feedback_c} >= 1 is the moderate-feedback regime '
            '(multivalued phase), which is out of scope'
        )
    if feedback_c < 0.0:
        raise ValueError(f'feedback_c must be >= 0, got {feedback_c}')


def smi_excess_phase(
    phase: torch.Tensor, feedback_c: float, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the weak-feedback excess phase equation by bracketed Newton.

    The feedback phase phi_F = phi_0 - C sin(phi_F + atan(alpha)) is solved as
    delta = phi_F - phi_0 on the reduced phase psi = (phi_0 + atan(alpha)) mod
    2 pi, because phi_0 reaches hundreds of rad, where float32 cannot resolve
    1e-6 rad. h(delta) = delta + C sin(psi + delta) has h' >= 1 - C > 0, so the
    root is unique and lies in [-C, C]. The bracket starts at [-2C, 2C]: the
    root sits exactly ON -C at psi = pi/2 + C (and on +C at psi = 3 pi/2 - C), where a
    strict [-C, C] bracket rejects every Newton step and degrades to bisection.

    Returns:
        ``(delta, psi)``, so the residual is ``delta + C sin(psi + delta)``.
    """
    _check_feedback(feedback_c)
    psi = torch.remainder(phase + math.atan(alpha), 2 * math.pi)
    delta = -feedback_c * torch.sin(psi)
    lo = torch.full_like(psi, -2 * feedback_c)
    hi = torch.full_like(psi, 2 * feedback_c)
    for _ in range(SMI_NEWTON_ITERATIONS):
        h = delta + feedback_c * torch.sin(psi + delta)
        lo = torch.where(h < 0, delta, lo)
        hi = torch.where(h > 0, delta, hi)
        step = delta - h / (1 + feedback_c * torch.cos(psi + delta))
        delta = torch.where((step > lo) & (step < hi), step, 0.5 * (lo + hi))
    return delta, psi


def fringe(phase: torch.Tensor, feedback_c: float, alpha: float) -> torch.Tensor:
    """Fringe shape cos(phi_F): Michelson for C = 0, weak-feedback SMI above.

    Sign convention: displacement d increasing (v > 0) increases the phase.
    For C > 0 and alpha > 0 the fringe then rises slowly and falls steeply,
    so the third moment of dI/dt over whole fringes is negative; v < 0
    mirrors it. With alpha = 0 the fringe is even under (phase -> -phase): the
    low point is narrowed and the peak broadened, which carries no direction, so
    direction is observable only for C > 0 AND alpha != 0.
    """
    if feedback_c == 0.0:
        return torch.cos(phase)
    delta, _ = smi_excess_phase(phase, feedback_c, alpha)
    return torch.cos(phase + delta)


def noise_std(snr_db: float) -> float:
    """Additive noise std for ``snr_db``: nominal-fringe power 0.5 over noise power."""
    return math.sqrt(0.5 * 10 ** (-snr_db / 10))


def one_pole_coefficient(cutoff_hz: float, sample_rate: float) -> float:
    """Pole p of the one-pole low-pass whose discrete response is -3 dB at cutoff."""
    b = 2.0 - math.cos(2 * math.pi * cutoff_hz / sample_rate)
    return b - math.sqrt(b * b - 1.0)


def pd_lowpass(x: torch.Tensor, cutoff_hz: float, sample_rate: float) -> torch.Tensor:
    """Causal one-pole low-pass y[n] = p y[n-1] + (1 - p) x[n] along the last dim.

    p is solved so the DISCRETE response is exactly -3 dB at ``cutoff_hz``
    (exp(-2 pi f_c / f_s) is only the small-f_c approximation). The recursion
    is evaluated as an FFT convolution with the kernel (1 - p) p^n truncated
    at the record length, which is exact inside the record, zero-padded to 2L
    so nothing wraps. The state starts at y[-1] = x[0] (steady state), adding
    x[0] p^(n+1); both terms use only samples at or before n.
    """
    length = x.shape[-1]
    kernel, state = _lowpass_kernel(length, cutoff_hz, sample_rate, x.device, x.dtype)
    nfft = 2 * length
    y = torch.fft.irfft(torch.fft.rfft(x, nfft) * kernel, nfft)[..., :length]
    return y + x[..., :1] * state


@functools.cache
def _lowpass_kernel(
    length: int,
    cutoff_hz: float,
    sample_rate: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``pd_lowpass``'s kernel spectrum and steady-state term, per geometry.

    Cached because every simulated batch of a config rebuilds the same two.
    """
    p = one_pole_coefficient(cutoff_hz, sample_rate)
    n = torch.arange(length, device=device, dtype=dtype)
    decay = torch.exp(n * math.log(p))
    return torch.fft.rfft((1 - p) * decay, 2 * length), decay * p


@functools.cache
def _wavelengths_tensor(
    wavelengths_um: tuple[float, ...], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """``[1, C, 1]`` wavelengths, cached per config and device for simulate_batch."""
    return torch.tensor(wavelengths_um, device=device, dtype=dtype).view(1, -1, 1)


def _quantize(x: torch.Tensor, bits: int, full_scale: float) -> torch.Tensor:
    """Mid-tread two's-complement ADC: codes -2^(b-1) .. 2^(b-1) - 1.

    The step is full_scale / 2^(b-1), so the top code is one step BELOW
    +full_scale; anything beyond the range clips to the end codes.
    """
    half = 2 ** (bits - 1)
    step = full_scale / half
    return torch.clamp(torch.round(x / step), -half, half - 1) * step


def _walk(
    shape: tuple[int, ...], device: torch.device, generator: torch.Generator | None
) -> torch.Tensor:
    """Gaussian random walk with unit-variance increments along the last dim."""
    return torch.randn(shape, device=device, generator=generator).cumsum(-1)


@functools.cache
def drive(
    acquisition: AcquisitionConfig, start_freq_hz: float, end_freq_hz: float
) -> tuple[Waveform, CoilDriver]:
    """The drive ``simulate_batch`` feeds ``sample_trajectory`` for this band.

    Cached because Waveform builds its frequency grid in Python.
    """
    return (
        Waveform(
            start_freq=start_freq_hz, end_freq=end_freq_hz, acquisition=acquisition
        ),
        CoilDriver(),
    )


def simulate_batch(
    cfg: SimConfig,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> SimBatch:
    """Simulate one batch: trajectory -> phase -> fringe -> detector.

    Order: trajectory, optical phase (+ phase drift), fringe (Michelson or
    SMI), visibility / contrast / amplitude, DC offset, PD low-pass, additive / shot
    noise, ADC. Random draws follow that order and a knob that is off draws
    nothing, so no knob shifts the motion, phi_k or visibility (all drawn
    first). The remaining coupling: each ON knob's draws shift the draws after
    it (phase drift, contrast drift, amplitude drift, DC offset, noise), so
    turning one on re-rolls the later ones' realizations.

    Args:
        cfg: Simulator configuration.
        batch_size: Number of shots.
        device: Device of the returned tensors.
        generator: Seeded generator, or None for the global RNG on ``device``.
            A generator is device-bound, so a seeded call simulates on the
            generator's device (CPU for reproducible frozen sets) and moves
            the result to ``device``.
    """
    sim_device = generator.device if generator is not None else device
    waveform, coil = drive(cfg.acquisition, cfg.start_freq_hz, cfg.end_freq_hz)
    if waveform.acq_sample_rate != cfg.sample_rate:
        raise ValueError(
            f'sample_rate {cfg.sample_rate} does not match the Waveform grid '
            f'({waveform.acq_sample_rate} Hz)'
        )
    voltage, velocity, displacement = sample_trajectory(
        batch_size,
        waveform,
        coil,
        cfg.drive_gain,
        cfg.sample_rate,
        sim_device,
        generator,
    )
    n_ch, length = len(cfg.wavelengths_um), displacement.shape[-1]
    full = (batch_size, n_ch, length)

    lam = _wavelengths_tensor(
        cfg.wavelengths_um, torch.device(sim_device), torch.get_default_dtype()
    )
    phi_k = torch.rand(batch_size, n_ch, 1, device=sim_device, generator=generator)
    lo, hi = cfg.visibility_range
    u = torch.rand(batch_size, n_ch, 1, device=sim_device, generator=generator)
    visibility = lo + (hi - lo) * u
    phase = 4 * math.pi * displacement[:, None] / lam + 2 * math.pi * phi_k
    if cfg.phase_drift_rad_per_sqrt_s > 0:
        sigma = cfg.phase_drift_rad_per_sqrt_s / math.sqrt(cfg.sample_rate)
        phase = phase + sigma * _walk(full, sim_device, generator)

    # Drift walks are scaled so their std reaches the stated value at shot end.
    if cfg.contrast_drift_rel > 0:
        rel = cfg.contrast_drift_rel / math.sqrt(length)
        walk = _walk(full, sim_device, generator)
        visibility = (visibility * (1 + rel * walk)).clamp(0.0, 1.0)
    intensity = 1 + visibility * fringe(phase, cfg.feedback_c, cfg.alpha)
    if cfg.amplitude_drift_rel > 0:
        rel = cfg.amplitude_drift_rel / math.sqrt(length)
        intensity = intensity * (1 + rel * _walk(full, sim_device, generator))
    if cfg.dc_offset_range != (0.0, 0.0):
        dc_lo, dc_hi = cfg.dc_offset_range
        dc = torch.rand(batch_size, n_ch, 1, device=sim_device, generator=generator)
        intensity = intensity + dc_lo + (dc_hi - dc_lo) * dc
    if cfg.pd_bandwidth_hz is not None:
        intensity = pd_lowpass(intensity, cfg.pd_bandwidth_hz, cfg.sample_rate)

    signals = intensity - 1
    if cfg.snr_db is not None:
        noise = noise_std(cfg.snr_db) * torch.randn(
            full, device=sim_device, generator=generator
        )
        if cfg.shot_noise:
            noise = noise * intensity.clamp(min=0).sqrt()
        signals = signals + noise
    if cfg.adc_bits is not None:
        signals = _quantize(signals, cfg.adc_bits, cfg.adc_full_scale)

    return SimBatch(
        signals=signals.to(device),
        voltage=voltage.to(device),
        velocity=velocity.to(device),
        displacement=displacement.to(device),
        optical_phase=phase.to(device),
    )


def normalization_stats(
    cfg: SimConfig, n_shots: int, seed: int, target: str
) -> dict[str, torch.Tensor]:
    """Fixed affine stats for ``Model``, from a seeded CPU sample of ``cfg``.

    Replaces the per-shot z-score, which used whole-shot statistics and so
    leaked future samples into causal models. The output stats are those of
    ``target``, the quantity the network emits: scaling a displacement output
    by the velocity's std would leave the network a target far from unit
    variance. (A displacement's mean is irrelevant to the loss, which removes
    each shot's start offset, but it centers the network's output range.)

    Args:
        cfg: Simulator config.
        n_shots: Shots in the sample.
        seed: Seed of the sample's generator.
        target: ``'velocity'`` or ``'displacement'``.

    Returns:
        ``Model`` keyword arguments: ``input_mean`` / ``input_std`` ``[C]`` and
        scalar ``output_mean`` / ``output_std``.

    Raises:
        ValueError: For any other ``target``.
    """
    if target not in ('velocity', 'displacement'):
        raise ValueError(f"target must be 'velocity' or 'displacement', not {target!r}")
    batch = simulate_batch(
        cfg, n_shots, torch.device('cpu'), torch.Generator().manual_seed(seed)
    )
    output = getattr(batch, target)
    return {
        'input_mean': batch.signals.mean(dim=(0, 2)),
        'input_std': batch.signals.std(dim=(0, 2)),
        'output_mean': output.mean(),
        'output_std': output.std(),
    }
