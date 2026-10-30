"""Synthetic forward model: drive-voltage trajectory -> fringe -> detector."""

import functools
import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch

from smi.synthetic.coil_driver import CoilDriver
from smi.synthetic.waveform import Waveform

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

# Evaluator limit for baseline_fit.inverse_fit(max_speed_um_per_ms=...):
# measured 99.9th percentile |v| at drive_gain = 4 x NOMINAL_DRIVE_GAIN (the
# top of the sweep) over the same 256 shots, times a margin of 1.5, rounded up.
# Measured 10.87 um/ms, x1.5 = 16.30, rounded up.
MAX_SPEED_UM_PER_MS = 17.0


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
    a per-shot U(0, 1) power draw, then multiplied by ``drive_gain`` before the
    coil transfer function.

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

    def hash(self) -> str:
        """Stable sha1 of the field values (keys frozen test sets)."""
        blob = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha1(blob.encode(), usedforsecurity=False).hexdigest()

    @classmethod
    def noise_free(
        cls,
        *,
        wavelengths_um: tuple[float, ...],
        sample_rate: float,
        start_freq_hz: float,
        end_freq_hz: float,
        drive_gain: float,
        feedback_c: float,
        alpha: float,
    ) -> 'SimConfig':
        """Config with every detector knob at its exact-identity value."""
        return cls(
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
    b = 2.0 - math.cos(2 * math.pi * cutoff_hz / sample_rate)
    p = b - math.sqrt(b * b - 1.0)
    n = torch.arange(length, device=x.device, dtype=x.dtype)
    decay = torch.exp(n * math.log(p))
    nfft = 2 * length
    spectrum = torch.fft.rfft(x, nfft) * torch.fft.rfft((1 - p) * decay, nfft)
    y = torch.fft.irfft(spectrum, nfft)[..., :length]
    return y + x[..., :1] * (decay * p)


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
def _drive(start_freq_hz: float, end_freq_hz: float) -> tuple[Waveform, CoilDriver]:
    # Waveform builds its frequency grid in Python; cache it across steps.
    return Waveform(start_freq=start_freq_hz, end_freq=end_freq_hz), CoilDriver()


def simulate_batch(
    cfg: SimConfig,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> SimBatch:
    """Simulate one batch: trajectory -> phase -> fringe -> detector.

    Order: trajectory, optical phase (+ phase drift), fringe (Michelson or
    SMI), visibility / contrast / amplitude, PD low-pass, additive / shot
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
    waveform, coil = _drive(cfg.start_freq_hz, cfg.end_freq_hz)
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

    lam = torch.tensor(cfg.wavelengths_um, device=sim_device).view(1, -1, 1)
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
        sigma = math.sqrt(0.5 * 10 ** (-cfg.snr_db / 10))
        noise = sigma * torch.randn(full, device=sim_device, generator=generator)
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
    cfg: SimConfig, n_shots: int, seed: int
) -> dict[str, torch.Tensor]:
    """Fixed affine stats for ``Model``, from a seeded CPU sample of ``cfg``.

    Replaces the per-shot z-score, which used whole-shot statistics and so
    leaked future samples into causal models. The output stats are the
    velocity target's (the synthetic training target).

    Returns:
        ``Model`` keyword arguments: ``input_mean`` / ``input_std`` ``[C]`` and
        scalar ``output_mean`` / ``output_std``.
    """
    batch = simulate_batch(
        cfg, n_shots, torch.device('cpu'), torch.Generator().manual_seed(seed)
    )
    return {
        'input_mean': batch.signals.mean(dim=(0, 2)),
        'input_std': batch.signals.std(dim=(0, 2)),
        'output_mean': batch.velocity.mean(),
        'output_std': batch.velocity.std(),
    }
