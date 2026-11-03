"""Alignment generalization: nominal checkpoints scored on shifted test sets.

A grid's trained checkpoints are scored, unchanged, on frozen sets whose
``SimConfig`` differs from the training one in one knob. Nothing is retrained.

Analytic predictions, per knob (``eps`` is the relative shift):

* ``wavelength_scale`` ``s = 1 + eps`` multiplies every wavelength: the
  degenerate, benign case, since real diodes do not tune in lockstep. The optical
  phase is ``4 pi d / (s lambda) = 4 pi (d / s) / lambda``, so the shifted
  signals are *exactly* those of the trajectory ``d / s`` at the trained
  wavelengths: a wavelength shift common to all channels is indistinguishable
  from a velocity rescale, by any estimator. A model that is right on the
  nominal set therefore returns ``v / s``, an error ``-v eps / (1 + eps)``, and
  ``RMSE(eps)^2 ~= RMSE_0^2 + (eps / (1 + eps))^2 <v^2>`` (the nominal error
  and the scale error being uncorrelated). The scale term overtakes the
  nominal error at ``|eps| ~ RMSE_0 / v_rms``. The ranges are illustrative: a
  relative shift of 1e-3 is 0.6 nm at 635 nm; no diode tuning coefficient is
  cited here.
* ``wavelength_independent`` (value ``eps``): channel ``k`` is scaled by
  ``1 + eps * s_k`` with the fixed pattern ``s = WAVELENGTH_PATTERN`` (+1, -1,
  +0.5: opposite signs, unequal sizes, so no common scale absorbs it). Each
  channel alone is a velocity rescale by its own factor, so the channels now
  disagree about ``d`` and no trajectory reproduces all three; the
  per-channel phase error ``4 pi d eps s_k / lambda_k`` grows with excursion.
  The prediction: at least the common-scale cost at the same ``|eps|``
  (``~ |eps| v_rms`` once above RMSE_0), and more where the model leans on
  the inter-channel phase relations that a common scale preserves.
* ``phase_offset``: with ``channel_phase_offsets_rad`` None (the training
  path) every channel draws its own uniform phase per shot, so a static
  per-channel offset added to it leaves the test distribution unchanged: that
  knob is a no-op by construction. What can be set is the shared-phase path
  (one uniform phase plus offsets ``k * value`` on channel ``k``), whose joint
  phases lie inside the support the model was trained on. The nominal score
  averages the error over inter-channel phase configurations; a pinned
  configuration samples one slice of that average. The prediction: the error
  moves with ``value`` by the within-distribution spread of error over phase
  configuration, which the nominal score does not show (several per-shot SEs
  in the recorded sweep, not a no-op).
* ``feedback_c``: C changes the fringe shape (the SMI sawtooth asymmetry that
  carries the direction), not its rate, so no closed-form scale error exists.
  The prediction is only qualitative: the minimum at the trained C = 0.2,
  rising on both sides, fastest toward C -> 0 where the asymmetry vanishes.

The checkpoint's metadata keeps its trained wavelengths, which a shift beyond
``WAVELENGTH_TOL_UM`` would make ``CheckpointEstimator`` reject; the
:class:`_TrainedChannels` adapter selects the file's (shifted) channels and
runs the model as trained, on purpose.
"""

import dataclasses
import math
from pathlib import Path
from typing import Literal

from torch import Tensor

from smi.data.frozen import load_or_create
from smi.experiments.evaluate import CheckpointEstimator, evaluate
from smi.experiments.grids import grid as get_grid
from smi.experiments.results_artifact import RunRecord, append_results
from smi.experiments.runner import Cell, expand_grid
from smi.theory.forward import SimConfig

Knob = Literal[
    'feedback_c', 'wavelength_scale', 'wavelength_independent', 'phase_offset'
]
KNOBS: tuple[Knob, ...] = (
    'feedback_c',
    'wavelength_scale',
    'wavelength_independent',
    'phase_offset',
)

#: Per-channel signs and sizes of the ``wavelength_independent`` shift.
WAVELENGTH_PATTERN = (1.0, -1.0, 0.5)

# The noise-free SMI sim of q3_s1_families; the shifts perturb it.
NOMINAL_SIM = 'smi-a3'


def shifted_config(cfg: SimConfig, knob: Knob, value: float) -> SimConfig:
    """``cfg`` with one knob set; the identity knob value returns ``cfg``."""
    if knob == 'feedback_c':
        return dataclasses.replace(cfg, feedback_c=value)
    if knob == 'wavelength_scale':
        return dataclasses.replace(
            cfg, wavelengths_um=tuple(w * value for w in cfg.wavelengths_um)
        )
    if knob == 'wavelength_independent':
        return dataclasses.replace(
            cfg,
            wavelengths_um=tuple(
                w * (1 + value * p)
                for w, p in zip(cfg.wavelengths_um, WAVELENGTH_PATTERN, strict=False)
            ),
        )
    offsets = tuple(k * value for k in range(len(cfg.wavelengths_um)))
    return dataclasses.replace(cfg, channel_phase_offsets_rad=offsets)


class _TrainedChannels:
    """Feeds a checkpoint the shifted channels as if they were its own.

    Every attribute but ``predict_velocity`` is the checkpoint's, so the record
    describes the checkpoint itself.
    """

    def __init__(
        self, inner: CheckpointEstimator, shifted_um: tuple[float, ...]
    ) -> None:
        self.inner = inner
        self.shifted_um = shifted_um

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)

    def predict_velocity(
        self, signals: Tensor, wavelengths_um: tuple[float, ...]
    ) -> Tensor:
        expected = self.shifted_um
        if len(wavelengths_um) != len(expected) or not all(
            math.isclose(a, b, rel_tol=1e-9)
            for a, b in zip(wavelengths_um, expected, strict=True)
        ):
            raise ValueError(f'asked for {wavelengths_um}, expected {expected}')
        return self.inner.predict_velocity(signals, self.inner.wavelengths_um)


def _nominal_cells(grid_name: str, families: tuple[str, ...]) -> list[Cell]:
    cells = [
        c
        for c in expand_grid(get_grid(grid_name))
        if c.family in families and c.sim_name == NOMINAL_SIM
    ]
    missing = set(families) - {c.family for c in cells}
    if missing:
        raise ValueError(f'{grid_name} has no {NOMINAL_SIM} cell for {missing}')
    return cells


def evaluate_shifts(
    grid_name: str,
    families: tuple[str, ...],
    knob: Knob,
    values: tuple[float, ...],
    out: Path,
    runs_dir: Path,
    frozen_root: Path,
    device: str,
    test_seed: int,
) -> list[RunRecord]:
    """Score each family's nominal checkpoint at each shifted knob value.

    Records go to ``out`` (appended), with ``sim_name`` ``<knob>=<value>`` and
    ``cell_key`` the nominal cell's plus ``__shift-<knob>=<value>``.
    ``converged`` is recorded None: it lives in the grid's own artifact, not
    in the checkpoint. ``test_seed`` picks the frozen test draw (the grid's
    own seed reproduces its nominal score).
    """
    test = get_grid(grid_name).test_set
    records = []
    for cell in _nominal_cells(grid_name, families):
        est = CheckpointEstimator(runs_dir / grid_name / f'{cell.key}.ckpt', device)
        for value in values:
            cfg = shifted_config(cell.sim, knob, value)
            dataset = load_or_create(cfg, test_seed, test.n_shots, frozen_root)
            result = evaluate(
                _TrainedChannels(est, cfg.wavelengths_um),
                dataset,
                cfg.wavelengths_um,
                test.n_shots,
                False,
                cell.seed,
            )
            tag = f'{knob}={value:.6g}'
            if test_seed != test.seed:
                tag += f'@test{test_seed}'
            records.append(
                dataclasses.replace(
                    result.record,
                    cell_key=f'{cell.key}__shift-{tag}',
                    sim_name=tag,
                    converged=None,
                )
            )
    append_results(out, records)
    return records
