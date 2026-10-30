"""One evaluator for every velocity estimator, on any wavelength subset.

A trained checkpoint and the classical ``inverse_fit`` decoder are scored by the
same code on the same shots, so every number in a results table is comparable.
The data is a frozen synthetic set or a file in the same HDF5 schema (an
acquisition); channels are always selected by wavelength.

Conventions that matter when reading the numbers:

* ``velocity_rmse`` is the per-shot RMSE of the signed prediction.
  ``velocity_rmse_sign_invariant`` takes, per shot, the better of ``+pred`` and
  ``-pred`` (see :func:`smi.analysis.metrics.sign_invariant_rmse`).
* NRMSE is that RMSE divided by the per-shot population std of the true
  velocity, for both variants.
* Per-sample errors are binned by the true ``|v|`` (10 quantile bins over the
  evaluated set) and by the phase distance to the nearest fringe extremum of
  the shortest selected wavelength. The sign-invariant bin errors use the sign
  that minimizes the shot's RMSE, applied to every sample of the shot.
"""

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import h5py
import numpy as np
import torch
from torch import Tensor

from smi.analysis.baseline_fit import PD_WAVELENGTHS_UM, inverse_fit
from smi.analysis.datasets import file_pd_channels, select_pd_keys
from smi.analysis.lit_module import LitModule
from smi.analysis.metrics import best_sign, sign_invariant_rmse, summarize_per_shot
from smi.analysis.models.factory import is_causal
from smi.analysis.results_artifact import RunRecord, append_results
from smi.analysis.synthetic_lit_module import SyntheticLitModule
from smi.redpitaya.redpitaya_config import RedPitayaConfig
from smi.synthetic.coil_driver import CoilDriver
from smi.synthetic.forward import MAX_SPEED_UM_PER_MS, SimConfig

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_PATH = (
    Path(__file__).resolve().parent / 'data' / 'synthetic_results.json'
)

# Shots per estimator call: bounds memory on 16384-sample shots; the result
# does not depend on it.
EVAL_CHUNK_SHOTS = 16
N_BINS = 10
# Wavelengths of a checkpoint and a request must agree to this (um); looser
# than float32 rounding, tighter than the 0.6748 vs 0.675 laser spelling.
WAVELENGTH_TOL_UM = 1e-3
# Margin on the true excursion for the inverse_fit search grid; the recipe
# baseline_fit's own evaluation uses.
DISPLACEMENT_LIMIT_MARGIN = 1.15
PRECISION = '32-true'


class Estimator(Protocol):
    """Anything that maps photodiode signals to a velocity waveform."""

    name: str
    model_type: str
    causal: bool
    receptive_field: int | None  # None: unbounded lookback
    total_params: int
    device: str
    config: dict[str, Any]

    def predict_velocity(
        self, signals: Tensor, wavelengths_um: tuple[float, ...]
    ) -> Tensor:
        """Raw ``[B, C, L]`` signals (``wavelengths_um`` order) -> ``[B, L]``."""
        ...


def load_lit_module(path: str | Path, *, device: str) -> LitModule:
    """Rebuild a checkpoint's Lightning module with its weights.

    The checkpoint's own ``Model`` wrapper carries the normalization buffers,
    so predictions are raw units in and out; nothing is re-normalized here.
    Keys carry a ``_orig_mod.`` infix when training used ``torch.compile``.

    Args:
        path: Lightning ``.ckpt`` file.
        device: Device to place the module on.

    Returns:
        The module in eval mode.
    """
    ckpt = torch.load(path, weights_only=False, map_location='cpu')
    hparams = ckpt['hyper_parameters']
    module = (
        SyntheticLitModule(**hparams, restoring_checkpoint=True)
        if 'sim' in hparams
        else LitModule(**hparams)
    )
    module.load_state_dict(
        {k.replace('._orig_mod.', '.'): v for k, v in ckpt['state_dict'].items()}
    )
    return module.to(device).eval()


def _checkpoint_wavelengths(module: LitModule) -> tuple[float, ...]:
    """The wavelengths a checkpoint was trained on, in channel order."""
    if isinstance(module, SyntheticLitModule):
        return tuple(module.sim_config.wavelengths_um)
    n = module.model_hparams['in_channels']
    return tuple(PD_WAVELENGTHS_UM.values())[:n]


class CheckpointEstimator:
    """A trained network, evaluated through its own normalizing wrapper."""

    def __init__(self, path: str | Path, device: str) -> None:
        self.module = load_lit_module(path, device=device)
        self.device = device
        hp = self.module.model_hparams
        inner = self.module.inner_model
        self.model_type = hp['type']
        self.name = f'{self.model_type}:{Path(path).stem}'
        self.wavelengths_um = _checkpoint_wavelengths(self.module)
        self.causal = is_causal(hp)
        # Recurrent models have no finite window; recording the shot length
        # instead would read as a 16384-sample lookback, so it stays None.
        field = getattr(inner, 'receptive_field', None)
        self.receptive_field = None if field is None else int(field)
        self.total_params = sum(p.numel() for p in inner.parameters())
        self.config = {
            k: v for k, v in hp.items() if k not in ('total_params', 'receptive_field')
        }

    def predict_velocity(
        self, signals: Tensor, wavelengths_um: tuple[float, ...]
    ) -> Tensor:
        """Run the checkpoint; refuses a channel subset it was not trained on."""
        if len(wavelengths_um) != len(self.wavelengths_um) or any(
            abs(a - b) > WAVELENGTH_TOL_UM
            for a, b in zip(wavelengths_um, self.wavelengths_um, strict=True)
        ):
            raise ValueError(
                f'checkpoint was trained on {self.wavelengths_um} um, '
                f'asked to evaluate {wavelengths_um} um'
            )
        with torch.no_grad():
            out = self.module.model(signals.to(self.device))
        return out[:, 0].cpu()


class InverseFitEstimator:
    """``baseline_fit.inverse_fit`` as an offline oracle decoder.

    It is told the true excursion (``displacement_limit_um``) and a speed bound
    (:data:`MAX_SPEED_UM_PER_MS`), which a deployed estimator would not know;
    its score is therefore a classical ceiling, not a fair competitor to a
    causal network. Offline: it sees the whole shot, in both directions.
    """

    name = 'inverse_fit'
    model_type = 'inverse_fit'
    causal = False
    total_params = 0
    # float64 numpy/torch; MPS has no float64.
    device = 'cpu'

    def __init__(self, sample_rate: float, displacement_limit_um: float) -> None:
        self.sample_rate = sample_rate
        self.displacement_limit_um = displacement_limit_um
        # The search fits the whole shot at once: no finite lookback.
        self.receptive_field: int | None = None
        self.config = {
            'displacement_limit_um': displacement_limit_um,
            'max_speed_um_per_ms': MAX_SPEED_UM_PER_MS,
        }

    @classmethod
    def for_dataset(cls, dataset: Path, n_shots: int | None) -> 'InverseFitEstimator':
        """Size the search grid from the stored displacement of the shots scored."""
        with h5py.File(dataset, 'r') as f:
            disp = np.asarray(f['displacement'][:n_shots], dtype=np.float64)
            sample_rate = float(f.attrs['sample_rate'])
        centered = disp - disp.mean(axis=1, keepdims=True)
        limit = float(np.abs(centered).max()) * DISPLACEMENT_LIMIT_MARGIN
        return cls(sample_rate, limit)

    def predict_velocity(
        self, signals: Tensor, wavelengths_um: tuple[float, ...]
    ) -> Tensor:
        """Decode displacement and return its derivative, ``[B, L]`` um/ms."""
        result = inverse_fit(
            signals.numpy(),
            self.sample_rate,
            list(wavelengths_um),
            self.displacement_limit_um,
            max_speed_um_per_ms=MAX_SPEED_UM_PER_MS,
            device='cpu',
        )
        return torch.from_numpy(result.velocity).float()


@dataclass
class EvalRecord:
    """A results record plus the per-shot arrays it summarizes."""

    record: RunRecord
    per_shot: dict[str, np.ndarray]


def _sample_rate(f: h5py.File) -> float:
    if 'sample_rate' in f.attrs:
        return float(f.attrs['sample_rate'])
    return RedPitayaConfig.SAMPLE_RATE_DEC1 / 256


def _iter_chunks(
    f: h5py.File,
    keys: list[str],
    fringe_wavelength_um: float,
    fringe_key: str,
    n_shots: int,
    with_signals: bool,
) -> Iterator[tuple[Tensor | None, Tensor, np.ndarray]]:
    """Yield ``(signals [b,C,L] | None, velocity [b,L], phase [b,L])`` chunks.

    Truth comes from the file when stored (frozen sets), else from the drive
    voltage through the coil model, as the training datasets do. The phase is
    the stored optical phase of the fringe channel, else ``4 pi d / lambda``
    from the displacement label, which lacks the unknown per-channel offset, so
    fringe-distance bins from such files are approximate.
    """
    driver = CoilDriver()
    for start in range(0, n_shots, EVAL_CHUNK_SHOTS):
        sl = slice(start, min(start + EVAL_CHUNK_SHOTS, n_shots))
        if 'velocity' in f:
            velocity = torch.from_numpy(np.asarray(f['velocity'][sl], dtype=np.float32))
            displacement = np.asarray(f['displacement'][sl], dtype=np.float64)
        else:
            volts = torch.from_numpy(np.asarray(f['RP1_CH1'][sl], dtype=np.float32))
            velocity, disp_t = driver.get_velocity_and_displacement_torch(
                volts, _sample_rate(f)
            )
            displacement = disp_t.double().numpy()
        if 'optical_phase' in f:
            channel = list(file_pd_channels(f)).index(fringe_key)
            # Column order is the file's channel order (pd_wavelengths_um).
            assert f['optical_phase'].shape[1] == len(file_pd_channels(f))
            phase = np.asarray(f['optical_phase'][sl, channel], dtype=np.float64)
        else:
            phase = 4 * np.pi * displacement / fringe_wavelength_um
        signals = None
        if with_signals:
            signals = torch.from_numpy(
                np.stack([np.asarray(f[k][sl], dtype=np.float32) for k in keys], axis=1)
            )
        yield signals, velocity.float(), phase


def evaluate(
    est: Estimator,
    dataset: Path,
    wavelengths_um: tuple[float, ...],
    n_shots: int | None,
    converged: bool,
    train_seed: int | None,
) -> EvalRecord:
    """Score ``est`` on the first ``n_shots`` shots of ``dataset``.

    Args:
        est: Velocity estimator.
        dataset: Frozen set or acquisition-schema HDF5 file.
        wavelengths_um: Channel subset, selected by wavelength and fed to the
            estimator in this order.
        n_shots: Shots to score; ``None`` for all.
        converged: Whether training converged, from the caller's training run
            (a checkpoint does not say). ``True`` for ``InverseFitEstimator``,
            which is not trained.
        train_seed: Seed the estimator was trained with; ``None`` for
            ``InverseFitEstimator``.

    Returns:
        The results record and the per-shot arrays behind it.

    Raises:
        ValueError: If a shot's true velocity has zero std (NRMSE undefined),
            or the estimator returns the wrong shape.
    """
    wavelengths_um = tuple(wavelengths_um)
    with h5py.File(dataset, 'r') as f:
        keys = select_pd_keys(f, wavelengths_um)
        n = len(f['RP1_CH1'] if 'RP1_CH1' in f else f['velocity'])
        n = n if n_shots is None else min(n_shots, n)
        shortest = int(np.argmin(wavelengths_um))
        fringe = (keys[shortest], wavelengths_um[shortest])

        # Pass 1: quantile edges of the true |v| need the whole evaluated set.
        speeds = np.concatenate(
            [
                v.abs().numpy().ravel()
                for _, v, _ in _iter_chunks(f, keys, fringe[1], fringe[0], n, False)
            ]
        )
        v_edges = np.quantile(speeds, np.linspace(0.0, 1.0, N_BINS + 1))
        del speeds
        p_edges = np.linspace(0.0, np.pi / 2, N_BINS + 1)

        per_shot: dict[str, list[Tensor]] = {
            k: []
            for k in (
                'velocity_rmse',
                'velocity_rmse_sign_invariant',
                'velocity_nrmse',
                'velocity_nrmse_sign_invariant',
            )
        }
        # [count, squared-error sum, sign-invariant squared-error sum] per bin
        acc = {name: np.zeros((3, N_BINS)) for name in ('v', 'p')}
        finite = True
        shot = 0
        for signals, velocity, phase in _iter_chunks(
            f, keys, fringe[1], fringe[0], n, True
        ):
            assert signals is not None
            pred = est.predict_velocity(signals, wavelengths_um)
            if pred.shape != velocity.shape:
                raise ValueError(
                    f'estimator returned {tuple(pred.shape)}, '
                    f'expected {tuple(velocity.shape)}'
                )
            finite &= bool(torch.isfinite(pred).all())
            rmse = torch.sqrt(torch.mean((pred - velocity) ** 2, dim=-1))
            rmse_si = sign_invariant_rmse(pred, velocity)
            std = velocity.std(dim=-1, unbiased=False)
            if (std == 0).any():
                flat = shot + int(torch.argmax((std == 0).int()))
                raise ValueError(
                    f'shot {flat} of {dataset} has zero velocity std; NRMSE is '
                    'undefined (invalid data)'
                )
            shot += len(std)
            for key, value in (
                ('velocity_rmse', rmse),
                ('velocity_rmse_sign_invariant', rmse_si),
                ('velocity_nrmse', rmse / std),
                ('velocity_nrmse_sign_invariant', rmse_si / std),
            ):
                per_shot[key].append(value)

            sq = ((pred - velocity) ** 2).numpy().astype(np.float64)
            sq_si = (
                ((best_sign(pred, velocity) * pred - velocity) ** 2)
                .numpy()
                .astype(np.float64)
            )
            half = np.abs(phase - np.pi * np.round(phase / np.pi))
            for name, idx in (
                ('v', np.digitize(velocity.abs().numpy(), v_edges[1:-1])),
                ('p', np.digitize(half, p_edges[1:-1])),
            ):
                flat = idx.ravel()
                acc[name][0] += np.bincount(flat, minlength=N_BINS)
                acc[name][1] += np.bincount(flat, weights=sq.ravel(), minlength=N_BINS)
                acc[name][2] += np.bincount(
                    flat, weights=sq_si.ravel(), minlength=N_BINS
                )

        attrs = dict(f.attrs)
        sim_hash = (
            SimConfig(**json.loads(attrs['sim_config'])).hash()
            if 'sim_config' in attrs
            else None
        )

    arrays = {k: torch.cat(v).numpy() for k, v in per_shot.items()}
    binned = {
        'abs_velocity': _table_from_sums(acc['v'], v_edges),
        'fringe_distance': _table_from_sums(acc['p'], p_edges),
    }
    record = RunRecord(
        model_type=est.model_type,
        config=est.config,
        total_params=est.total_params,
        receptive_field=est.receptive_field,
        receptive_field_unbounded=est.receptive_field is None,
        sees_future=not est.causal,
        seed=-1 if train_seed is None else train_seed,
        device=est.device,
        dataset=Path(dataset).stem,
        metrics={k: summarize_per_shot(v) for k, v in arrays.items()},
        estimator=est.name,
        wavelengths_um=list(wavelengths_um),
        precision=PRECISION,
        converged=converged,
        train_seed=train_seed,
        test_seed=int(attrs.get('seed', -1)),
        finite_predictions=finite,
        sim_config_hash=sim_hash,
        simulator_version=(
            int(attrs['simulator_version']) if 'simulator_version' in attrs else None
        ),
        binned=binned,
    )
    return EvalRecord(record, arrays)


def _table_from_sums(sums: np.ndarray, edges: np.ndarray) -> list[dict[str, float]]:
    """Per-bin table from accumulated ``[count, sq_sum, sq_sum_sign_invariant]``."""
    n, sq, sq_si = sums
    with np.errstate(invalid='ignore', divide='ignore'):
        rmse = np.sqrt(sq / n)
        rmse_si = np.sqrt(sq_si / n)
    return [
        {
            'lo': float(edges[i]),
            'hi': float(edges[i + 1]),
            'n': int(n[i]),
            'rmse': float(rmse[i]),
            'rmse_sign_invariant': float(rmse_si[i]),
        }
        for i in range(N_BINS)
    ]


def evaluate_to_artifact(
    est: Estimator,
    dataset: Path,
    wavelengths_um: tuple[float, ...],
    n_shots: int | None,
    converged: bool,
    train_seed: int | None,
    results_path: Path,
) -> EvalRecord:
    """:func:`evaluate`, then append the record to the results artifact."""
    result = evaluate(est, dataset, wavelengths_um, n_shots, converged, train_seed)
    append_results(results_path, [result.record])
    return result
