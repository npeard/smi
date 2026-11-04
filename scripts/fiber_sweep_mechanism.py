"""Mechanism tier of q3_fiber_sweep: why static fading costs the network most.

The analytic tier (fiber_sweep_analytics.json) predicted static fading free
and path drift dominant; TCAN-44k measured the reverse. This tests, from the
existing records plus one small checkpoint probe, the three candidate
mechanisms of the fading cost:

- H1, C_eff loss (C scales with |kappa|): the noise-free C sweep
  (q3_s0c_ceiling, TCN-44k) read at the sweep's C_eff, and the sign
  evidence ``sign_information`` at (C_eff, SNR - loss) matched to the nominal
  SNR ladder;
- H2, fringe-SNR loss: the nominal SNR ladder (q1_s4_snr, TCN-44k, 3 colors)
  read at 30 dB minus the sweep's SNR loss;
- H3, in-shot fading: the speckle-drift ladder, which adds in-shot fades;
- H4, colours fading apart: q3_fiber_fading's colour-correlated N = 8 arm and
  its one-colour N = 1 / 4 / 8 arms (``colour_fading_verdict``).

Each prediction is the share of learned signal (``explained_fraction``)
retained relative to the M = 1 point. Then per-shot: where the extra error
lands (``per_shot_excess``), and whether undoing each shot's amplitude
(``rescale_probe``) recovers it. Writes ``fiber_sweep_mechanism.json``.

Usage::

    pixi run python scripts/fiber_sweep_mechanism.py --help
"""

import argparse
import dataclasses
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from smi.core.paths import FROZEN_DIR, GRID_RUNS_DIR, RESULTS_DIR
from smi.data.schema import PD_KEYS
from smi.experiments.diagnostics import (
    channel_stats,
    colour_fading_verdict,
    explained_fraction,
    matched_retention,
    per_shot_excess,
    rescale_probe,
    shot_speed,
)
from smi.experiments.grids import GRIDS
from smi.experiments.runner import cell_key, expand_grid
from smi.theory.bounds import sign_information

W3 = (0.635, 0.6748, 0.515)
MODES = ('modes-1', 'modes-2', 'modes-4', 'modes-8')
SNR_RUNGS = (0.0, 10.0, 20.0, 30.0, 40.0)
C_RUNGS = (0.05, 0.1, 0.2, 0.4, 0.7, 0.9)
SIGN_SHOTS = 8  # matches channel_analytics.py: the mirror fit is the slow call


def _records() -> dict[str, dict]:
    """Cell key -> record, over every synthetic dataset."""
    data = json.loads((RESULTS_DIR / 'synthetic_results.json').read_text())
    return {r['cell_key']: r for lst in data['datasets'].values() for r in lst}


def _mean(rec: dict, metric: str) -> float:
    return rec['metrics'][metric]['mean']


def _frozen(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """(per-shot velocity RMS, per-shot per-color fringe std) of a frozen set."""
    path = FROZEN_DIR / f'{dataset}.h5'
    with h5py.File(path, 'r') as f:
        amp = np.stack([f[k][:].std(-1) for k in PD_KEYS], -1)
    return shot_speed(path), amp


def analyse(device: str) -> dict:
    """Every hypothesis test of the fiber sweep's fading cost."""
    recs = _records()
    per_shot = json.loads((RESULTS_DIR / 'per_shot_rmse.json').read_text())
    analytic = json.loads((RESULTS_DIR / 'fiber_sweep_analytics.json').read_text())
    points = analytic['points']
    cells = {
        c.sim_name: c
        for c in expand_grid(GRIDS['q3_fiber_sweep'])
        if c.family == 'tcan-44k' and c.seed == 0
    }
    nominal = cells['nominal'].sim

    def sweep_key(name: str) -> str:
        return cell_key('tcan-44k', W3, name, 0)

    speed, amp_free = _frozen(per_shot[sweep_key('nominal')]['dataset'])
    zero = float(speed.mean())
    sweep = {}
    for name in cells:
        rec = recs[sweep_key(name)]
        sweep[name] = {
            'rmse': _mean(rec, 'velocity_rmse'),
            'rmse_sign_invariant': _mean(rec, 'velocity_rmse_sign_invariant'),
        }
        sweep[name]['explained'] = explained_fraction(sweep[name]['rmse'], zero)

    c_curve = {
        c: explained_fraction(
            _mean(
                recs[cell_key('tcn-44k', W3, f'smi-a3-feedback_c={c:g}', 0)],
                'velocity_rmse',
            ),
            zero,
        )
        for c in C_RUNGS
    }
    snr_curve = {
        s: explained_fraction(
            _mean(
                recs[cell_key('tcn-44k', W3, f'q1-nominal-snr_db={s:g}', 0)],
                'velocity_rmse',
            ),
            zero,
        )
        for s in SNR_RUNGS
    }
    # Sign evidence of each SNR rung (C = 0.2) and each fiber point, so the
    # fiber points can be placed on the SNR ladder at matched sign evidence.
    sign_snr = {
        s: sign_information(dataclasses.replace(nominal, snr_db=s), SIGN_SHOTS, 0)
        for s in SNR_RUNGS
    }
    log_sign = {np.log10(v): snr_curve[s] for s, v in sign_snr.items()}

    def snr_eff(name: str) -> float:
        return nominal.snr_db - points[name]['snr_loss_db']

    nats = {
        name: sign_information(
            dataclasses.replace(
                nominal, feedback_c=points[name]['feedback_c_eff'], snr_db=snr_eff(name)
            ),
            SIGN_SHOTS,
            0,
        )
        for name in MODES
    }
    ref = points['modes-1']
    hypotheses = {
        name: {
            'feedback_c_eff': points[name]['feedback_c_eff'],
            'snr_loss_db': points[name]['snr_loss_db'],
            'sign_nats_per_shot': nats[name],
            'observed_retention': sweep[name]['explained']
            / sweep['modes-1']['explained'],
            'h1_c_sweep_retention': matched_retention(
                c_curve, ref['feedback_c_eff'], points[name]['feedback_c_eff']
            ),
            'h2_snr_retention': matched_retention(
                snr_curve, snr_eff('modes-1'), snr_eff(name)
            ),
            'h1_sign_evidence_retention': matched_retention(
                log_sign, float(np.log10(nats['modes-1'])), float(np.log10(nats[name]))
            ),
        }
        for name in MODES
    }

    base = np.array(per_shot[sweep_key('modes-1')]['velocity_rmse'])
    shots = {}
    for name in ('modes-2', 'modes-4', 'modes-8', 'path-2.8'):
        entry = per_shot[sweep_key(name)]
        _, amp = _frozen(entry['dataset'])
        rel = (amp / amp_free).mean(-1)
        shots[name] = per_shot_excess(
            base, np.array(entry['velocity_rmse']), speed, rel
        )
    _, amp8 = _frozen(per_shot[sweep_key('modes-8')]['dataset'])
    color_corr = np.corrcoef((amp8 / amp_free).T)
    shots['modes-8']['max_abs_corr_between_color_amplitudes'] = float(
        np.abs(color_corr[np.triu_indices(3, 1)]).max()
    )

    def ckpt(name: str) -> Path:
        return GRID_RUNS_DIR / 'q3_fiber_sweep' / f'{sweep_key(name)}.ckpt'

    m1, m8 = cells['modes-1'].sim, cells['modes-8'].sim
    stats1 = channel_stats(m1, 64, 7)
    probe = {
        'modes-1_model_on_modes-1': rescale_probe(
            ckpt('modes-1'), m1, stats1, 64, 1000, device
        ),
        'modes-1_model_on_modes-8': rescale_probe(
            ckpt('modes-1'), m8, stats1, 64, 1000, device
        ),
        'modes-8_model_on_modes-8': rescale_probe(
            ckpt('modes-8'), m8, channel_stats(m8, 64, 7), 64, 1000, device
        ),
    }

    def excess(rmse: float, floor: float) -> float:
        return rmse**2 - floor**2

    path = {
        name: {
            'network_excess_mse': excess(sweep[name]['rmse'], sweep['modes-1']['rmse']),
            'reference_excess_mse': excess(points[name]['rmse'], ref['rmse']),
        }
        for name in ('path-0.1', 'path-0.3', 'path-1', 'path-2.8')
    }
    fading = {
        name: explained_fraction(_mean(recs[key], 'velocity_rmse'), zero)
        for name, key in (
            ('modes-8-corr', sweep_key('modes-8-corr')),
            *(
                (
                    f'modes-{n}-1color',
                    cell_key('tcan-44k', W3[:1], f'modes-{n}-1color', 0),
                )
                for n in (1, 4, 8)
            ),
        )
    }
    h4 = {
        'explained': fading,
        **colour_fading_verdict(
            three_m1=sweep['modes-1']['explained'],
            three_m8=sweep['modes-8']['explained'],
            three_m8_corr=fading['modes-8-corr'],
            one_m1=fading['modes-1-1color'],
            one_m8=fading['modes-8-1color'],
        ),
    }
    return {
        'zero_predictor': zero,
        'sweep': sweep,
        'c_sweep_explained': {f'{c:g}': v for c, v in c_curve.items()},
        'snr_ladder_explained': {f'{s:g}': v for s, v in snr_curve.items()},
        'snr_ladder_sign_nats_per_shot': {f'{s:g}': v for s, v in sign_snr.items()},
        'fading_hypotheses': hypotheses,
        'per_shot': shots,
        'rescale_probe': probe,
        'path_drift_excess_mse': path,
        'h4_colour_decorrelation': h4,
        'notes': [
            'C sweep is noise-free (q3_s0c_ceiling, TCN-44k 30-epoch); SNR ladder is'
            ' the nominal point (q1_s4_snr, TCN-44k); the fiber sweep is TCAN-44k at'
            ' the nominal point. Retentions are relative to each curve, so the'
            ' family and floor differences cancel to first order only.',
            'Sign evidence is the fixed-trajectory upper bound (sign_information)'
            ' at the shot-mean C_eff and SNR loss, 8 shots, seed 0.',
            'rescale_probe draws fresh shots (seed 1000 through simulate_batch), not'
            ' the frozen set, so its raw scores differ slightly from the records.',
            'H4 arms (q3_fiber_fading) share the sweep frozen-set seed and motion;'
            ' the one-colour arms are scored against the same zero predictor.',
        ],
    }


def main() -> None:
    """Parse arguments and write the artifact."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    out = analyse(args.device)
    path = RESULTS_DIR / 'fiber_sweep_mechanism.json'
    path.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
