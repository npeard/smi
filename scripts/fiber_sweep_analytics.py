"""Analytic tier of q3_fiber_sweep: predicted cost of each fiber knob alone.

Per grid point: the coupling statistics (``coupling_stats``: E|kappa|^2, the
fringe-SNR loss, C_eff, in-shot fading, speckle-phase and path excursions) and
the linear-Gaussian perfectly-demodulated reference RMSE (``fiber_reference``,
caveats recorded beside it), against the free-space point. Writes
``fiber_sweep_analytics.json``. CPU, seeded.

Usage::

    pixi run python scripts/fiber_sweep_analytics.py --help
"""

import argparse
import json

import torch

from smi.core.paths import RESULTS_DIR
from smi.experiments.diagnostics import Knockouts, coupling_stats, fiber_reference
from smi.experiments.grids import GRIDS
from smi.experiments.runner import expand_grid

_ALL = Knockouts(fading=True, speckle=True, path=True)
_CAVEATS = [
    'Reference RMSE is not a bound in either direction: perfect demodulation'
    ' (fringe order known) is optimistic; linear-stationary-Gaussian estimation'
    ' is pessimistic for non-Gaussian speckle and the chirp drive.',
    'Fading enters the reference only via the shot-mean |kappa|^2, so it is'
    ' blind to in-shot fades and cycle slips; in_shot_fade_rel reports those.',
    'A static |kappa| is a fixed per-shot gain: it costs fringe SNR'
    ' (snr_loss_db) and C, not information about the phase law.',
]


def analyse(n_shots: int, seed: int) -> dict:
    """Coupling statistics and reference RMSE of every q3_fiber_sweep sim."""
    points = {}
    for cell in expand_grid(GRIDS['q3_fiber_sweep']):
        sim = cell.sim
        rec = fiber_reference(sim, _ALL, n_shots, seed)
        if sim.fiber is not None:
            rec |= coupling_stats(sim, n_shots, seed)
        points[cell.sim_name] = rec
    free = points['nominal']['rmse']
    for rec in points.values():
        rec['mse_ratio_to_free_space'] = (free / rec['rmse']) ** 2
    return {
        'n_shots': n_shots,
        'seed': seed,
        'reference_name': 'linear-Gaussian perfectly-demodulated reference RMSE',
        'caveats': _CAVEATS,
        'points': points,
    }


def main() -> None:
    """Parse arguments and write the artifact."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--shots', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    out = analyse(args.shots, args.seed)
    path = RESULTS_DIR / 'fiber_sweep_analytics.json'
    path.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
