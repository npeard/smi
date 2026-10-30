#!/usr/bin/env python
"""Numbers that synthetic.typ quotes in prose and in its sign-evidence table.

The document reads this JSON at compile time, so a change to
smi.analysis.bounds (or to the results artifacts) rebuilds the quoted values
with the figures instead of leaving the prose stale. Each value is written as
the string the document prints, at the precision it quotes; the calls repeat
the figure scripts' (plot_velocity_bounds.py, plot_sign_information.py,
plot_fisher_phase.py) so the prose and the plotted points agree. It lives
beside them for their shared ``_figcommon`` but is not a figure, so it is not
named ``plot_*``.

Usage: python synthetic_values.py --output build/synthetic_values.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import DATA_DIR, SYNTHETIC_WAVELENGTHS_UM

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from smi.analysis import bounds
from smi.synthetic.forward import NOMINAL_ALPHA, SimConfig

# Matches plot_velocity_bounds.py and plot_sign_information.py.
N_STEPS = 2000
N_SHOTS = 8
SEED = 0
NOMINAL_C = 0.2
TABLE_C = (0.05, 0.2, 0.7)
TABLE_ALPHA = (3.0, 5.0)
TABLE_K = (1, 3)
# The real-data causal TCN of report.typ's results table: the matched 44k
# configuration on the free-space held-out test split, mean over model seeds.
REAL_TCN_CONFIG = 'tcn-matched'


def sig(x: float, digits: int) -> str:
    """``x`` to ``digits`` significant figures, in fixed notation."""
    decimals = max(digits - 1 - math.floor(math.log10(abs(x))), 0)
    return f'{x:.{decimals}f}'


def sci(x: float) -> dict[str, str | int]:
    """``x`` to three significant figures as mantissa times 10^exponent."""
    mantissa, exponent = f'{x:.2e}'.split('e')
    return {'mantissa': mantissa, 'exponent': int(exponent)}


def velocity_bounds(prior: bounds.StateSpace, n_channels: int) -> tuple[float, float]:
    cfg = SimConfig.nominal(
        SYNTHETIC_WAVELENGTHS_UM[:n_channels], NOMINAL_C, NOMINAL_ALPHA
    )
    causal = float(bounds.pcrb_filter(prior, cfg, N_STEPS)[-1]) ** 0.5
    offline = float(bounds.bcrb_smoother(prior, cfg, N_STEPS)[N_STEPS // 2]) ** 0.5
    return causal, offline


def real_tcn_rmse() -> float:
    results = json.loads((DATA_DIR / 'learned_results.json').read_text('utf-8'))
    runs = [
        run['metrics']['velocity_rmse_um_per_ms']['mean']
        for run in results['datasets']['free-space']
        if run['config']['config_name'] == REAL_TCN_CONFIG
    ]
    return sum(runs) / len(runs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output

    prior = bounds.motion_prior(
        SimConfig.nominal(SYNTHETIC_WAVELENGTHS_UM, 0.0, NOMINAL_ALPHA)
    )
    causal_k1, _ = velocity_bounds(prior, 1)
    causal_k3, offline_k3 = velocity_bounds(prior, 3)
    info = [
        bounds.expected_information(
            SimConfig.nominal(SYNTHETIC_WAVELENGTHS_UM[:k], NOMINAL_C, NOMINAL_ALPHA)
        )
        for k in (1, 3)
    ]
    real = real_tcn_rmse()

    values = {
        'prior_psd_rms_db': sig(prior.psd_rms_db, 2),
        'prior_velocity_std': sig(prior.velocity_var() ** 0.5, 3),
        'phase_factor_c02': sig(bounds.phase_factor(0.2, NOMINAL_ALPHA), 3),
        'phase_factor_c07': sig(bounds.phase_factor(0.7, NOMINAL_ALPHA), 3),
        'causal_k3': sig(causal_k3, 2),
        'offline_k3': sig(offline_k3, 2),
        'info_ratio_k3_k1': sig(info[1] / info[0], 2),
        'causal_ratio_k3_k1': sig(causal_k3 / causal_k1, 2),
        'real_tcn_rmse': sig(real, 4),
        'real_gap': f'{real / causal_k3:.0f}',
        'sign_table': [
            {
                'c': f'{c:g}',
                'cells': [
                    sci(
                        bounds.sign_information(
                            SimConfig.nominal(SYNTHETIC_WAVELENGTHS_UM[:k], c, alpha),
                            N_SHOTS,
                            SEED,
                        )
                    )
                    for alpha in TABLE_ALPHA
                    for k in TABLE_K
                ],
            }
            for c in TABLE_C
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(values, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
