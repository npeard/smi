"""Write the Q4 Sagnac feedback-gain table and curves from smi.theory.sagnac.

The report quotes these numbers (via report/figures/synthetic_values.py) and
plots the curves (plot_sagnac_fisher.py); computing them here once keeps the
prose, the table and the figure on one set of values.

Usage::

    pixi run python scripts/sagnac_table.py
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from smi.core.paths import RESULTS_DIR
from smi.theory.bounds import phase_factor
from smi.theory.sagnac import feedback_gains

# Quadrature bias of the loop reflectivity: dc/d(dphi) peaks there.
BIAS_RAD = math.pi / 2
ALPHAS = (0.0, 3.0, 5.0)
# Full-reflection feedback levels of the table; above 1 the bias lowers C_eff
# below 1, at the cost of a bounded swing (``allowed_swing``).
TABLE_C0 = (0.3, 0.5, 0.7, 0.9, 0.99, 1.1, 1.2, 1.3)
# The best-phase gain diverges as C_eff -> 1, so the curve stops short of it.
CURVE_C_EFF = tuple(float(x) for x in np.linspace(0.0, 0.95, 39))


def allowed_swing(c0: float, bias: float) -> float:
    """Largest |dphi - bias| keeping C0 cos(dphi / 2) < 1 (inf when C0 <= 1).

    Feedback C_eff = C0 cos(dphi / 2) grows as dphi falls below the bias, so
    the binding edge is dphi = 2 arccos(1 / C0).
    """
    if c0 <= 1.0:
        return math.inf
    return bias - 2 * math.acos(1 / c0)


def sagnac_results() -> dict[str, object]:
    """Table rows, dense curves and the conventional-SMI comparison."""
    c_bias = math.cos(BIAS_RAD / 2)
    table = []
    for alpha in ALPHAS:
        for c0 in TABLE_C0:
            avg, best = feedback_gains(c0, alpha, BIAS_RAD)
            swing = allowed_swing(c0, BIAS_RAD)
            table.append(
                {
                    'alpha': alpha,
                    'c0': c0,
                    'c_eff': c0 * c_bias,
                    'avg': avg,
                    'best': best,
                    # JSON has no inf: None marks an unconstrained swing.
                    'swing_rad': None if math.isinf(swing) else swing,
                    # The conventional SMI phase readout at the same C:
                    # 2 Phi(C, alpha), 1 at C = 0 (eq. phase-factor).
                    'smi_phase_gain': 2 * phase_factor(c0 * c_bias, alpha),
                }
            )
    curves = {}
    for alpha in ALPHAS:
        gains = [feedback_gains(c / c_bias, alpha, BIAS_RAD) for c in CURVE_C_EFF]
        curves[f'{alpha:g}'] = {
            'c_eff': list(CURVE_C_EFF),
            'avg': [g[0] for g in gains],
            'best': [g[1] for g in gains],
        }
    return {'bias_rad': BIAS_RAD, 'table': table, 'curves': curves}


def main() -> None:
    """Parse arguments and write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, default=RESULTS_DIR / 'sagnac_results.json')
    out = parser.parse_args().out
    out.write_text(json.dumps(sagnac_results(), indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
