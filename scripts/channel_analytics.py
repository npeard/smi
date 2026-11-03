"""Write the Q1.1/Q1.2 analytic channel tables from smi.theory.bounds.

Q1.1: profiled velocity information against channel count N for independent
channel phases. Q1.2: N distinct colors against N copies of 635 nm sharing
one phase, with quadrature-spread or zero offsets, by local Fisher and by
sign evidence. The report reads the JSON through
report/figures/synthetic_values.py.

Usage::

    pixi run python scripts/channel_analytics.py
"""

import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np

from smi.core.acquisition import STEMLAB_125_14
from smi.core.paths import RESULTS_DIR
from smi.theory.bounds import fisher_matrix, sign_information
from smi.theory.forward import SimConfig

COLORS_UM = (0.635, 0.6748, 0.515, 0.785, 0.850, 0.405)
SNR_DB = 30.0
ALPHA = 3.0
FEEDBACK_C = (0.2, 0.0)
FISHER_SHOTS = 16
# sign_information fits a mirror phase per shot and channel, so it is the
# slow call; 8 shots matches plot_sign_information.py.
SIGN_SHOTS = 8
SEED = 0


def config(
    wavelengths_um: tuple[float, ...],
    feedback_c: float,
    offsets: tuple[float, ...] | None,
) -> SimConfig:
    """noise_free detector at SNR_DB: visibility 1, no drift, additive noise."""
    nominal = SimConfig.nominal(STEMLAB_125_14, wavelengths_um, feedback_c, ALPHA)
    free = SimConfig.noise_free(
        acquisition=nominal.acquisition,
        wavelengths_um=wavelengths_um,
        sample_rate=nominal.sample_rate,
        start_freq_hz=nominal.start_freq_hz,
        end_freq_hz=nominal.end_freq_hz,
        drive_gain=nominal.drive_gain,
        feedback_c=feedback_c,
        alpha=ALPHA,
    )
    return dataclasses.replace(free, snr_db=SNR_DB, channel_phase_offsets_rad=offsets)


def profiled_velocity_information(fisher: np.ndarray) -> float:
    """J_vv - J_vp J_pp^-1 J_pv: velocity information with the phases unknown."""
    j_vp = fisher[0, 1:]
    return float(fisher[0, 0] - j_vp @ np.linalg.solve(fisher[1:, 1:], j_vp))


def info(cfg: SimConfig) -> float:
    """Profiled per-sample velocity information of ``cfg``."""
    return profiled_velocity_information(fisher_matrix(cfg, FISHER_SHOTS, SEED))


def offsets_quadrature(n: int) -> tuple[float, ...]:
    """N = 2: (0, pi/2); otherwise 2 pi k / N."""
    if n == 2:
        return (0.0, math.pi / 2)
    return tuple(2 * math.pi * k / n for k in range(n))


def channel_count() -> dict[str, object]:
    """Q1.1 rows: info of the first N colors and the sum of their singles."""
    out = {}
    for c in FEEDBACK_C:
        singles = [info(config((w,), c, None)) for w in COLORS_UM]
        out[f'{c:g}'] = [
            {
                'n': n,
                'wavelengths_um': list(COLORS_UM[:n]),
                'info': info(config(COLORS_UM[:n], c, None)),
                'sum_singles': sum(singles[:n]),
            }
            for n in range(1, len(COLORS_UM) + 1)
        ]
    return out


def colors_vs_array() -> dict[str, object]:
    """Q1.2 rows: colors, 635 nm x N with spread offsets, and with zero offsets."""
    rows = []
    for c in FEEDBACK_C:
        for n in (2, 3):
            arms = {
                'colors': config(COLORS_UM[:n], c, None),
                'array_quadrature': config((0.635,) * n, c, offsets_quadrature(n)),
                'array_zero': config((0.635,) * n, c, (0.0,) * n),
            }
            for arm, cfg in arms.items():
                rows.append(
                    {
                        'feedback_c': c,
                        'n': n,
                        'arm': arm,
                        'info': info(cfg),
                        'sign_nats_per_shot': sign_information(cfg, SIGN_SHOTS, SEED),
                    }
                )
    return {'rows': rows}


def main() -> None:
    """Parse arguments and write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--out', type=Path, default=RESULTS_DIR / 'channel_analytics.json'
    )
    out = parser.parse_args().out
    results = {
        'snr_db': SNR_DB,
        'alpha': ALPHA,
        'fisher_shots': FISHER_SHOTS,
        'sign_shots': SIGN_SHOTS,
        'seed': SEED,
        'q1_count': channel_count(),
        'q1_colors': colors_vs_array(),
    }
    out.write_text(json.dumps(results, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
