"""Write the Q1.1/Q1.2/Q2.3 analytic channel tables from smi.theory.bounds.

Q1.1: profiled velocity information against channel count N for independent
channel phases, on common random numbers: one set of trajectories, phases and
visibilities per seed, sliced to the first N channels and reused for every C,
so additivity is exact and the feedback ratio is a paired comparison; several
seeds give its Monte-Carlo spread. Q1.2: N distinct colors against N
copies of 635 nm sharing one phase, with quadrature-spread or zero offsets,
by local Fisher and by sign evidence. Q2.3: the shared-phase profiled
information and sign evidence of the three synthetic colors at the
phase-offset configurations shift_eval scores the checkpoints on. The report
reads the JSON through report/figures/synthetic_values.py.

Usage::

    pixi run python scripts/channel_analytics.py
"""

import argparse
import dataclasses
import json
import math
from pathlib import Path

from smi.core.acquisition import STEMLAB_125_14
from smi.core.paths import RESULTS_DIR
from smi.experiments.analytics import channels
from smi.theory.bounds import (
    IdealShots,
    fisher_from_shots,
    fisher_matrix,
    ideal_shots,
    phase_factor,
    profiled_information,
    sign_information,
)
from smi.theory.forward import SimConfig
from smi.theory.offsets import offsets_quadrature, random_offsets

COLORS_UM = (0.635, 0.6748, 0.515, 0.785, 0.850, 0.405)
SNR_DB = 30.0
ALPHA = 3.0
FEEDBACK_C = (0.2, 0.0)
FISHER_SHOTS = 16
# sign_information fits a mirror phase per shot and channel, so it is the
# slow call; 8 shots matches plot_sign_information.py.
SIGN_SHOTS = 8
SEED = 0
COUNT_SEEDS = tuple(range(8))
# shift_eval's smi-a3 cell: three colors, C = 0.2; its steps and indices.
SHIFT_COLORS_UM = (0.635, 0.6748, 0.515)
SHIFT_C = 0.2
SHIFT_STEPS = (0.0, 0.5, 1.0, math.pi / 3, 2.0, 3.0)
SHIFT_RANDOM = tuple(range(16))


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


def info(cfg: SimConfig) -> float:
    """Profiled per-sample velocity information of ``cfg``."""
    return profiled_information(fisher_matrix(cfg, FISHER_SHOTS, SEED), 0)


def channel_count() -> dict[str, object]:
    """Q1.1 rows per seed: info of the first N colors and the sum of singles."""
    out: dict[str, object] = {
        'phase_factor': {f'{c:g}': phase_factor(c, ALPHA) for c in FEEDBACK_C}
    }
    for seed in COUNT_SEEDS:
        shots = ideal_shots(config(COLORS_UM, 0.0, None), FISHER_SHOTS, seed)

        def info_of(idx: list[int], c: float, shots: IdealShots = shots) -> float:
            cfg = config(tuple(COLORS_UM[k] for k in idx), c, None)
            return profiled_information(fisher_from_shots(cfg, channels(shots, idx)), 0)

        per_c = {}
        for c in FEEDBACK_C:
            singles = [info_of([k], c) for k in range(len(COLORS_UM))]
            per_c[f'{c:g}'] = [
                {
                    'n': n,
                    'wavelengths_um': list(COLORS_UM[:n]),
                    'info': info_of(list(range(n)), c),
                    'sum_singles': sum(singles[:n]),
                }
                for n in range(1, len(COLORS_UM) + 1)
            ]
        out[str(seed)] = per_c
    return out


def shift_phase_information() -> dict[str, object]:
    """Q2.3: shared-phase profiled info at shift_eval's offset configurations."""
    configs = {
        f'phase_offset={v:.6g}': tuple(k * v for k in range(3)) for v in SHIFT_STEPS
    }
    configs |= {f'phase_random={i}': random_offsets(i, 3) for i in SHIFT_RANDOM}
    nominal = config(SHIFT_COLORS_UM, SHIFT_C, None)
    return {
        'nominal_independent': info(nominal),
        'nominal_sign_nats_per_shot': sign_information(nominal, SIGN_SHOTS, SEED),
        'rows': [
            {
                'tag': tag,
                'offsets_rad': list(offsets),
                'info': info(config(SHIFT_COLORS_UM, SHIFT_C, offsets)),
                'sign_nats_per_shot': sign_information(
                    config(SHIFT_COLORS_UM, SHIFT_C, offsets), SIGN_SHOTS, SEED
                ),
            }
            for tag, offsets in configs.items()
        ],
    }


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
        'count_seeds': list(COUNT_SEEDS),
        'q1_count': channel_count(),
        'q1_colors': colors_vs_array(),
        'q2_phase': shift_phase_information(),
    }
    out.write_text(json.dumps(results, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
