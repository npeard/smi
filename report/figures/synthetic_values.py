#!/usr/bin/env python
"""Numbers that report.typ quotes in its synthetic prose and tables.

The document reads this JSON at compile time, so a change to
smi.theory.bounds (or to the results artifacts) rebuilds the quoted values
with the figures instead of leaving the prose stale. Learned-model values
come from the runner's records in synthetic_results.json, through the same
cell lookup as plot_exit_check.py and plot_feedback_curve.py; the one
exception, the zero predictor's error, reads the alpha = 0 control's frozen
test set. Each value is written as the string the document prints, at the
precision it quotes; the calls repeat the figure scripts'
(plot_velocity_bounds.py, plot_sign_information.py, plot_fisher_phase.py) so
the prose and the plotted points agree. It lives
beside them for their shared ``_figcommon`` but is not a figure, so it is not
named ``plot_*``.

Usage: python synthetic_values.py --output build/synthetic_values.json
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import sys
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import (
    CEILING_GRID,
    DISPLACEMENT_METRICS,
    FIBER_ARMS,
    MISMATCH_SHOTS,
    NOISE_KNOBS,
    Q1_BINNED_JSON,
    Q1_FAMILIES,
    Q1_MULTI,
    Q1_S2_GRID,
    Q1_S2_SIMS,
    Q1_SINGLES,
    Q1_SNR_DB,
    Q1_SUBSETS,
    Q2_CAUSAL,
    Q2_NAMED,
    Q2_OFFLINE,
    Q2_SIMS,
    Q2_STEMS,
    Q3_BUDGETS,
    Q3_FEEDBACK_C,
    R1_COLORS_UM,
    R1_SEEDS,
    R4_FAMILIES,
    R_FAMILIES,
    R_FAMILY_LABELS,
    RESULTS_DIR,
    S1D_FAMILIES,
    SHIFT_DEFAULT_TEST_SEED,
    STALL_TOLERANCE,
    SYNTHETIC_WAVELENGTHS_UM,
    LadderRow,
    c_sim,
    displacement_pending,
    flow_rows,
    matched_length_cells,
    mean_metric,
    mismatch_spectra,
    noise_ladder,
    noise_seeds,
    oracle_cell,
    q2_record,
    r4_cells,
    r_cell,
    s1d_record,
    shift_scores,
    shot_se,
    synthetic_cell,
    synthetic_records,
    velocity_bounds,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from smi.core.acquisition import STEMLAB_125_14
from smi.data.frozen import DEFAULT_ROOT, load_or_create
from smi.experiments.grids import GRIDS
from smi.experiments.results_artifact import RunRecord
from smi.experiments.runner import ORACLE_FAMILY, cell_key, expand_grid
from smi.learning.configs import MODEL_CONFIGS, R8_FIBER
from smi.learning.models.factory import create_model
from smi.learning.models.sliding_window import SlidingWindowModel
from smi.learning.synthetic_lit_module import VAL_SEED
from smi.theory import bounds
from smi.theory.forward import (
    NOMINAL_ALPHA,
    NOMINAL_PHASE_DRIFT_RAD_PER_SQRT_S,
    SimConfig,
)

NOMINAL_C = 0.2
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


def nominal_bounds(
    prior: bounds.StateSpace, wavelengths_um: tuple[float, ...]
) -> tuple[float, float]:
    """Causal and offline bounds of ``wavelengths_um`` at the nominal point."""
    return velocity_bounds(
        prior,
        SimConfig.nominal(STEMLAB_125_14, wavelengths_um, NOMINAL_C, NOMINAL_ALPHA),
    )


def real_tcn_rmse() -> float:
    results = json.loads((RESULTS_DIR / 'learned_results.json').read_text('utf-8'))
    runs = [
        run['metrics']['velocity_rmse_um_per_ms']['mean']
        for run in results['datasets']['free-space']
        if run['config']['config_name'] == REAL_TCN_CONFIG
    ]
    return sum(runs) / len(runs)


# Key stems of the learned configurations the real-data prose quotes.
REAL_CONFIGS = {
    'barland-matched': 'barland',
    'scnn-matched': 'scnn44',
    'tcn-matched': 'tcn44',
    'scnn-matched-small': 'scnn19',
    'mlp-matched': 'mlp',
    'tcn-matched-small': 'tcn19',
    'tcn-repeated': 'tcn36',
    'tcn-varying': 'tcn28',
    'lightgbm': 'lgbm',
    'ridge': 'ridge',
}
REAL_DATASETS = {'free-space': 'fs', 'mmfiber': 'mmf'}


def real_values() -> dict[str, str]:
    """Real-data values the prose and tables quote, from the committed records.

    Every RMSE is the mean over a run's test shots, for the learned rows and
    the decode alike, so the two compare under one statistic; a learned
    configuration's value is then the mean over its seeds. NRMSE and the
    decode's Pearson r are medians over shots, because a near-motionless
    shot's NRMSE runs into the hundreds and drags any mean.
    """
    learned = json.loads((RESULTS_DIR / 'learned_results.json').read_text('utf-8'))
    baseline = json.loads((RESULTS_DIR / 'baseline_results.json').read_text('utf-8'))
    out: dict[str, str] = {}
    vel: dict[tuple[str, str], float] = {}
    for dataset, ds in REAL_DATASETS.items():
        runs = learned['datasets'][dataset]
        seeds: dict[str, list[dict[str, Any]]] = {}
        for run in runs:
            seeds.setdefault(REAL_CONFIGS[run['config']['config_name']], []).append(run)
        for stem, group in seeds.items():
            v = [r['metrics']['velocity_rmse_um_per_ms']['mean'] for r in group]
            d = [r['metrics']['displacement_rmse_um']['mean'] for r in group]
            vel[stem, ds] = sum(v) / len(v)
            out[f'rl_{stem}_v_{ds}'] = f'{vel[stem, ds]:.4f}'
            out[f'rl_{stem}_d_{ds}'] = f'{sum(d) / len(d):.4f}'
            out[f'rl_{stem}_spread_{ds}'] = f'{max(v) - min(v):.4f}'
            out[f'rl_{stem}_params'] = f'{group[0]["total_params"]:,}'
        every = [r['metrics']['velocity_rmse_um_per_ms']['mean'] for r in runs]
        out[f'rl_all_mean_{ds}'] = f'{sum(every) / len(every):.4f}'
        out[f'rl_best_{ds}'] = f'{min(every):.4f}'
        out[f'rl_runs_{ds}'] = str(len(every))

        m2 = baseline['datasets'][dataset]['method2']
        scale = baseline['datasets'][dataset]['signal_scale']
        vel['decode', ds] = m2['velocity_rmse_um_per_ms']['mean']
        out[f'rd_v_{ds}'] = f'{vel["decode", ds]:.4f}'
        out[f'rd_d_{ds}'] = f'{m2["displacement_rmse_um"]["mean"]:.4f}'
        out[f'rd_vnrmse_{ds}'] = sig(m2['velocity_nrmse']['median'], 3)
        out[f'rd_dnrmse_{ds}'] = sig(m2['displacement_nrmse']['median'], 3)
        out[f'rd_dnrmse_mean_{ds}'] = sig(m2['displacement_nrmse']['mean'], 3)
        out[f'rd_dnrmse_max_{ds}'] = f'{m2["displacement_nrmse"]["max"]:.0f}'
        out[f'rd_r_{ds}'] = f'{m2["displacement_correlation"]["median"]:.2f}'
        zero = scale['true_velocity_rms_um_per_ms']['mean']
        out[f'rs_vrms_{ds}'] = f'{zero:.3f}'
        out[f'rs_drms_{ds}'] = f'{scale["true_displacement_rms_um"]["mean"]:.3f}'
        out[f'rs_drms_min_{ds}'] = f'{scale["true_displacement_rms_um"]["min"]:.3f}'
        out[f'rd_over_zero_{ds}'] = f'{vel["decode", ds] / zero:.2f}'
        out[f'rd_over_zero_pct_{ds}'] = pct(vel['decode', ds] / zero - 1)

    def ratio(stem: str) -> float:
        return vel[stem, 'mmf'] / vel[stem, 'fs']

    def lower_pct(a: str, b: str, ds: str) -> str:
        return pct(1 - vel[a, ds] / vel[b, ds])

    stems = [*REAL_CONFIGS.values(), 'decode']
    out |= {f'rl_{stem}_ratio': f'{ratio(stem):.2f}' for stem in stems}
    out |= {
        'real_decode_nrmse': out['rd_vnrmse_fs'],
        'rd_over_ridge_fs': f'{vel["decode", "fs"] / vel["ridge", "fs"]:.2f}',
        'rd_over_barland_fs': f'{vel["decode", "fs"] / vel["barland", "fs"]:.1f}',
        'rl_scnn_over_tcn_pct_fs': lower_pct('scnn44', 'tcn44', 'fs'),
        'rl_barland_over_scnn_pct_fs': lower_pct('barland', 'scnn44', 'fs'),
        'rl_barland_over_tcn_pct_fs': lower_pct('barland', 'tcn44', 'fs'),
        'rl_barland_over_tcn_pct_mmf': lower_pct('barland', 'tcn44', 'mmf'),
        'rl_field_cut_pct': lower_pct('tcn19', 'tcn36', 'fs'),
        'rl_field29_gain': f'{vel["tcn28", "fs"] - vel["tcn36", "fs"]:.4f}',
        'rl_lgbm_over_mlp_fs': f'{vel["lgbm", "fs"] / vel["mlp", "fs"]:.2f}',
        'rl_ratio_lo': f'{min(ratio(s) for s in REAL_CONFIGS.values()):.2f}',
        'rl_ratio_hi': f'{max(ratio(s) for s in REAL_CONFIGS.values()):.2f}',
    }
    rho = spearmanr(
        [vel[s, 'fs'] for s in REAL_CONFIGS.values()],
        [ratio(s) for s in REAL_CONFIGS.values()],
    ).statistic
    out['rl_ratio_spearman'] = f'{rho:.2f}'
    params = {
        s: int(out[f'rl_{s}_params'].replace(',', ''))
        for s in ('tcn19', 'tcn28', 'tcn36')
    }
    out['rl_field29_capacity_pct'] = pct(params['tcn36'] / params['tcn28'] - 1)
    out['rl_tcn19_fewer_pct'] = pct(1 - params['tcn19'] / params['tcn36'])
    return out


def pct(x: float) -> str:
    """``x`` as a whole-number percentage."""
    return f'{100 * x:.0f}'


def lo_hi(prefix: str, xs: Iterable[Any], fmt: Callable[[Any], str]) -> dict[str, str]:
    """``<prefix>_lo`` and ``<prefix>_hi``: the range of ``xs``, as printed."""
    xs = list(xs)
    return {f'{prefix}_lo': fmt(min(xs)), f'{prefix}_hi': fmt(max(xs))}


def epochs(record: RunRecord) -> int:
    """Epochs ``record`` trained; 0 for the untrained oracle."""
    return record.epochs_run or 0


def test_shots(record: RunRecord) -> int:
    """Shot count of the frozen set, from its dataset key ``<hash>_seed<s>_n<n>``.

    The record stores the set's seed but not its size; the key is written by
    smi.data.frozen.frozen_set_path.
    """
    return int(record.dataset.rsplit('_n', 1)[1])


def zero_predictor_rmse(records: dict[str, RunRecord], grid: str, key: str) -> float:
    """Test RMSE of predicting zero velocity on cell ``key``'s test set.

    That is each shot's velocity RMS, averaged over shots. The sim config
    comes from expanding the grid that produced the record, hash-checked, so
    the emitter cannot drift from the runner or score another set after a
    grid edit.
    """
    record = records[key]
    (cfg,) = [cell.sim for cell in expand_grid(GRIDS[grid]) if cell.key == key]
    if record.sim_config_hash != cfg.hash():
        raise ValueError(f'{key}: record was run under another sim config')
    if record.test_seed is None:
        raise ValueError(f'{record.cell_key}: no test seed recorded')
    path = load_or_create(cfg, record.test_seed, test_shots(record), DEFAULT_ROOT)
    with h5py.File(path, 'r') as f:
        velocity = f['velocity'][:]
    return float(np.sqrt((velocity**2).mean(axis=1)).mean())


def shot_length(records: dict[str, RunRecord]) -> int:
    """Samples per simulated shot, read off the alpha = 0 control's frozen set."""
    record = records[cell_key('tcn-44k-T90', SYNTHETIC_WAVELENGTHS_UM, 'smi-a0', 0)]
    (cfg,) = [
        cell.sim
        for cell in expand_grid(GRIDS[CEILING_GRID])
        if cell.key == record.cell_key
    ]
    assert record.test_seed is not None
    path = load_or_create(cfg, record.test_seed, test_shots(record), DEFAULT_ROOT)
    with h5py.File(path, 'r') as f:
        return int(f['velocity'].shape[1])


def exit_check_values(records: dict[str, RunRecord]) -> dict[str, str]:
    """Training budget of section 1.1's three exit-check runs."""
    family = 'tcn-44k-E20'
    signed = synthetic_cell(records, family, 'michelson-signed')
    return {'exit_max_epochs': str(signed.max_epochs)}


def q3_values(records: dict[str, RunRecord], zero: float) -> dict[str, str | bool]:
    """Values section 2 quotes from the fixed-budget accuracy-against-C grids.

    Args:
        records: The synthetic records.
        zero: The zero predictor's test RMSE (``zero_predictor_rmse``).
    """

    def rmse(family: str, sim: str, metric: str = 'velocity_rmse') -> float:
        return mean_metric(synthetic_cell(records, family, sim), metric)

    def sweep(family: str, metric: str = 'velocity_rmse') -> list[float]:
        return [rmse(family, c_sim(c), metric) for c in Q3_FEEDBACK_C]

    t90, t30 = (budget.family for budget in Q3_BUDGETS)
    curve = sweep(t90)
    t30_curve = sweep(t30)
    michelson = rmse(t90, 'michelson', 'velocity_rmse_sign_invariant')
    a0 = [synthetic_cell(records, f, 'smi-a0') for f in (t90, t30)]
    t30_epochs = {
        c: synthetic_cell(records, t30, c_sim(c)).epochs_run for c in Q3_FEEDBACK_C
    }

    worst = [
        synthetic_cell(records, t90, c_sim(c)).metrics['velocity_nrmse']['max']
        for c in Q3_FEEDBACK_C
    ]
    # A sign-flipped zero-mean shot scores NRMSE 2. The text says the worst
    # shot reaches it from some C upward, so that C must start a suffix.
    flipped = [w >= 2 for w in worst]
    n_flip = flipped.index(True)
    assert all(flipped[n_flip:]), worst

    def slowest(family: str) -> list[float]:
        """Slowest-|v|-decile sign-invariant RMSE of Michelson and each C."""
        sims = ('michelson', *(c_sim(c) for c in Q3_FEEDBACK_C))
        bins = [synthetic_cell(records, family, sim).binned for sim in sims]
        return [b['abs_velocity'][0]['rmse_sign_invariant'] for b in bins if b]

    return {
        # The text claims the 90-epoch curve falls monotonically in RMSE; the
        # document asserts this, so a regenerated artifact that breaks it
        # fails the build.
        'q3_monotone': all(b < a for a, b in itertools.pairwise(curve)),
        'q3_c_top': f'{Q3_FEEDBACK_C[-1]:g}',
        'q3_gain_top': sig(michelson / curve[-1], 2),
        'q3_a0_epochs': str(max(epochs(r) for r in a0)),
        # The 30-epoch curve's one inversion, which the text attributes to the
        # C = 0.4 cell's longer run; asserted by the document like q3_monotone.
        'q3_t30_c04_below': all(
            t30_curve[Q3_FEEDBACK_C.index(0.4)] < t30_curve[Q3_FEEDBACK_C.index(c)]
            for c in (0.7, 0.9)
        ),
        'q3_t30_c04_epochs': str(t30_epochs[0.4]),
    }


def noise_values(
    records: dict[str, RunRecord], prior: bounds.StateSpace
) -> dict[str, object]:
    """Values the noise-ladder section quotes, from its three grids."""

    def rmse(record: RunRecord) -> float:
        return mean_metric(record, 'velocity_rmse')

    ladder = noise_ladder(records)
    base_runs = noise_seeds(records, 'ladder-base')
    nominal_runs = noise_seeds(records, 'nominal')
    base = [rmse(r) for r in base_runs]
    nominal = [rmse(r) for r in nominal_runs]
    base_mean = sum(base) / len(base)
    excess = sum(nominal) / len(nominal) - base_mean

    def at_nominal(field: str) -> LadderRow:
        (row,) = [row for row in ladder[field] if row.nominal]
        return row

    def share(field: str) -> float:
        """Fraction of the nominal point's excess over the base this knob gives."""
        return (rmse(at_nominal(field).record) - base_mean) / excess

    phase, snr = 'phase_drift_rad_per_sqrt_s', 'snr_db'
    minor = [f for f in ladder if f not in (phase, snr)]
    minor_shares = [share(f) for f in minor]

    width = max(nominal) - min(nominal)
    # The equal-length grid: every cell trained the same length, its cap.
    e60_cells, _ = matched_length_cells(records)
    e60 = {name: rmse(r) for name, r in e60_cells.items()}
    e60_nominal = e60['nominal']
    e60_nodrift = e60['nominal-no-phasedrift']
    return {
        'loo_e60_cut': pct(1 - e60_nodrift / e60_nominal),
        # The text: at equal length the cut exceeds the nominal seed spread,
        # and the ranking base < no-phase-drift < nominal is unchanged.
        'loo_e60_outside': e60_nodrift < e60_nominal - width,
        'noise_phase_rad': sig(NOMINAL_PHASE_DRIFT_RAD_PER_SQRT_S, 3),
        # The ranking the text states: phase drift, then SNR, then the rest.
        'noise_ranking': share(phase) > share(snr) > max(minor_shares),
    }


# The text says no family comes near the Michelson oracle; a ratio this large
# is well clear of the seed range, which a bare `oracle < best` would not be.
ORACLE_MARGIN = 1.5


def q2_table(records: dict[str, RunRecord]) -> dict[str, object]:
    """The rows of section Q2's table, and checks on what they are labeled.

    Context is each family's ``receptive_field``, read off its model built
    from the matched config it trained under (tests/test_receptive_field.py
    pins each against a gradient probe).
    """
    models = {
        stem: create_model(MODEL_CONFIGS[f'matched/{stem}'].model_config)
        for stem in Q2_STEMS
    }

    def cell(r: RunRecord | None, sim: str) -> str:
        if r is None:
            return 'pending'
        if not r.converged:
            # Labeled stalled only if it never left the zero-output plateau;
            # any other non-convergence would need its own wording.
            assert r.best_monitor is not None
            assert r.zero_predictor_monitor is not None
            assert r.best_monitor >= (1 - STALL_TOLERANCE) * r.zero_predictor_monitor, (
                r.cell_key
            )
            return 'stalled'
        return f'{mean_metric(r, Q2_SIMS[sim]):.3f}'

    def context(r: RunRecord | None, stem: str) -> str:
        if r is None:
            return '--'
        field = models[stem].receptive_field
        if field is None:
            return 'unbounded' if r.causal else 'whole shot'
        return str(field)

    def rows(group: tuple[tuple[str, str], ...], causal: bool) -> list[dict]:
        out = []
        for stem, label in group:
            cells = {sim: q2_record(records, stem, sim) for sim in Q2_SIMS}
            assert all(r is None or r.causal is causal for r in cells.values()), stem
            out.append(
                {
                    'label': label,
                    'context': context(cells['smi-a3'], stem),
                    'cells': [cell(r, sim) for sim, r in cells.items()],
                }
            )
        return out

    stalled = sorted(
        (stem, sim)
        for stem in Q2_STEMS
        for sim in Q2_SIMS
        if (r := q2_record(records, stem, sim)) is not None and not r.converged
    )
    fields = {stem: models[stem].receptive_field for stem in Q2_NAMED}
    sliding = {
        stem: fields[stem]
        for stem in Q2_NAMED
        if isinstance(models[stem], SlidingWindowModel)
    }
    # The families whose context is a fixed window, which the text says are
    # matched; the sliding-window families' wider spans are named apart.
    matched = {
        stem: field
        for stem, field in fields.items()
        if stem not in sliding and field is not None
    }
    return {
        'q2_stalled_named': set(stalled)
        == {
            ('lstm-44k', 'michelson'),
            ('lstm-44k', 'smi-a3'),
            ('lstm-bidir-44k', 'michelson'),
            ('lstm-bidir-44k', 'smi-a3'),
            ('neuraloperator-offline-44k', 'smi-a3'),
        },
        'q2_stall_tol_pct': pct(STALL_TOLERANCE),
        'q2_sliding_named': set(sliding) == {'barland-44k', 'mlp-44k'},
        'q2_tcn_field': str(fields['tcn-44k']),
        'q2_matched_named': set(matched)
        == {
            'tcn-44k',
            'tcan-44k',
            'patchtransformer-causal-44k',
            'unettransformer-causal-44k',
            'neuraloperator-causal-44k',
            'scnn-44k',
            'patchtransformer-offline-44k',
            'unettransformer-offline-44k',
        },
    }


def seed_resolution(records: dict[str, RunRecord]) -> float:
    """The ranking resolution: the noise-free base's seed range over its mean.

    Section 2B's grid: TCN-44k, three channels, C = 0.2, the 30-epoch budget.
    Two cells closer than this are not ranked apart.
    """
    base = [
        mean_metric(r, 'velocity_rmse') for r in noise_seeds(records, 'ladder-base')
    ]
    return (max(base) - min(base)) / (sum(base) / len(base))


def separated(a: float, b: float, resolution: float) -> bool:
    """Whether ``b`` exceeds ``a`` by more than ``resolution``."""
    return b / a - 1 > resolution


def at_edge(gain: float, resolution: float) -> bool:
    """Whether a relative ``gain`` is about ``resolution``, within a factor 1.5.

    For gains the text calls "at the edge of the seed range": neither clearly
    resolved nor clearly inside it, so a rerun that moves the gain well past
    either side should change the wording.
    """
    return resolution / 1.5 <= gain <= 1.5 * resolution


def q2_values(records: dict[str, RunRecord], zero: float) -> dict[str, object]:
    """Values section Q2 quotes from the S1 and S1m family grids.

    Args:
        records: The synthetic records.
        zero: The zero predictor's test RMSE (``zero_predictor_rmse``).
    """

    def trained(sim: str, among: list[str]) -> dict[str, float]:
        """Test score of each converged cell of ``among`` on ``sim``."""
        return {
            s: mean_metric(r, Q2_SIMS[sim])
            for s in among
            if (r := q2_record(records, s, sim)) is not None and r.converged
        }

    resolution = seed_resolution(records)

    def apart(a: float, b: float) -> bool:
        return separated(a, b, resolution)

    mich, smi = trained('michelson', Q2_STEMS), trained('smi-a3', Q2_STEMS)
    mich_named = trained('michelson', Q2_NAMED)
    oracle_mich, oracle_smi = (oracle_cell(records, sim) for sim in Q2_SIMS)
    oracle = mean_metric(oracle_mich, 'velocity_rmse_sign_invariant')
    oracle_si = mean_metric(oracle_smi, 'velocity_rmse_sign_invariant')
    tcn_smi = smi['tcn-44k']

    weakest = max(mich_named, key=mich_named.__getitem__)
    rest = [v for s, v in mich_named.items() if s != weakest]
    best = min(smi, key=smi.__getitem__)
    runner_up = min(v for s, v in smi.items() if s != best)
    # The zero predictor scored the alpha = 0 control's frozen set, which
    # shares the oracle's test seed, so it scores the same velocities.
    a0 = synthetic_cell(records, 'tcn-44k-T90', 'smi-a0')
    assert a0.test_seed == oracle_smi.test_seed
    oracle_signed = mean_metric(oracle_smi, 'velocity_rmse')
    return {
        'q2_seed_range_pct': pct(resolution),
        # Michelson: every trained family but the offline neural operator lies
        # within the seed range, and that one lies above the rest.
        'q2_mich_tied': weakest == 'neuraloperator-offline-44k'
        and not apart(min(rest), max(rest))
        and apart(max(rest), mich_named[weakest]),
        'q2_oracle_beats_all': ORACLE_MARGIN * oracle <= min(mich.values()),
        'q2_tcan_gain': sig(tcn_smi / smi['tcan-44k'], 3),
        'q2_tcan_best': best == 'tcan-44k' and apart(smi[best], runner_up),
        # Barland's 498-sample span ties SCNN's 85: the wider context shows no gain.
        'q2_barland_ties_scnn': not apart(
            min(smi['barland-44k'], smi['scnn-44k']),
            max(smi['barland-44k'], smi['scnn-44k']),
        ),
        'q2_oracle_worse_than_zero': oracle_signed > zero,
        'q2_learned_beat_oracle_smi': max(smi.values()) < oracle_si,
    }


# The text says a Michelson single channel scores at the zero predictor, and
# that pairs and triples sit far below it.
AT_ZERO_TOL = 0.03
FAR_FROM_ZERO = 0.6
# The text calls an error curve flat in extremum distance when no bin departs
# from the curve's mean over bins by more than this.
FLAT_TOL = 0.1


def lowest(scores: dict[tuple, float], among: list[tuple]) -> tuple:
    """The subset of ``among`` with the lowest score."""
    return min(among, key=scores.__getitem__)


def q1_s2_values(records: dict[str, RunRecord], resolution: float) -> dict[str, object]:
    """Values section Q1 quotes from the noise-free channel-subset grid (S2).

    Args:
        records: The synthetic records.
        resolution: The seed range (``seed_resolution``).
    """
    triple = SYNTHETIC_WAVELENGTHS_UM
    zero = zero_predictor_rmse(
        records, Q1_S2_GRID, cell_key('tcn-44k', triple, 's2-smi-a3', 0)
    )

    def scores(family: str, sim: str) -> dict[tuple, float]:
        seed = None if family == ORACLE_FAMILY else 0
        return {
            w: mean_metric(records[cell_key(family, w, sim, seed)], Q1_S2_SIMS[sim])
            for w, _ in Q1_SUBSETS
        }

    mich = {f: scores(f, 's2-michelson') for f in Q1_FAMILIES}
    smi = {f: scores(f, 's2-smi-a3') for f in Q1_FAMILIES}
    oracle = scores(ORACLE_FAMILY, 's2-michelson')
    mich_single = [mich[f][w] for f in Q1_FAMILIES for w in Q1_SINGLES]
    mich_multi = [mich[f][w] for f in Q1_FAMILIES for w in Q1_MULTI]
    single_records = [
        records[cell_key(f, w, 's2-michelson', 0)]
        for f in Q1_FAMILIES
        for w in Q1_SINGLES
    ]
    # How far each Michelson single channel's best validation error sits below
    # the zero predictor's: the quantity the runner's stall test reads.
    monitor_gap = []
    for r in single_records:
        assert r.best_monitor is not None
        assert r.zero_predictor_monitor is not None
        monitor_gap.append(1 - r.best_monitor / r.zero_predictor_monitor)

    def ranked_singles(family: str) -> list[tuple]:
        return sorted(Q1_SINGLES, key=smi[family].__getitem__)

    def runner_up(family: str) -> float:
        first, second = ranked_singles(family)[:2]
        return smi[family][second] / smi[family][first] - 1

    def gain(family: str) -> float:
        best = smi[family][lowest(smi[family], Q1_SINGLES)]
        return best / min(smi[family][w] for w in Q1_MULTI)

    return {
        'q1_zero': sig(zero, 2),
        # Michelson: one channel scores at the zero predictor, sign-invariant,
        # in both families; every pair and the triple sit far below it.
        'q1_mich_single_at_zero': all(
            abs(x / zero - 1) < AT_ZERO_TOL for x in mich_single
        ),
        'q1_mich_multi_far': max(mich_multi) < FAR_FROM_ZERO * zero,
        'q1_oracle_single_above_zero': min(oracle[w] for w in Q1_SINGLES) > zero,
        # The stall test passes the single channels, so the text calls them
        # trained to the zero predictor, not stalled.
        'q1_mich_single_converged': all(r.converged for r in single_records),
        'q1_stall_tol_pct': pct(STALL_TOLERANCE),
        # The text: 515 nm is the best single channel in both families.
        'q1_515_best': all(ranked_singles(f)[0] == (0.515,) for f in Q1_FAMILIES),
        'q1_runner_up_tcn_at_edge': at_edge(runner_up('tcn-44k'), resolution),
        # Every multi-channel set beats the best single channel by more than
        # the seed range, in both families.
        'q1_multi_beats_single': all(
            separated(smi[f][w], smi[f][lowest(smi[f], Q1_SINGLES)], resolution)
            for f in Q1_FAMILIES
            for w in Q1_MULTI
        ),
    }


def q1_binned_values(
    records: dict[str, RunRecord], prior: bounds.StateSpace
) -> dict[str, object]:
    """Values section Q1 quotes from the binned errors (smi.experiments.q1_binned).

    The JSON holds the TCN on 515 nm alone, on 635 + 515 nm and on all three
    channels, noise-free and at the nominal point at 30 dB.
    """
    binned = json.loads(Q1_BINNED_JSON.read_text('utf-8'))
    # It was rescored from the checkpoints the records came from.
    assert all(
        math.isclose(c['rmse'], mean_metric(records[k], 'velocity_rmse'), rel_tol=1e-3)
        for k, c in binned['cells'].items()
    )
    cells = {
        (c['condition'], tuple(c['wavelengths_um'])): c
        for c in binned['cells'].values()
    }
    one, _pair, three = (0.515,), (0.635, 0.515), SYNTHETIC_WAVELENGTHS_UM

    def by_distance(condition: str, w: tuple, channel: str) -> list[float]:
        return cells[condition, w]['fringe_distance'][channel]['rmse']

    def rise(condition: str, w: tuple, channel: str = '515') -> float:
        """Nearest-to-extremum bin over the farthest, minus one."""
        curve = by_distance(condition, w, channel)
        return curve[0] / curve[-1] - 1

    def flat(condition: str, w: tuple) -> bool:
        curve = by_distance(condition, w, '515')
        mean = sum(curve) / len(curve)
        return all(abs(x / mean - 1) <= FLAT_TOL for x in curve)

    nf_one = by_distance('noise_free', one, '515')
    nf_three = by_distance('noise_free', three, '515')
    gain_near, gain_far = nf_one[0] / nf_three[0], nf_one[-1] / nf_three[-1]

    v_gain = [
        a / b
        for a, b in zip(
            cells['noise_free', one]['abs_velocity']['rmse'],
            cells['noise_free', three]['abs_velocity']['rmse'],
            strict=True,
        )
    ]
    slow_one, slow_three, slow_zero = (
        cells['noise_free', w]['abs_velocity'][key][0]
        for w, key in ((one, 'rmse'), (three, 'rmse'), (one, 'zero'))
    )
    return {
        'q1_bin_width': f'{binned["fringe_distance_edges_fringes"][1]:.3g}',
        # The text: noise-free, neither curve rises toward the extremum.
        'q1_bin_nf_flat': flat('noise_free', one)
        and flat('noise_free', three)
        and rise('noise_free', one) <= 0,
        # The text: at 30 dB both curves rise toward the extremum only a
        # little, within FLAT_TOL.
        'q1_bin_nom_small_rise': flat('nominal_30db', one)
        and flat('nominal_30db', three),
        'q1_bin_gain_not_near': gain_near < gain_far,
        # The text: in the slowest bin both models score above zero.
        'q1_bin_v_slow_above_zero': min(slow_one, slow_three) > slow_zero,
        # The text: no gain in the slowest bin, the largest above 1.5x.
        'q1_bin_v_shape': v_gain[0] < 1 and max(v_gain) > 1.5,
    }


def q1_s4_values(records: dict[str, RunRecord], resolution: float) -> dict[str, object]:
    """Values section Q1 quotes from the nominal-point SNR sweep (S4), TCN only."""
    triple = SYNTHETIC_WAVELENGTHS_UM

    def s4(w: tuple, snr: float, seed: int | None = 0) -> RunRecord:
        family = 'tcn-44k' if seed is not None else ORACLE_FAMILY
        return records[cell_key(family, w, f'q1-nominal-snr_db={snr:g}', seed)]

    rmse = {
        snr: {w: mean_metric(s4(w, snr), 'velocity_rmse') for w, _ in Q1_SUBSETS}
        for snr in Q1_SNR_DB
    }
    # Fraction by which the best multi-channel set beats the best single.
    gain = {
        snr: 1 - min(row[w] for w in Q1_MULTI) / min(row[w] for w in Q1_SINGLES)
        for snr, row in rmse.items()
    }
    oracle = [
        mean_metric(s4(w, snr, None), 'velocity_rmse')
        for w in ((0.515,), triple)
        for snr in Q1_SNR_DB
    ]
    zero = zero_predictor_rmse(
        records, Q1_S2_GRID, cell_key('tcn-44k', triple, 's2-smi-a3', 0)
    )
    at30 = rmse[30.0]
    nominal = [mean_metric(r, 'velocity_rmse') for r in noise_seeds(records, 'nominal')]
    return {
        # The text: at 30 dB the gain is several times the nominal seed range,
        # and the pair without 515 nm does no better than 515 nm alone.
        'q1_s4_30_outside': gain[30.0]
        > 3 * (max(nominal) - min(nominal)) / min(nominal)
        and not separated(at30[0.515,], at30[0.635, 0.6748], resolution),
        'q1_s4_oracle_worse': min(oracle) > zero,
    }


def spectral_values() -> dict[str, str]:
    """Numbers quoted beside the data/target mismatch figure.

    The PSDs share one bin grid, so sums over bins are power shares.
    """
    spectra = mismatch_spectra(False)

    return {
        'spec_shots': str(MISMATCH_SHOTS),
        'spec_bin_hz': f'{spectra.freq_hz[0]:.0f}',
    }


def budget_range(cells: list[RunRecord], per_epoch: int) -> dict[str, str]:
    """Epochs and shots the ``cells`` trained on, as printed ranges."""
    n = [epochs(r) for r in cells]
    return {
        **lo_hi('epochs', n, str),
        **lo_hi('kshots', (x * per_epoch / 1e3 for x in n), lambda x: f'{x:.0f}'),
        'cells': str(len(cells)),
    }


def methods_values(
    records: dict[str, RunRecord], shot_length: int
) -> dict[str, object]:
    """The data and training protocol behind every trained cell, from the records.

    Budgets are read off the records the cells wrote and the grids that ran
    them, and each is asserted single-valued where the text states one value.
    """
    trained = [r for r in records.values() if r.steps_per_epoch is not None]
    grids = GRIDS.values()
    convergence = MODEL_CONFIGS['matched/tcn-44k'].training_config['convergence']

    def one(values: Iterable[Any]) -> Any:
        (value,) = set(values)
        return value

    batch = one(r.batch_size for r in trained)
    steps = one(r.steps_per_epoch for r in trained)
    crop = one(r.train_crop_length for r in trained if r.train_crop_length is not None)
    per_epoch = batch * steps
    val_steps = one(g.synthetic.val_steps for g in grids)
    test_set = one(g.test_set for g in grids)
    full_length = sorted({r.model_type for r in trained if r.train_crop_length is None})
    # Each matched family's stopping rule, which the text states once.
    for name, config in MODEL_CONFIGS.items():
        if not (name.startswith('matched/') and name.endswith('-44k')):
            continue
        rule = config.training_config['convergence']
        assert {k: rule[k] for k in convergence if k != 'monitor'} == {
            k: v for k, v in convergence.items() if k != 'monitor'
        }, name
    return {
        'm_batch': str(batch),
        'm_steps': str(steps),
        'm_shots_per_epoch': f'{per_epoch:,}',
        'm_crop': str(crop),
        'm_shot_length': str(shot_length),
        'm_samples_per_epoch': f'{per_epoch * crop / 1e6:.1f}',
        'm_full_samples_per_epoch': f'{per_epoch * shot_length / 1e6:.1f}',
        # The text names the offline neural operator as the one full-shot family.
        'm_full_length_is_no': full_length == ['NeuralOperator'],
        **{
            f'm_t{cycle}_{k}': v
            for cycle in (30, 90)
            for k, v in budget_range(
                [r for r in trained if cycle == r.T_0], per_epoch
            ).items()
        },
        'm_val_batches': str(val_steps),
        'm_val_shots': str(val_steps * batch),
        'm_val_seed': str(VAL_SEED),
        'm_test_shots': str(test_set.n_shots),
        'm_test_seed': str(test_set.seed),
        'm_test_sets': str(len({r.dataset for r in records.values()})),
        'm_patience': str(convergence['patience_epochs']),
        'm_min_rel_delta_pct': pct(convergence['min_rel_delta']),
        'm_grace': str(convergence['min_epochs']),
        'm_lr': f'{one(r.lr for r in trained):g}',
        'm_device': one(r.device.split(':')[0] for r in trained).upper(),
        'm_precision': one(r.precision for r in trained).removesuffix('-true'),
    }


# The cells whose displacement RMSE the text quotes: the 90-epoch C curve and
# its Michelson reference, and the S1 families at both sims.
DISPLACEMENT_CURVE = 'tcn-44k-T90'
# The families the text names lowest on displacement at C = 0.2.
DISPLACEMENT_LOWEST = {
    'patchtransformer-offline-44k',
    'tcan-44k',
    'unettransformer-offline-44k',
}


def displacement_values(records: dict[str, RunRecord]) -> dict[str, object]:
    """Displacement RMSE of the quoted cells, or the pending flag alone.

    Families are ranked at the displacement seed range, measured on the same
    seeds as ``seed_resolution``: integration changes the spread, so the
    velocity resolution does not carry over.
    """
    curve = {
        c: synthetic_cell(records, DISPLACEMENT_CURVE, c_sim(c)) for c in Q3_FEEDBACK_C
    }
    michelson = synthetic_cell(records, DISPLACEMENT_CURVE, 'michelson')
    families = [
        r
        for stem in Q2_STEMS
        for sim in Q2_SIMS
        if (r := q2_record(records, stem, sim)) is not None
    ]
    if displacement_pending([*curve.values(), michelson, *families]):
        warnings.warn(
            'displacement RMSE is PENDING: no record carries it yet, and the '
            'report marks it so (rerun plan item a backfills it)',
            stacklevel=1,
        )
        return {'disp_pending': True}
    signed = DISPLACEMENT_METRICS['velocity_rmse']
    base = [mean_metric(r, signed) for r in noise_seeds(records, 'ladder-base')]
    resolution = (max(base) - min(base)) / (sum(base) / len(base))

    def trained(sim: str) -> dict[str, float]:
        return {
            s: mean_metric(r, DISPLACEMENT_METRICS[Q2_SIMS[sim]])
            for s in Q2_STEMS
            if (r := q2_record(records, s, sim)) is not None and r.converged
        }

    smi = trained('smi-a3')
    tcn = smi['tcn-44k']
    ranked = sorted(smi, key=smi.__getitem__)
    lowest = [smi[s] for s in ranked[: len(DISPLACEMENT_LOWEST)]]
    return {
        # The three lowest are the named ones, unordered by the seed range,
        # and each sits below the TCN by more than it.
        'disp_lowest_named': set(ranked[: len(DISPLACEMENT_LOWEST)])
        == DISPLACEMENT_LOWEST
        and not separated(lowest[0], lowest[-1], resolution)
        and separated(lowest[-1], tcn, resolution)
    }


def s1d_values(records: dict[str, RunRecord]) -> dict[str, object]:
    """Values section Q2 quotes from the displacement-target grid (S1d).

    Each S1d cell is compared with its velocity-target S1 cell on the same
    test shots, both scored on displacement: signed at C = 0.2, sign-invariant
    on Michelson (``Q2_SIMS``).
    """

    def ratio(r: RunRecord) -> float:
        assert r.best_monitor is not None
        assert r.zero_predictor_monitor is not None
        return r.best_monitor / r.zero_predictor_monitor

    stems = [stem for stem, _ in S1D_FAMILIES]
    smi = [s1d_record(records, s, 'smi-a3') for s in stems]
    mich = {s: s1d_record(records, s, 'michelson') for s in stems}
    si = DISPLACEMENT_METRICS['velocity_rmse_sign_invariant']
    disp = {s: mean_metric(r, si) for s, r in mich.items()}
    vel = {s: mean_metric(q2_record(records, s, 'michelson'), si) for s in stems}
    lstms = ('lstm-44k', 'lstm-bidir-44k')
    mamba = q2_record(records, 'mamba-44k', 'smi-a3')
    assert mamba is not None

    def three(x: float) -> str:
        return sig(x, 3)

    return {
        # The text: at C = 0.2 every retargeted cell, the TCN control
        # included, stalls; on Michelson every one trains.
        's1d_smi_stalled': all(
            not r.converged and ratio(r) >= 1 - STALL_TOLERANCE for r in smi
        ),
        's1d_mich_trained': all(
            r.converged and ratio(r) < 1 - STALL_TOLERANCE for r in mich.values()
        ),
        # The text: retargeting helps the LSTMs and costs the neural operator
        # and the TCN, and every retargeted cell stays behind the
        # velocity-trained TCN.
        's1d_shape': all(disp[s] < vel[s] for s in lstms)
        and all(disp[s] > vel[s] for s in ('neuraloperator-offline-44k', 'tcn-44k'))
        and min(disp.values()) > vel['tcn-44k'],
        's1d_mamba_trained': mamba.converged,
    }


def prediction_values(records: dict[str, RunRecord], zero: float) -> dict[str, str]:
    """Synthetic error over the zero predictor's, for the real-data predictions.

    A ratio to the zero predictor transfers to the real acquisitions better
    than an absolute RMSE, whose scale is set by the motion prior.
    """

    def ratio(sim_name: str) -> str:
        runs = [mean_metric(r, 'velocity_rmse') for r in noise_seeds(records, sim_name)]
        return sig(sum(runs) / len(runs) / zero, 2)

    def triple_ratio(sim: str, metric: str) -> str:
        """The best single channel's TCN error over the three-channel set's."""
        rmse = {
            w: mean_metric(records[cell_key('tcn-44k', w, sim, 0)], metric)
            for w in [*Q1_SINGLES, SYNTHETIC_WAVELENGTHS_UM]
        }
        best = min(rmse[w] for w in Q1_SINGLES)
        return sig(best / rmse[SYNTHETIC_WAVELENGTHS_UM], 3)

    return {
        # The two ends of prediction 6: noise-free, and the nominal 30 dB point.
        'pred_triple_free': triple_ratio('s2-smi-a3', Q1_S2_SIMS['s2-smi-a3']),
        'pred_triple_30': triple_ratio('q1-nominal-snr_db=30', 'velocity_rmse'),
        'pred_base_ratio': ratio('ladder-base'),
        'pred_c09_ratio': ratio('nominal-c0.9'),
    }


def _spread(xs: list[float]) -> float:
    """Seed range over seed mean."""
    return (max(xs) - min(xs)) / float(np.mean(xs))


def r1_values(records: dict[str, RunRecord], zero: float) -> dict[str, object]:
    """Q1.1 R1 and Q1.2 R2: RMSE against channel count, pair and array."""
    data = json.loads((RESULTS_DIR / 'channel_analytics.json').read_text())
    seeds = [data['q1_count'][str(k)]['0.2'] for k in data['count_seeds']]
    info = [float(np.mean([sd[i]['info'] for sd in seeds])) for i in range(6)]

    def runs(family: str, n: int, sim: str) -> list[float]:
        ws = R1_COLORS_UM[:n]
        return [
            mean_metric(records[k], 'velocity_rmse')
            for seed in R1_SEEDS
            if (k := cell_key(family, ws, sim, seed)) in records
        ]

    def rmse(family: str, n: int, sim: str) -> float:
        return float(np.mean(runs(family, n, sim)))

    tcan, unet = R_FAMILIES
    out: dict[str, object] = {}
    rows = []
    for i in range(6):
        n = i + 1
        row = {'n': str(n), 'lam': f'{R1_COLORS_UM[i] * 1e3:.1f}'}
        row['bound'] = f'{(info[i] / info[0]) ** -0.5:.2f}'
        row['bstep'] = '' if i == 0 else pct((info[i] / info[i - 1]) ** -0.5 - 1)
        for fam, tag in ((tcan, 'tcan'), (unet, 'unet')):
            for sim, stag in (('s2-smi-a3', 'nf'), ('nominal', 'nom')):
                row[f'{tag}_{stag}'] = f'{rmse(fam, n, sim):.3f}'
                row[f'{tag}_{stag}_ratio'] = (
                    f'{rmse(fam, n, sim) / rmse(fam, 1, sim):.2f}'
                )
                row[f'{tag}_{stag}_step'] = (
                    '' if i == 0 else pct(rmse(fam, n, sim) / rmse(fam, n - 1, sim) - 1)
                )
        rows.append(row)
    out['r1_table'] = rows
    bound = [(info[i] / info[0]) ** -0.5 for i in range(6)]
    tnf = [rmse(tcan, n, 's2-smi-a3') / rmse(tcan, 1, 's2-smi-a3') for n in range(1, 7)]
    tnom = [rmse(tcan, n, 'nominal') / rmse(tcan, 1, 'nominal') for n in range(1, 7)]
    # Noise-free TCAN follows the bound's ratio to within 25% at every N >= 3.
    out['r1_tcan_nf_tracks'] = all(0.75 < tnf[i] / bound[i] < 1.25 for i in range(2, 6))
    # ... but beats it at N = 2, where the learned gain exceeds the Fisher one.
    out['r1_tcan_nf_n2_beats'] = tnf[1] < 0.85 * bound[1]
    out['r1_tcan_nom_monotone'] = all(a > b for a, b in itertools.pairwise(tnom))
    out['r1_unet_nf_spread_big'] = _spread(runs(unet, 3, 's2-smi-a3')) > 0.3
    out['r1_unet_nom_flat'] = (
        max(rmse(unet, n, 'nominal') for n in range(1, 7))
        / min(rmse(unet, n, 'nominal') for n in range(1, 7))
        < 1.25
    )
    return out


def r2_values(records: dict[str, RunRecord], zero: float) -> dict[str, object]:
    """Q1.2 R2: 0.635 um x N with quadrature offsets against N colors."""
    tcan, unet = R_FAMILIES
    out: dict[str, object] = {}
    r2 = []
    within = []
    for n in (2, 3):
        for sim, stag in (('s2-smi-a3', 'noise-free'), ('nominal', 'nominal')):
            row = {'n': str(n), 'sim': stag}
            for fam, tag in ((tcan, 'tcan'), (unet, 'unet')):
                arr = r_cell(records, fam, (0.635,) * n, f'{sim}-1color')
                col = r_cell(records, fam, R1_COLORS_UM[:n], sim)
                a, c = (mean_metric(r, 'velocity_rmse') for r in (arr, col))
                row[f'{tag}_arr'] = f'{a:.4f}' if a < 0.01 else f'{a:.3f}'
                row[f'{tag}_col'] = f'{c:.3f}'
                row[f'{tag}_ratio'] = f'{c / a:.2g}' if c / a > 3 else f'{c / a:.2f}'
                row[f'{tag}_se'] = f'{math.hypot(shot_se(arr), shot_se(col)):.3f}'
                out[f'r2_{tag}_{stag[:3]}_ratio_n{n}'] = c / a
                if stag == 'nominal':
                    within.append(
                        abs(c - a) < 2 * math.hypot(shot_se(arr), shot_se(col))
                    )
            r2.append(row)
    out['r2_table'] = r2
    nf_ratios = [
        out[f'r2_{t}_noi_ratio_n{n}'] for t in ('tcan', 'unet') for n in (2, 3)
    ]
    out['r2_nf_array_wins'] = min(nf_ratios) > 5
    out['r2_nom_parity'] = all(within)
    # Each arm has its own frozen set; equal zero-predictor RMSE shows the
    # motion draw is the same in all of them.
    zs = [
        zero_predictor_rmse(records, grid, cell_key(tcan, ws, f'{sim}-1color', 0))
        for grid, ws in (
            ('q1_r2_colors', (0.635,) * 2),
            ('q1_r2_colors_n3', (0.635,) * 3),
        )
        for sim in ('s2-smi-a3', 'nominal')
    ]
    out['r2_zero_shared'] = max(abs(z - zero) for z in zs) < 1e-9
    for k in [k for k in out if k.startswith('r2_') and k[-3:] in ('_n2', '_n3')]:
        del out[k]
    return out


def r4_values(records: dict[str, RunRecord], zero: float) -> dict[str, object]:
    """Q2.2 R4: each knob alone at nominal, the stressed rungs, the nominal point."""
    cells = r4_cells(records)
    ladder = noise_ladder(records)

    def rmse(family: str, field: str, stressed: bool = False) -> float:
        return mean_metric(cells[family, field, stressed], 'velocity_rmse')

    knobs = [f for f in NOISE_KNOBS if (R_FAMILIES[0], f, False) in cells]
    out: dict[str, object] = {}
    rows = []
    for field in knobs:
        tcn = next(r for r in ladder[field] if r.nominal).record
        row = {
            'knob': NOISE_KNOBS[field],
            'tcn': f'{mean_metric(tcn, "velocity_rmse"):.3f}',
        }
        for fam in R4_FAMILIES:
            row[fam] = f'{rmse(fam, field):.3f}'
        rows.append(row)
    for label, field, stressed in (
        ('phase drift, 3.3x', 'phase_drift_rad_per_sqrt_s', True),
        ('white noise, 20 dB', 'snr_db', True),
        ('all nominal', 'nominal', False),
    ):
        row = {'knob': label, 'tcn': ''}
        for fam in R4_FAMILIES:
            row[fam] = f'{rmse(fam, field, stressed):.3f}'
        rows.append(row)
    out['r4_table'] = rows
    rank = {
        fam: sorted(knobs, key=lambda f, fam=fam: -rmse(fam, f)) for fam in R4_FAMILIES
    }
    neural = R4_FAMILIES[:3]
    out['r4_phase_first'] = all(
        rank[f][0] == 'phase_drift_rad_per_sqrt_s' for f in neural
    )
    # The predicted SNR-over-visibility order holds for UNetTransformer only.
    out['r4_vis_over_snr'] = [
        R_FAMILY_LABELS[f]
        for f in R4_FAMILIES
        if rmse(f, 'visibility_range') > rmse(f, 'snr_db')
    ] == ['TCAN', 'PatchTransformer (offline)', 'LightGBM']
    return out


def r7_values(records: dict[str, RunRecord], zero: float) -> dict[str, object]:
    """Q3.1 and Q3.2 R7: the fiber arms against free space and each other."""
    out: dict[str, object] = {}
    rows = []
    zeros = {
        arm: zero_predictor_rmse(
            records, 'q3_fiber', cell_key(R_FAMILIES[0], ws, arm, 0)
        )
        for arm, ws in FIBER_ARMS.items()
    }
    # The arms share trajectories, so every arm has the same zero predictor.
    out['r7_zero_shared'] = max(zeros.values()) - min(zeros.values()) < 1e-9
    for arm, ws in FIBER_ARMS.items():
        row = {'arm': arm}
        for fam, tag in zip(R_FAMILIES, ('tcan', 'unet'), strict=True):
            r = r_cell(records, fam, ws, arm)
            row[tag] = f'{mean_metric(r, "velocity_rmse"):.3f}'
            row[f'{tag}_oz'] = f'{mean_metric(r, "velocity_rmse") / zeros[arm]:.2f}'
            row[f'{tag}_se'] = f'{shot_se(r):.3f}'
            row[f'{tag}_ep'] = str(epochs(r))
            row[f'{tag}_conv'] = 'yes' if r.converged else 'no'
        rows.append(row)
    out['r7_table'] = rows

    per_shot = json.loads((RESULTS_DIR / 'per_shot_rmse.json').read_text('utf-8'))

    def diff(fam: str, a: str, b: str) -> tuple[float, float]:
        # The arms share trajectories shot for shot, so the paired SE of the
        # per-shot differences is the honest error of an arm difference.
        ra, rb = (
            np.asarray(per_shot[cell_key(fam, FIBER_ARMS[x], x, 0)]['velocity_rmse'])
            for x in (a, b)
        )
        for x, shots in ((a, ra), (b, rb)):
            record = mean_metric(
                r_cell(records, fam, FIBER_ARMS[x], x), 'velocity_rmse'
            )
            assert math.isclose(shots.mean(), record, rel_tol=1e-5), x
        d = ra - rb
        return float(d.mean()), float(d.std(ddof=1) / math.sqrt(d.size))

    tcan = R_FAMILIES[0]
    for name, a, b in (
        ('cost', 'fiber', 'nominal'),
        ('monitor', 'fiber-monitor', 'fiber-4color'),
        ('pbs', 'fiber-pbs', 'fiber'),
        ('fourth', 'fiber-4color', 'fiber'),
    ):
        d, se = diff(tcan, a, b)
        out[f'r7_tcan_{name}'] = f'{d:+.3f}'
        out[f'r7_tcan_{name}_se'] = f'{se:.3f}'
    fiber_oz = [
        mean_metric(r_cell(records, f, ws, arm), 'velocity_rmse') / zeros[arm]
        for f in R_FAMILIES
        for arm, ws in FIBER_ARMS.items()
        if arm != 'nominal'
    ]
    out['r7_all_near_zero'] = min(fiber_oz) > 0.9
    out['r7_tcan_all_converged'] = all(
        r_cell(records, tcan, ws, arm).converged for arm, ws in FIBER_ARMS.items()
    )
    return out


def sagnac_values() -> dict[str, Any]:
    """Q4 gains quoted in the prose, from scripts/sagnac_table.py's JSON."""
    rows = json.loads((RESULTS_DIR / 'sagnac_results.json').read_text())['table']

    def row(alpha: float, c0: float) -> dict[str, Any]:
        return next(r for r in rows if r['alpha'] == alpha and r['c0'] == c0)

    out = {}
    # The table of the Q4 section: one row per C_0, cells "avg / best" per alpha.
    alphas = sorted({r['alpha'] for r in rows})
    out['sagnac_table_alpha'] = [f'{a:g}' for a in alphas]
    out['sagnac_table'] = [
        {
            'c0': f'{r["c0"]:g} ({r["c_eff"]:.2f})',
            # Rows above C_0 = 1 hold only within this swing of the bias.
            'swing': '' if r['swing_rad'] is None else f'{r["swing_rad"]:.2f}',
            'cells': [
                f'{row(a, r["c0"])["avg"]:.2f} / {row(a, r["c0"])["best"]:.2f}'
                for a in alphas
            ],
            'smi': f'{row(3.0, r["c0"])["smi_phase_gain"]:.2f}',
        }
        for r in rows
        if r['alpha'] == alphas[0]
    ]
    return out


def channel_values() -> dict[str, Any]:
    """Q1.1 and Q1.2 analytic values, from scripts/channel_analytics.py's JSON."""
    data = json.loads((RESULTS_DIR / 'channel_analytics.json').read_text())
    count = data['q1_count']
    rows = data['q1_colors']['rows']

    def arm(c: float, n: int, name: str) -> dict[str, Any]:
        return next(
            r for r in rows if r['feedback_c'] == c and r['n'] == n and r['arm'] == name
        )

    seeds = [count[str(k)] for k in data['count_seeds']]
    n_max = len(seeds[0]['0.2'])

    def mean_info(c: str, i: int) -> float:
        return float(np.mean([sd[c][i]['info'] for sd in seeds]))

    dev = max(
        abs(r['info'] / r['sum_singles'] - 1)
        for sd in seeds
        for rs in sd.values()
        for r in rs
    )
    # Paired C = 0.2 over C = 0 ratios: same draws, so only C differs.
    ratios = np.array(
        [
            [sd['0.2'][i]['info'] / sd['0'][i]['info'] for i in range(n_max)]
            for sd in seeds
        ]
    )
    pf = count['phase_factor']
    info02 = [mean_info('0.2', i) for i in range(n_max)]
    out = {
        'ch_fisher_shots': str(data['fisher_shots']),
        'ch_count_seeds': str(len(seeds)),
        'ch_additive_exact': dev < 1e-9,
        'ch_fb_pred_pct': f'{100 * (pf["0.2"] / pf["0"] - 1):.1f}',
        'ch_fb_meas_pct': f'{100 * (ratios.mean() - 1):.1f}',
        # Seed-to-seed spread of one table entry's ratio.
        'ch_fb_spread_pct': f'{100 * ratios.std(ddof=1, axis=0).mean():.1f}',
        'ch_fb_se_pct': f'{100 * ratios.mean(1).std(ddof=1) / len(seeds) ** 0.5:.1f}',
        'ch_info_jump_405_pct': f'{100 * (info02[5] / info02[4] - 1):.0f}',
        'ch_info_jump_785_pct': f'{100 * (info02[3] / info02[2] - 1):.0f}',
        'ch_count_table': [
            {
                'n': str(i + 1),
                'lam': f'{seeds[0]["0.2"][i]["wavelengths_um"][-1] * 1e3:.1f}',
                'info02': f'{info02[i] / 1e7:.2f}',
                'info0': f'{mean_info("0", i) / 1e7:.2f}',
                'bound02': f'{(info02[i] / info02[0]) ** -0.5:.2f}',
            }
            for i in range(n_max)
        ],
        'ch_colors_table': [
            {
                'c': f'{r["feedback_c"]:g}',
                'n': str(r['n']),
                'arm': r['arm'],
                'info': f'{r["info"] / 1e7:.2f}',
                'sign': (
                    '0'
                    if r['sign_nats_per_shot'] < 1
                    else f'{r["sign_nats_per_shot"]:.2g}'
                ),
            }
            for r in rows
        ],
    }
    for n in (2, 3):
        col = arm(0.2, n, 'colors')['info']
        arrays = [arm(0.2, n, a)['info'] for a in ('array_quadrature', 'array_zero')]
        out[f'ch_colors_vs_array_n{n}_pct'] = (
            f'{100 * (col / max(arrays) - 1):+.0f} to '
            f'{100 * (col / min(arrays) - 1):+.0f}'
        )
    quad = arm(0.0, 3, 'array_quadrature')['sign_nats_per_shot']
    out['ch_sign_quad_n3_c0'] = sci(quad)
    out['ch_sign_ratio_n3'] = (
        f'{quad / arm(0.2, 3, "colors")["sign_nats_per_shot"]:.0f}'
    )
    return out


SHIFT_FALSIFY_TEMP_PCT = 10
SHIFT_FALSIFY_REALIGN_PCT = 20


def shift_values(prior: bounds.StateSpace) -> dict[str, str]:
    """Q2.3 values from shift_results.json and the falsifier thresholds."""
    nominal = shift_scores()[
        ('tcan-44k-T90', 'feedback_c', 0.2, SHIFT_DEFAULT_TEST_SEED)
    ]
    v_rms = prior.empirical_velocity_var**0.5
    out = {'shift_tcan_eps_cross': f'{nominal.rmse / v_rms:.2f}'}
    # Q2.3 real-data falsifier thresholds, in percent of the frozen model's
    # real error: a temperature-only change above the first falsifies the
    # scale prediction; a realignment that halves C below the second
    # falsifies the feedback one (the synthetic factor at C = 0.1 is >2).
    out['shift_falsify_temp_pct'] = str(SHIFT_FALSIFY_TEMP_PCT)
    out['shift_falsify_realign_pct'] = str(SHIFT_FALSIFY_REALIGN_PCT)
    return out


def _bounds_at(
    prior: bounds.StateSpace, cfg: SimConfig, **change: Any
) -> tuple[float, float]:
    """``velocity_bounds`` of ``cfg`` with ``change`` applied."""
    return velocity_bounds(prior, dataclasses.replace(cfg, **change))


def _equivalent_snr(
    prior: bounds.StateSpace, cfg: SimConfig, rmse: float, which: int
) -> float:
    """The SNR (dB) at which bound ``which`` (0 causal, 1 offline) equals ``rmse``.

    Bisection: both bounds fall monotonically with SNR, since J-bar scales as
    1 / sigma^2 (eq-info-avg).
    """
    lo, hi = -80.0, 120.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if _bounds_at(prior, cfg, snr_db=mid)[which] > rmse:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def q2_analytic_values(
    records: dict[str, RunRecord], prior: bounds.StateSpace
) -> dict[str, str]:
    """The Q2.1, Q2.1.1 and Q2.2 analytic tiers, from smi.theory.bounds."""
    out: dict[str, str] = {}
    sims = {'michelson': 0.0, 'smi-a3': NOMINAL_C}
    for sim, c in sims.items():
        cfg = SimConfig.nominal(
            STEMLAB_125_14, SYNTHETIC_WAVELENGTHS_UM, c, NOMINAL_ALPHA
        )
        causal, offline = velocity_bounds(prior, cfg)
        tag = 'mich' if c == 0 else 'c02'
        for which, (name, families, bound) in enumerate(
            (('causal', Q2_CAUSAL, causal), ('offline', Q2_OFFLINE, offline))
        ):
            best = min(
                mean_metric(r, Q2_SIMS[sim])
                for stem, _ in families
                if (r := q2_record(records, stem, sim)) is not None and r.converged
            )
            out[f'an_{tag}_{name}_bound'] = sig(bound, 2)
            out[f'an_{tag}_{name}_net'] = sig(best, 2)
            out[f'an_{tag}_{name}_factor'] = f'{best / bound:.0f}'
            out[f'an_{tag}_{name}_snr'] = (
                f'{_equivalent_snr(prior, cfg, best, which):.0f}'
            )

    # Q2.1.1: the C sweep of q3_s0c_ceiling at alpha = 3, and alpha = 0.
    def causal_at(c: float, alpha: float) -> float:
        cfg = SimConfig.nominal(STEMLAB_125_14, SYNTHETIC_WAVELENGTHS_UM, c, alpha)
        return velocity_bounds(prior, cfg)[0]

    c_lo, c_hi = Q3_FEEDBACK_C[0], Q3_FEEDBACK_C[-1]
    base = causal_at(0.0, NOMINAL_ALPHA)
    out |= {
        'an_info_c_lo': sig(2 * bounds.phase_factor(c_lo, NOMINAL_ALPHA), 3),
        'an_info_c_hi': sig(2 * bounds.phase_factor(c_hi, NOMINAL_ALPHA), 3),
        'an_info_c_hi_a0': sig(2 * bounds.phase_factor(c_hi, 0.0), 3),
        'an_c_lo': f'{c_lo:g}',
        'an_bound_gain_c02': sig(base / causal_at(NOMINAL_C, NOMINAL_ALPHA), 3),
        'an_bound_gain_c_hi': sig(base / causal_at(c_hi, NOMINAL_ALPHA), 3),
        'an_bound_gain_c_hi_a0': sig(base / causal_at(c_hi, 0.0), 3),
    }

    # Q2.2: what each representable knob does to the causal bound at C = 0.2.
    cfg = SimConfig.nominal(
        STEMLAB_125_14, SYNTHETIC_WAVELENGTHS_UM, NOMINAL_C, NOMINAL_ALPHA
    )
    out |= {'an_snr_nominal': f'{cfg.snr_db:.0f}'}
    return out


def fiber_values() -> dict[str, Any]:
    """Q3.1 and Q3.2 analytic values, from scripts/fiber_analytics.py's JSON."""
    data = json.loads((RESULTS_DIR / 'fiber_results.json').read_text('utf-8'))
    rows = data['q3_1']['rows']

    def arm(name: str, dispersion: float) -> dict[str, Any]:
        return next(
            r for r in rows if r['arm'] == name and r['dispersion_per_um'] == dispersion
        )

    none0 = arm('fiber_no_monitor', 0.0)
    d_lo, d_hi = arm('fiber_no_monitor', 0.001), arm('fiber_no_monitor', 0.1)
    mon1, mon3 = (
        arm('fiber_dichroic_monitor', 0.0),
        arm('fiber_dichroic_monitor_x3', 0.0),
    )
    pol = {r['n_modes']: r for r in data['q3_2']['rows']}
    big = [r for n, r in pol.items() if n >= 8]
    return {
        'fb_cond_d0': sci(none0['cond']),
        'fb_info_d0': sci(none0['info_rel_free_space']),
        'fb_info_dlo': sci(d_lo['info_rel_free_space']),
        'fb_info_dhi': sci(d_hi['info_rel_free_space']),
        # info ~ D^2: the slope of log info against log D over two decades.
        'fb_disp_slope': sig(math.log10(d_hi['info'] / d_lo['info']) / 2, 3),
        'fb_mon1': sig(mon1['info_rel_free_space'], 2),
        'fb_mon1_pred': sig(mon1['predicted_rel_free_space'], 2),
        'fb_mon3': sig(mon3['info_rel_free_space'], 2),
        'fb_mon3_pred': sig(mon3['predicted_rel_free_space'], 2),
        'fb_mon_um': f'{1000 * data["q3_1"]["monitor_um"]:.0f}',
        'fb_mon3_um': ', '.join(
            f'{1000 * w:.0f}' for w in data['q3_1']['monitors_x3_um']
        ),
        'fb_shots': str(data['q3_1']['n_shots']),
        'fb_corr_n1': sig(pol[1]['corr'], 3),
        'fb_corr_sqrt_lo': sig(min(-r['corr_times_sqrt_n'] for r in big), 2),
        'fb_corr_sqrt_hi': sig(max(-r['corr_times_sqrt_n'] for r in big), 2),
        'fb_r2n_lo': sig(min(r['r2'] * r['n_modes'] for r in big), 2),
        'fb_r2n_hi': sig(max(r['r2'] * r['n_modes'] for r in big), 2),
        'fb_k2_n8': sig(pol[8]['mean_kappa2'], 2),
        'fb_coe_n8': sig(pol[8]['coe_mean_kappa2'], 2),
        'fb_k2_n1': sig(pol[1]['mean_kappa2'], 2),
        'fb_draws': str(data['q3_2']['n_draws']),
        'fb_corr_n8': sig(pol[8]['corr'], 2),
        'fb_r2_n8': sig(pol[8]['r2'], 2),
        **r8_values(data['drift']['shot_s']),
    }


def fiber_sweep_values() -> dict[str, Any]:
    """Q3 degradation: per-knob reference vs TCAN, and the fading hypotheses."""
    mech = json.loads((RESULTS_DIR / 'fiber_sweep_mechanism.json').read_text('utf-8'))
    ref = json.loads((RESULTS_DIR / 'fiber_sweep_analytics.json').read_text('utf-8'))
    pts, sweep = ref['points'], mech['sweep']
    base = 'modes-1'
    knobs = [
        ('M', [1, 2, 4, 8], 'modes'),
        ('speckle', [0.1, 0.3, 1, 3], 'speckle'),
        ('path', [0.1, 0.3, 1, 2.8], 'path'),
    ]
    rows = []
    for knob, values, stem in knobs:
        for v in values:
            name = f'{stem}-{v:g}'
            net = sweep[name]['rmse'] ** 2 - sweep[base]['rmse'] ** 2
            refx = pts[name]['rmse'] ** 2 - pts[base]['rmse'] ** 2
            rows.append(
                {
                    'knob': knob,
                    'value': f'{v:g}',
                    'ref_ratio': f'{pts[name]["mse_ratio_to_free_space"]:.2f}',
                    'ref_excess': f'{refx * 1e3:.1f}',
                    'tcan': f'{sweep[name]["rmse"]:.3f}',
                    'tcan_si': f'{sweep[name]["rmse_sign_invariant"]:.3f}',
                    'tcan_excess': f'{net * 1e3:.1f}',
                }
            )
    hyp = mech['fading_hypotheses']
    shots = mech['per_shot']
    hrows = [
        {
            'm': name.split('-')[1],
            'c_eff': f'{h["feedback_c_eff"]:.3f}',
            'snr_loss': f'{h["snr_loss_db"]:.1f}',
            'observed': f'{h["observed_retention"]:.2f}',
            'h1': f'{h["h1_c_sweep_retention"]:.2f}',
            'h2': f'{h["h2_snr_retention"]:.2f}',
            'h1s': f'{h["h1_sign_evidence_retention"]:.2f}',
            'r_speed': f'{shots[name]["corr_excess_mse_vs_speed"]:.2f}'
            if name in shots
            else '',
            'r_amp': f'{shots[name]["corr_excess_mse_vs_amplitude"]:+.2f}'
            if name in shots
            else '',
            'median': f'{shots[name]["median_rmse_faded"]:.3f}'
            if name in shots
            else f'{shots["modes-8"]["median_rmse_base"]:.3f}',
        }
        for name, h in hyp.items()
    ]
    probe = mech['rescale_probe']
    sel = 'modes-1_model_on_modes-8'
    return {
        'fs_table': rows,
        'fs_hyp_table': hrows,
        'fs_zero': f'{mech["zero_predictor"]:.3f}',
        'fs_nominal': f'{sweep["nominal"]["rmse"]:.3f}',
        'fs_probe_raw': f'{probe[sel]["raw"]:.3f}',
        'fs_probe_rescaled': f'{probe[sel]["rescaled"]:.3f}',
        'fs_color_corr': sig(
            shots['modes-8']['max_abs_corr_between_color_amplitudes'], 1
        ),
        # Guards for the prose: the fading cost exceeds every hypothesis'.
        'fs_fading_unexplained': all(
            h['observed_retention']
            < min(
                h['h1_c_sweep_retention'],
                h['h2_snr_retention'],
                h['h1_sign_evidence_retention'],
            )
            for n, h in hyp.items()
            if n != base
        ),
        'fs_no_sign_flips': all(
            abs(p['rmse'] - p['rmse_sign_invariant']) < 0.005 for p in sweep.values()
        ),
        'fs_speckle_free': max(
            sweep[f'speckle-{r:g}']['rmse'] for r in (0.1, 0.3, 1, 3)
        )
        - sweep[base]['rmse']
        < 0.005,
    }


def diagnosis_values() -> dict[str, Any]:
    """Q3.1 and Q3.2: the reference RMSE and the training-failure checks."""
    data = json.loads((RESULTS_DIR / 'fiber_diagnosis.json').read_text('utf-8'))
    ref = data['reference_rmse']
    full = 'fading=True_speckle=True_path=True'
    no_path = 'fading=True_speckle=True_path=False'
    return {
        'dg_ref_fs': sig(ref['nominal'][full]['rmse'], 2),
        'dg_ref_fiber': sig(ref['fiber'][full]['rmse'], 2),
        'dg_ratio_fiber': sig(ref['fiber'][full]['mse_ratio_to_free_space'], 2),
        'dg_ratio_monitor': sig(
            ref['fiber-monitor'][full]['mse_ratio_to_free_space'], 2
        ),
        'dg_ratio_pbs': sig(ref['fiber-pbs'][full]['mse_ratio_to_free_space'], 2),
        'dg_ratio_nopath': sig(ref['fiber'][no_path]['mse_ratio_to_free_space'], 2),
        'dg_shots': str(data['n_shots']),
    }


def flow_values(records: dict[str, RunRecord]) -> dict[str, Any]:
    """Q3.3 and Q3.4: the R8 flow-matching scores, from scripts/flow.py's JSON.

    The encoder flows (``_enc`` keys) are the scored method.
    """
    data = json.loads((RESULTS_DIR / 'flow_results.json').read_text('utf-8'))
    den = data['q3_3_denoise_enc']
    rmse = {k: v['mean'] for k, v in den['velocity_rmse'].items()}
    check = den['sampling_check']
    sign = data['q3_4_posterior_sign_enc']
    rows = flow_rows(records)
    out: dict[str, Any] = {
        'fl_den_chain': f'{rmse["free_tcan_on_denoised_fiber"]:.3f}',
        'fl_den_fiber': f'{rmse["fiber_tcan_on_fiber"]:.3f}',
        'fl_den_steps_plateau': check['steps_plateau'],
        'fl_den_beats_fiber': rmse['free_tcan_on_denoised_fiber']
        < rmse['fiber_tcan_on_fiber'],
        'fl_shots': str(den['test_set']['n_shots']),
        'fl_sign_fs': pct(
            sign['nominal']['sign_structure'][0]['shot_majority_agreement']
        ),
        'fl_head_gain': sig(rows['tcan'].fs / rows['head'].fs, 2),
        'fl_flow_gain_pct': pct(1 - rows['flow'].fs / rows['head'].fs),
    }
    for name, row in rows.items():
        out[f'fl_{name}_label'] = row.label
        out[f'fl_{name}_field'] = f'{row.field:,}'
        out[f'fl_{name}_params'] = f'{row.params:,}'
        for field in dataclasses.fields(row)[3:]:
            x = getattr(row, field.name)
            out[f'fl_{name}_{field.name}'] = '--' if x is None else f'{x:.3f}'
        for field in ('coverage_fs', 'coverage_fiber'):
            x = getattr(row, field)
            out[f'fl_{name}_{field}'] = '--' if x is None else sig(x, 2)
    return out


ARMS2 = ('nominal', 'fiber')


def r8_values(shot_s: float) -> dict[str, str | bool]:
    """Q3.3 and Q3.4 analytic values: arithmetic on the R8 fiber operating point."""
    fiber = R8_FIBER
    path_shot = fiber.path_drift_um_per_sqrt_s * math.sqrt(shot_s)
    return {
        'r8_modes': str(fiber.n_modes),
        'r8_path_shot_um': sig(path_shot, 2),
        # The text: the drift over one shot exceeds a fringe at every color.
        'r8_path_over_fringe': path_shot > max(SYNTHETIC_WAVELENGTHS_UM) / 2,
        'r8_coe': sig(2 / (2 * fiber.n_modes + 1), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output

    prior = bounds.motion_prior(
        SimConfig.nominal(STEMLAB_125_14, SYNTHETIC_WAVELENGTHS_UM, 0.0, NOMINAL_ALPHA)
    )
    causal_k3, _ = nominal_bounds(prior, SYNTHETIC_WAVELENGTHS_UM)
    real = real_tcn_rmse()
    records = synthetic_records()
    # The alpha = 0 control's test set, whose signed-MSE optimum is zero.
    zero = zero_predictor_rmse(
        records,
        CEILING_GRID,
        cell_key('tcn-44k-T90', SYNTHETIC_WAVELENGTHS_UM, 'smi-a0', 0),
    )

    values = {
        'prior_psd_rms_db': sig(prior.psd_rms_db, 2),
        'phase_factor_c02': sig(bounds.phase_factor(0.2, NOMINAL_ALPHA), 3),
        'phase_factor_c07': sig(bounds.phase_factor(0.7, NOMINAL_ALPHA), 3),
        'real_gap': f'{real / causal_k3:.0f}',
        **real_values(),
        **exit_check_values(records),
        **q3_values(records, zero),
        **noise_values(records, prior),
        **q2_table(records),
        **q2_values(records, zero),
        **q1_s2_values(records, seed_resolution(records)),
        **q1_binned_values(records, prior),
        **q1_s4_values(records, seed_resolution(records)),
        **spectral_values(),
        **prediction_values(records, zero),
        **methods_values(records, shot_length(records)),
        **displacement_values(records),
        **s1d_values(records),
        **sagnac_values(),
        **channel_values(),
        **shift_values(prior),
        **q2_analytic_values(records, prior),
        **fiber_values(),
        **r1_values(records, zero),
        **r2_values(records, zero),
        **r4_values(records, zero),
        **r7_values(records, zero),
        **diagnosis_values(),
        **fiber_sweep_values(),
        **flow_values(records),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(values, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
