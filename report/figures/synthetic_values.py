#!/usr/bin/env python
"""Numbers that synthetic.typ quotes in prose and in its sign-evidence table.

The document reads this JSON at compile time, so a change to
smi.analysis.bounds (or to the results artifacts) rebuilds the quoted values
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
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import (
    CEILING_GRID,
    DATA_DIR,
    DISPLACEMENT_METRICS,
    EXPERIMENTS_DIR,
    LEAVE_ONE_OUT,
    MATCHED_DIR,
    MISMATCH_SHOTS,
    NOISE_KNOBS,
    Q1_BINNED_JSON,
    Q1_FAMILIES,
    Q1_MULTI,
    Q1_PAIRS,
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
    S1D_FAMILIES,
    STALL_TOLERANCE,
    SYNTHETIC_WAVELENGTHS_UM,
    LadderRow,
    c_sim,
    displacement_pending,
    matched_length_cells,
    mean_metric,
    mismatch_spectra,
    noise_ladder,
    noise_seeds,
    oracle_cell,
    q2_record,
    s1d_record,
    synthetic_cell,
    synthetic_records,
    velocity_bounds,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from smi.analysis import bounds
from smi.analysis.experiments import ORACLE_FAMILY, cell_key, expand_grid
from smi.analysis.models.factory import create_model
from smi.analysis.models.sliding_window import SlidingWindowModel
from smi.analysis.results_artifact import RunRecord
from smi.analysis.synthetic_lit_module import VAL_SEED
from smi.synthetic.forward import (
    NOMINAL_ALPHA,
    NOMINAL_PHASE_DRIFT_FRINGES,
    NOMINAL_PHASE_DRIFT_RAD_PER_SQRT_S,
    SimConfig,
    rad_per_sqrt_s_to_fringes,
)
from smi.synthetic.frozen import DEFAULT_ROOT, load_or_create

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


def nominal_bounds(
    prior: bounds.StateSpace, wavelengths_um: tuple[float, ...]
) -> tuple[float, float]:
    """Causal and offline bounds of ``wavelengths_um`` at the nominal point."""
    return velocity_bounds(
        prior, SimConfig.nominal(wavelengths_um, NOMINAL_C, NOMINAL_ALPHA)
    )


def real_tcn_rmse() -> float:
    results = json.loads((DATA_DIR / 'learned_results.json').read_text('utf-8'))
    runs = [
        run['metrics']['velocity_rmse_um_per_ms']['mean']
        for run in results['datasets']['free-space']
        if run['config']['config_name'] == REAL_TCN_CONFIG
    ]
    return sum(runs) / len(runs)


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


def ran_third_cycle(record: RunRecord) -> bool:
    """Whether ``record`` trained past its second warm restart."""
    return epochs(record) > 2 * (record.T_0 or 0)


def test_shots(record: RunRecord) -> int:
    """Shot count of the frozen set, from its dataset key ``<hash>_seed<s>_n<n>``.

    The record stores the set's seed but not its size; the key is written by
    smi.synthetic.frozen.frozen_set_path.
    """
    return int(record.dataset.rsplit('_n', 1)[1])


def zero_predictor_rmse(records: dict[str, RunRecord], grid: Path, key: str) -> float:
    """Test RMSE of predicting zero velocity on cell ``key``'s test set.

    That is each shot's velocity RMS, averaged over shots. The sim config
    comes from expanding the grid that produced the record, hash-checked, so
    the emitter cannot drift from the runner or score another set after a
    grid edit.
    """
    record = records[key]
    (cfg,) = [cell.sim for cell in expand_grid(grid)[1] if cell.key == key]
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
        cell.sim for cell in expand_grid(CEILING_GRID)[1] if cell.key == record.cell_key
    ]
    assert record.test_seed is not None
    path = load_or_create(cfg, record.test_seed, test_shots(record), DEFAULT_ROOT)
    with h5py.File(path, 'r') as f:
        return int(f['velocity'].shape[1])


def alpha_factor(alpha: float) -> float:
    """The alpha dependence of the leading-order sign information."""
    return alpha**2 / (1 + alpha**2)


def exit_check_values(records: dict[str, RunRecord]) -> dict[str, str]:
    """Training budget and test set of section 1.1's three exit-check runs."""
    family = 'tcn-44k-E20'
    signed = synthetic_cell(records, family, 'michelson-signed')
    return {
        'exit_max_epochs': str(signed.max_epochs),
        'exit_steps': str(signed.steps_per_epoch),
        'exit_signed_epochs': str(signed.epochs_run),
        'test_seed': str(signed.test_seed),
        'test_shots': str(test_shots(signed)),
    }


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
    curve_si = sweep(t90, 'velocity_rmse_sign_invariant')
    michelson = rmse(t90, 'michelson', 'velocity_rmse_sign_invariant')
    michelson_n = rmse(t90, 'michelson', 'velocity_nrmse_sign_invariant')
    top = c_sim(Q3_FEEDBACK_C[-1])
    c02 = c_sim(0.2)
    a5_gap = max(
        abs(rmse(f, 'smi-a5') - rmse(f, c02)) / rmse(f, c02) for f in (t90, t30)
    )
    a0 = [synthetic_cell(records, f, 'smi-a0') for f in (t90, t30)]
    t30_epochs = {
        c: synthetic_cell(records, t30, c_sim(c)).epochs_run for c in Q3_FEEDBACK_C
    }
    others = [n for c, n in t30_epochs.items() if c != 0.4 and n is not None]
    long = synthetic_cell(records, 'tcn-44k-T270', c02)

    # Per shot (NRMSE) the low-C cells do not beat Michelson; the text names
    # the largest such C per budget, which only reads as a range if the worse
    # cells are the lowest ones.
    def nrmse_worse(family: str) -> str:
        ref = rmse(family, 'michelson', 'velocity_nrmse_sign_invariant')
        worse = [n > ref for n in sweep(family, 'velocity_nrmse')]
        n_worse = worse.index(False)
        assert n_worse > 0, (family, worse)
        assert not any(worse[n_worse:]), (family, worse)
        return f'{Q3_FEEDBACK_C[n_worse - 1]:g}'

    curve_n = sweep(t90, 'velocity_nrmse')
    curve_n_si = sweep(t90, 'velocity_nrmse_sign_invariant')
    n_gaps = [s / i - 1 for s, i in zip(curve_n, curve_n_si, strict=True)]
    worst = [
        synthetic_cell(records, t90, c_sim(c)).metrics['velocity_nrmse']['max']
        for c in Q3_FEEDBACK_C
    ]
    # A sign-flipped zero-mean shot scores NRMSE 2. The text says the worst
    # shot reaches it from some C upward, so that C must start a suffix.
    flipped = [w >= 2 for w in worst]
    n_flip = flipped.index(True)
    assert all(flipped[n_flip:]), worst

    def median_metric(sim: str, metric: str) -> float:
        return synthetic_cell(records, t90, sim).metrics[metric]['median']

    def slowest(family: str) -> list[float]:
        """Slowest-|v|-decile sign-invariant RMSE of Michelson and each C."""
        sims = ('michelson', *(c_sim(c) for c in Q3_FEEDBACK_C))
        bins = [synthetic_cell(records, family, sim).binned for sim in sims]
        return [b['abs_velocity'][0]['rmse_sign_invariant'] for b in bins if b]

    slow = slowest(t90)
    t90_epochs = [epochs(synthetic_cell(records, t90, c_sim(c))) for c in Q3_FEEDBACK_C]
    zero_gap = abs(mean_metric(a0[0], 'velocity_rmse') / zero - 1)
    return {
        # The text claims the 90-epoch curve falls monotonically in RMSE; the
        # document asserts this, so a regenerated artifact that breaks it
        # fails the build.
        'q3_monotone': all(b < a for a, b in itertools.pairwise(curve)),
        'q3_c_top': f'{Q3_FEEDBACK_C[-1]:g}',
        'q3_gain_top': sig(michelson / curve[-1], 2),
        'q3_gain_top_nrmse': sig(michelson_n / rmse(t90, top, 'velocity_nrmse'), 2),
        'q3_gain_top_nrmse_median': sig(
            median_metric('michelson', 'velocity_nrmse_sign_invariant')
            / median_metric(top, 'velocity_nrmse'),
            2,
        ),
        'q3_sign_gap_pct': pct(
            max((s - i) / i for s, i in zip(curve, curve_si, strict=True))
        ),
        'q3_nrmse_worse_t90': nrmse_worse(t90),
        'q3_nrmse_worse_t30': nrmse_worse(t30),
        'q3_n_sign_gap_max_pct': pct(max(n_gaps)),
        'q3_n_sign_gap_max_c': f'{Q3_FEEDBACK_C[n_gaps.index(max(n_gaps))]:g}',
        **lo_hi('q3_n_worst', worst, lambda x: sig(x, 3)),
        'q3_flip_c': f'{Q3_FEEDBACK_C[n_flip]:g}',
        **lo_hi('q3_slow', slow, lambda x: sig(x, 2)),
        # The text calls the slow-decile floor a budget property; the
        # document asserts that it is lower at 90 epochs in every cell.
        'q3_slow_falls_with_budget': len(slow) == len(Q3_FEEDBACK_C) + 1
        and all(a < b for a, b in zip(slow, slowest(t30), strict=True)),
        **lo_hi('q3_t90_epochs', t90_epochs, str),
        'q3_zero_gap_pct': sig(100 * zero_gap, 1),
        'q3_a5_gap_pct': pct(a5_gap),
        'q3_a0_epochs': str(max(epochs(r) for r in a0)),
        # eq-sign-info depends on alpha only through alpha^2 / (1 + alpha^2).
        'sign_info_a5_a3': sig(alpha_factor(5.0) / alpha_factor(3.0), 3),
        # The 30-epoch curve's one inversion, which the text attributes to the
        # C = 0.4 cell's longer run; asserted by the document like q3_monotone.
        'q3_t30_c04_below': all(
            t30_curve[Q3_FEEDBACK_C.index(0.4)] < t30_curve[Q3_FEEDBACK_C.index(c)]
            for c in (0.7, 0.9)
        ),
        'q3_t30_c04_epochs': str(t30_epochs[0.4]),
        **lo_hi('q3_t30_other_epochs', others, str),
        'q3_c02_t270': sig(rmse('tcn-44k-T270', c02), 2),
        'q3_t270_epochs': str(long.epochs_run),
        'q3_t270_cycle': str(long.T_0),
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
    (c09,) = noise_seeds(records, 'nominal-c0.9')
    (nominal0,) = [r for r in nominal_runs if r.seed == 0]
    # The causal bound at the 30 dB row's own config (unit visibility, no
    # drift), so it is the floor for exactly what that row adds to the base.
    snr_row = at_nominal(snr)
    snr_bound = float(bounds.pcrb_filter(prior, snr_row.sim, N_STEPS)[-1]) ** 0.5
    snr_cost = rmse(snr_row.record) - base_mean
    (snr20_cost,) = [
        rmse(row.record) - base_mean for row in ladder[snr] if row.value == 20
    ]
    # The DC-offset rows that score below every base seed; the text names them
    # as the +/-0.3 and +/-1 rows and reports how long they trained.
    dc_low = [row for row in ladder['dc_offset_range'] if rmse(row.record) < min(base)]
    rows = [row for rows in ladder.values() for row in rows]
    # Rows that trained into a third cosine cycle, unlike every base seed.
    long_rows = [row for row in rows if ran_third_cycle(row.record)]
    assert not any(ran_third_cycle(r) for r in base_runs)
    short_rows = [row for row in rows if row not in long_rows]

    def above_base(field: str) -> list[object]:
        return [row.value for row in ladder[field] if rmse(row.record) > max(base)]

    # Leave-one-out: the nominal point with one knob returned to noise-free,
    # read as the drop below the nominal seeds' mean.
    nominal_mean = sum(nominal) / len(nominal)
    width = max(nominal) - min(nominal)
    loo: dict[str, RunRecord] = {}
    for field, stem in LEAVE_ONE_OUT:
        (loo[field],) = noise_seeds(records, f'nominal-no-{stem}')

    def cut(field: str) -> float:
        return 1 - rmse(loo[field]) / nominal_mean

    def outside(field: str) -> bool:
        # Clearly outside: below the lowest nominal seed by more than the
        # nominal seeds' own spread.
        return rmse(loo[field]) < min(nominal) - width

    loo_long = {f for f, r in loo.items() if ran_third_cycle(r)}
    pdbw = 'pd_bandwidth_hz'
    # The equal-length grid: every cell trained the same length, its cap.
    e60_cells, e60_sweep = matched_length_cells(records)
    (e60_epochs,) = {r.epochs_run for r in e60_cells.values()}
    e60 = {name: rmse(r) for name, r in e60_cells.items()}
    e60_nominal = e60['nominal']
    e60_nodrift = e60['nominal-no-phasedrift']
    e60_base = e60['ladder-base']
    # The ladder's drift sweep rerun at 60 epochs on the same knob values,
    # in sweep order.
    phase_rows = [row.record for row in ladder[phase]]
    e60_drift_rmse = [rmse(e60_sweep[phase, row.value]) for row in ladder[phase]]
    e60_drift_nominal = rmse(e60_sweep[phase, at_nominal(phase).value])
    e60_half_kept = (e60_drift_rmse[0] - e60_base) / (e60_drift_nominal - e60_base)
    # Fractional gain from 35 to 60 epochs, against the ladder's seeds.
    nominal_gain = 1 - e60_nominal / nominal_mean
    base_gain = 1 - e60_base / base_mean
    loo_rows = sorted(loo.items(), key=lambda item: rmse(item[1]))
    return {
        'loo_rows': [
            {
                'label': NOISE_KNOBS[f],
                'rmse': f'{rmse(r):.3f}',
                'epochs': str(r.epochs_run),
            }
            for f, r in loo_rows
        ],
        **lo_hi('loo_nominal', nominal, lambda x: f'{x:.3f}'),
        **lo_hi('loo_nominal_epochs', map(epochs, nominal_runs), str),
        'loo_phase_cut': pct(cut(phase)),
        'loo_vis_cut': pct(cut('visibility_range')),
        'loo_pdbw_cut': pct(cut(pdbw)),
        # The text: phase drift is the largest removal, then visibility and
        # PD bandwidth are the only others clearly outside the nominal spread.
        'loo_outside_named': {f for f in loo if outside(f)}
        == {phase, 'visibility_range', pdbw}
        and loo_rows[0][0] == phase,
        # The text: only the phase-drift and PD-bandwidth cells trained into
        # a third cosine cycle.
        'loo_long_named': loo_long == {phase, pdbw},
        'loo_e60_epochs': str(e60_epochs),
        'loo_e60_cut': pct(1 - e60_nodrift / e60_nominal),
        'loo_e60_base_gain': pct(base_gain),
        'loo_e60_nominal_gain': pct(nominal_gain),
        # The text: at equal length the cut exceeds the nominal seed spread,
        # and the ranking base < no-phase-drift < nominal is unchanged.
        'loo_e60_outside': e60_nodrift < e60_nominal - width,
        'loo_e60_same_ranking': e60_base < e60_nodrift < e60_nominal,
        'loo_snr_saving': sig(nominal_mean - rmse(loo[snr]), 1),
        # The text: removing SNR saves under a third of what it costs alone.
        'loo_snr_subadditive': nominal_mean - rmse(loo[snr]) < snr_cost / 3,
        # The text: only 8 bits and a 3 kHz bandwidth leave the base spread.
        'noise_electronics_flat': above_base('adc_bits') == [8]
        and above_base('pd_bandwidth_hz') == [3e3],
        # The text: removing any one electronic knob moves the error at most
        # as much as removing the bandwidth limit.
        'loo_electronics_max_pdbw': max(cut(f) for f in (snr, 'adc_bits', pdbw))
        == cut(pdbw),
        # The text: at 20% each drift costs comparably to 30 dB white noise.
        'noise_drift_20_like_snr': all(
            2 / 3 < (rmse(row.record) - base_mean) / snr_cost < 3 / 2
            for f in ('amplitude_drift_rel', 'contrast_drift_rel')
            for row in ladder[f]
            if row.value == 0.2
        ),
        'noise_n_seeds': str(len(base)),
        'noise_phase_fringes': sig(NOMINAL_PHASE_DRIFT_FRINGES, 3),
        'noise_phase_rad': sig(NOMINAL_PHASE_DRIFT_RAD_PER_SQRT_S, 3),
        'noise_phase_share': pct(share(phase)),
        'noise_snr_share': pct(share(snr)),
        'noise_minor_share_hi': pct(max(minor_shares)),
        'noise_spread_share': pct((max(base) - min(base)) / excess),
        # The ranking the text states: phase drift, then SNR, then the rest.
        'noise_ranking': share(phase) > share(snr) > max(minor_shares),
        **lo_hi('noise_phase_epochs', map(epochs, phase_rows), str),
        # The text: at equal length the drift curve rises monotonically, and
        # cutting drift to its lowest swept level keeps most of its cost.
        'noise_e60_phase_monotone': e60_base < e60_drift_rmse[0]
        and all(a < b for a, b in itertools.pairwise(e60_drift_rmse)),
        'noise_e60_low_fringes': sig(
            rad_per_sqrt_s_to_fringes(ladder[phase][0].sim.phase_drift_rad_per_sqrt_s),
            1,
        ),
        'noise_e60_half_kept': pct(e60_half_kept),
        'noise_e60_half_keeps_most': e60_half_kept > 0.5,
        'noise_c09_ratio': sig(rmse(nominal0) / rmse(c09), 2),
        'noise_snr_cost': sig(snr_cost, 2),
        'noise_snr_bound': sig(snr_bound, 2),
        # The text: the base alone sits this far above the bound, so the
        # prediction's floor cannot bind at this budget.
        'noise_base_over_bound': f'{base_mean / snr_bound:.0f}',
        'noise_bound_no_power': base_mean > 5 * snr_bound,
        # The text: at 20 dB the white-noise cost about doubles.
        'noise_snr20_cost': sig(snr20_cost, 2),
        'noise_snr20_doubles': 1.7 < snr20_cost / snr_cost < 2.3,
        'noise_dc_low_named': {row.value[1] for row in dc_low} == {0.3, 1.0},
        'noise_dc_low_epochs': '/'.join(
            sorted({str(row.record.epochs_run) for row in dc_low})
        ),
        **lo_hi('noise_base_epochs', map(epochs, base_runs), str),
        'noise_n_long': str(len(long_rows)),
        'noise_n_rows': str(len(rows)),
        **lo_hi('noise_long_epochs', (epochs(r.record) for r in long_rows), str),
        **lo_hi('noise_short_epochs', (epochs(r.record) for r in short_rows), str),
    }


# A converged cell stopped well before its first restart; the rest end at it.
EARLY_STOP_FRACTION = 0.95
# The text says no family comes near the Michelson oracle; a ratio this large
# is well clear of the seed range, which a bare `oracle < best` would not be.
ORACLE_MARGIN = 1.5
# The text calls the top offline families' span close to the seed range.
CLOSE_FACTOR = 1.5


def q2_table(records: dict[str, RunRecord]) -> dict[str, object]:
    """The rows of section Q2's table, and checks on what they are labeled.

    Context is each family's ``receptive_field``, read off its model built
    from the matched config it trained under (tests/test_receptive_field.py
    pins each against a gradient probe).
    """
    models = {
        stem: create_model(
            yaml.safe_load((MATCHED_DIR / f'{stem}.yaml').read_text('utf-8'))['model']
        )
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

    oracle = [
        mean_metric(oracle_cell(records, sim), metric)
        for sim, metric in Q2_SIMS.items()
    ]
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
    # The text quotes the one-window field the MLP and Barland records store,
    # written before the models declared the averaged span; only the record
    # still carries it, so this one value is read from there.
    mlp = synthetic_cell(records, 'mlp-44k-T90', 'smi-a3')
    # The families whose context is a fixed window, which the text says are
    # matched; the sliding-window families' wider spans are named apart.
    matched = {
        stem: field
        for stem, field in fields.items()
        if stem not in sliding and field is not None
    }
    return {
        'q2_causal_rows': rows(Q2_CAUSAL, True),
        'q2_offline_rows': rows(Q2_OFFLINE, False),
        'q2_oracle_cells': [f'{v:.3f}' for v in oracle],
        'q2_mamba_pending': q2_record(records, 'mamba-44k', 'smi-a3') is None,
        # The text says Mamba, unlike the LSTMs, trained on both sims.
        'q2_mamba_trained': all(
            r is not None and r.converged
            for r in (q2_record(records, 'mamba-44k', sim) for sim in Q2_SIMS)
        ),
        'q2_stalled_named': set(stalled)
        == {
            ('lstm-44k', 'michelson'),
            ('lstm-44k', 'smi-a3'),
            ('lstm-bidir-44k', 'michelson'),
            ('lstm-bidir-44k', 'smi-a3'),
            ('neuraloperator-offline-44k', 'smi-a3'),
        },
        'q2_stall_tol_pct': pct(STALL_TOLERANCE),
        'q2_stall_epochs': '/'.join(
            sorted(
                {
                    str(synthetic_cell(records, f'{s}-T90', m).epochs_run)
                    for s, m in stalled
                }
            )
        ),
        'q2_sliding_named': set(sliding) == {'barland-44k', 'mlp-44k'},
        'q2_mlp_span': str(sliding['mlp-44k']),
        'q2_tcn_field': str(fields['tcn-44k']),
        'q2_barland_span': str(sliding['barland-44k']),
        'q2_window_field': str(mlp.receptive_field),
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
        **lo_hi('q2_matched', matched.values(), str),
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
    mich_named, smi_named = trained('michelson', Q2_NAMED), trained('smi-a3', Q2_NAMED)
    oracle_mich, oracle_smi = (oracle_cell(records, sim) for sim in Q2_SIMS)
    oracle = mean_metric(oracle_mich, 'velocity_rmse_sign_invariant')
    oracle_si = mean_metric(oracle_smi, 'velocity_rmse_sign_invariant')
    tcn_mich, tcn_smi = mich['tcn-44k'], smi['tcn-44k']

    weakest = max(mich_named, key=mich_named.__getitem__)
    rest = [v for s, v in mich_named.items() if s != weakest]
    best = min(smi, key=smi.__getitem__)
    runner_up = min(v for s, v in smi.items() if s != best)
    # The offline families the text calls unsettled: all trained but the MLP.
    offline_top = sorted(
        v for s, v in smi_named.items() if s in dict(Q2_OFFLINE) and s != 'mlp-44k'
    )
    below_tcn = {s for s, v in smi_named.items() if apart(v, tcn_smi)}
    above_tcn = {s for s, v in smi_named.items() if apart(tcn_smi, v)}
    runs = [
        (epochs(r), r.T_0 or 0)
        for s in Q2_NAMED
        for sim in Q2_SIMS
        if (r := q2_record(records, s, sim)) is not None and r.converged
    ]
    early = [n for n, t in runs if n < EARLY_STOP_FRACTION * t]
    late = [n for n, t in runs if n >= EARLY_STOP_FRACTION * t]
    second = [n for n, t in runs if n > t]
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
        **lo_hi('q2_mich', rest, lambda x: sig(x, 2)),
        'q2_mich_span_pct': pct(max(rest) / min(rest) - 1),
        'q2_no_offline_mich': sig(mich_named[weakest], 2),
        'q2_oracle_mich': sig(oracle, 2),
        'q2_oracle_gap': sig(min(mich.values()) / oracle, 2),
        'q2_oracle_beats_all': ORACLE_MARGIN * oracle <= min(mich.values()),
        # C = 0.2: the families the text names as beating or trailing the TCN.
        'q2_smi_vs_tcn': below_tcn
        == {
            'tcan-44k',
            'patchtransformer-causal-44k',
            'scnn-44k',
            'barland-44k',
            'patchtransformer-offline-44k',
            'unettransformer-offline-44k',
        }
        and above_tcn
        == {'unettransformer-causal-44k', 'neuraloperator-causal-44k', 'mlp-44k'},
        'q2_tcan_gain': sig(tcn_smi / smi['tcan-44k'], 3),
        'q2_patch_gain': sig(tcn_smi / smi['patchtransformer-causal-44k'], 3),
        'q2_tcan_best': best == 'tcan-44k' and apart(smi[best], runner_up),
        'q2_tcan_margin_pct': pct(runner_up / smi[best] - 1),
        'q2_offline_span_pct': pct(offline_top[-1] / offline_top[0] - 1),
        'q2_offline_span_close': offline_top[-1] / offline_top[0] - 1
        < CLOSE_FACTOR * resolution,
        'q2_scnn_gain_pct': pct(tcn_smi / smi['scnn-44k'] - 1),
        # Barland's 498-sample span ties SCNN's 85: the wider context shows no gain.
        'q2_barland_ties_scnn': not apart(
            min(smi['barland-44k'], smi['scnn-44k']),
            max(smi['barland-44k'], smi['scnn-44k']),
        ),
        'q2_barland_smi': sig(smi['barland-44k'], 3),
        'q2_scnn_smi': sig(smi['scnn-44k'], 3),
        # SCNN is the TCN with symmetric padding: the one matched pair.
        'q2_scnn_pair': not apart(
            min(tcn_mich, mich['scnn-44k']), max(tcn_mich, mich['scnn-44k'])
        )
        and apart(smi['scnn-44k'], tcn_smi),
        'q2_oracle_smi_si': sig(oracle_si, 2),
        'q2_oracle_smi_signed': sig(oracle_signed, 2),
        'q2_zero_smi': sig(zero, 2),
        'q2_oracle_worse_than_zero': oracle_signed > zero,
        'q2_learned_beat_oracle_smi': max(smi.values()) < oracle_si,
        'q2_oracle_smi_gap': sig(oracle_si / max(smi_named.values()), 2),
        'q2_early_n': str(len(early)),
        **lo_hi('q2_early', early, str),
        **lo_hi('q2_late', late, str),
        'q2_restart': str(max(t for _, t in runs)),
        'q2_second_n': str(len(second)),
        'q2_second_lo': str(min(second)),
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

    lattice_nm = max(math.gcd(round(1e4 * a), round(1e4 * b)) for a, b in Q1_PAIRS) / 10
    tcn_multi = [smi['tcn-44k'][w] for w in Q1_MULTI]
    tcan_pair = min(smi['tcan-44k'][w] for w in Q1_PAIRS)
    tcan_triple = smi['tcan-44k'][triple]
    table_columns = (
        mich['tcn-44k'],
        mich['tcan-44k'],
        oracle,
        smi['tcn-44k'],
        smi['tcan-44k'],
    )
    return {
        'q1_s2_rows': [
            {'label': name, 'cells': [f'{col[w]:.3f}' for col in table_columns]}
            for w, name in Q1_SUBSETS
        ],
        'q1_zero': sig(zero, 2),
        # Michelson: one channel scores at the zero predictor, sign-invariant,
        # in both families; every pair and the triple sit far below it.
        'q1_mich_single_at_zero': all(
            abs(x / zero - 1) < AT_ZERO_TOL for x in mich_single
        ),
        'q1_mich_single_gap_pct': sig(
            100 * max(abs(x / zero - 1) for x in mich_single), 1
        ),
        'q1_mich_multi_far': max(mich_multi) < FAR_FROM_ZERO * zero,
        'q1_oracle_single_above_zero': min(oracle[w] for w in Q1_SINGLES) > zero,
        # The stall test passes the single channels, so the text calls them
        # trained to the zero predictor, not stalled.
        'q1_mich_single_converged': all(r.converged for r in single_records),
        **lo_hi('q1_mich_single_monitor', monitor_gap, lambda x: f'{100 * x:.1f}'),
        'q1_mich_single_epochs': '/'.join(
            sorted({str(epochs(r)) for r in single_records})
        ),
        'q1_stall_tol_pct': pct(STALL_TOLERANCE),
        # The text: 515 nm is the best single channel in both families.
        'q1_515_best': all(ranked_singles(f)[0] == (0.515,) for f in Q1_FAMILIES),
        # Spacing of the lattice 2 pi (lambda_1 Z + lambda_2 Z) that wraps the
        # two-channel phase combination: the coarsest gcd over the pairs.
        'q1_lattice_nm': f'{lattice_nm:g}',
        'q1_runner_up_tcn_pct': pct(runner_up('tcn-44k')),
        'q1_runner_up_tcn_at_edge': at_edge(runner_up('tcn-44k'), resolution),
        'q1_runner_up_tcan_pct': pct(runner_up('tcan-44k')),
        'q1_gain_tcn': sig(gain('tcn-44k'), 3),
        'q1_gain_tcan': sig(gain('tcan-44k'), 3),
        # Every multi-channel set beats the best single channel by more than
        # the seed range, in both families.
        'q1_multi_beats_single': all(
            separated(smi[f][w], smi[f][lowest(smi[f], Q1_SINGLES)], resolution)
            for f in Q1_FAMILIES
            for w in Q1_MULTI
        ),
        'q1_tcn_multi_tied': not separated(min(tcn_multi), max(tcn_multi), resolution),
        'q1_tcan_triple_pct': pct(1 - tcan_triple / tcan_pair),
        'q1_tcan_triple_at_edge': at_edge(tcan_pair / tcan_triple - 1, resolution),
        **lo_hi(
            'q1_epochs',
            (
                epochs(records[cell_key(f, w, sim, 0)])
                for f in Q1_FAMILIES
                for w, _ in Q1_SUBSETS
                for sim in Q1_S2_SIMS
            ),
            str,
        ),
    }


def q1_binned_values(
    records: dict[str, RunRecord], prior: bounds.StateSpace
) -> dict[str, object]:
    """Values section Q1 quotes from the binned errors (smi.analysis.q1_binned).

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
    one, pair, three = (0.515,), (0.635, 0.515), SYNTHETIC_WAVELENGTHS_UM

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

    def all_share(w: tuple) -> str:
        """Share of samples in the nearest bin of every channel's extremum."""
        n = cells['noise_free', w]['fringe_distance']['all']['n']
        return sig(100 * n[0] / sum(n), 1)

    v_gain = [
        a / b
        for a, b in zip(
            cells['noise_free', one]['abs_velocity']['rmse'],
            cells['noise_free', three]['abs_velocity']['rmse'],
            strict=True,
        )
    ]
    v_edges = binned['abs_velocity_edges_um_per_ms']
    slow_one, slow_three, slow_zero = (
        cells['noise_free', w]['abs_velocity'][key][0]
        for w, key in ((one, 'rmse'), (three, 'rmse'), (one, 'zero'))
    )
    peak = v_gain.index(max(v_gain))
    bound_gap = min(
        min(cells['nominal_30db', w]['abs_velocity']['rmse'])
        / nominal_bounds(prior, w)[0]
        for w in (one, pair, three)
    )
    return {
        'q1_bin_width': f'{binned["fringe_distance_edges_fringes"][1]:.3g}',
        'q1_bin_nf_near': sig(nf_one[0], 3),
        'q1_bin_nf_far': sig(nf_one[-1], 3),
        # The text: noise-free, neither curve rises toward the extremum.
        'q1_bin_nf_flat': flat('noise_free', one)
        and flat('noise_free', three)
        and rise('noise_free', one) <= 0,
        'q1_bin_nom_rise_one_pct': pct(rise('nominal_30db', one)),
        'q1_bin_nom_rise_three_pct': pct(rise('nominal_30db', three)),
        # The text: at 30 dB both curves rise toward the extremum only a
        # little, within FLAT_TOL.
        'q1_bin_nom_small_rise': flat('nominal_30db', one)
        and flat('nominal_30db', three),
        'q1_bin_gain_near': sig(gain_near, 2),
        'q1_bin_gain_far': sig(gain_far, 2),
        'q1_bin_gain_not_near': gain_near < gain_far,
        **lo_hi(
            'q1_bin_all_rise',
            (rise('noise_free', w, 'all') for w in (pair, three)),
            pct,
        ),
        'q1_bin_all_share_pair': all_share(pair),
        'q1_bin_all_share_three': all_share(three),
        'q1_bin_v_gain_max': sig(max(v_gain), 2),
        'q1_bin_v_peak_lo': sig(v_edges[peak], 2),
        'q1_bin_v_peak_hi': sig(v_edges[peak + 1], 2),
        'q1_bin_v_slow_hi': sig(v_edges[1], 1),
        'q1_bin_v_slow_gain': sig(v_gain[0], 2),
        'q1_bin_v_slow_one': sig(slow_one, 2),
        'q1_bin_v_slow_three': sig(slow_three, 2),
        'q1_bin_v_slow_zero': sig(slow_zero, 2),
        # The text: in the slowest bin both models score above zero.
        'q1_bin_v_slow_above_zero': min(slow_one, slow_three) > slow_zero,
        # The text: no gain in the slowest bin, the largest above 1.5x.
        'q1_bin_v_shape': v_gain[0] < 1 and max(v_gain) > 1.5,
        'q1_bound_gap': f'{bound_gap:.0f}',
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
    # The one cell the text names as trained longer.
    long_single = s4((0.515,), 40.0)
    label = dict(Q1_SUBSETS)
    return {
        # The text quotes each subset's RMSE against zero at the ends of the
        # sweep.
        **lo_hi('q1_s4_0db', rmse[0.0].values(), lambda x: f'{x:.3f}'),
        **lo_hi(
            'q1_s4_high',
            (x for snr in (30.0, 40.0) for x in rmse[snr].values()),
            lambda x: f'{x:.3f}',
        ),
        'q1_s4_gain_max_pct': pct(max(gain.values())),
        'q1_s4_30_pair_no515': f'{at30[0.635, 0.6748]:.3f}',
        'q1_s4_30_gain_pct': pct(gain[30.0]),
        # The text: at 30 dB the gain is several times the nominal seed range,
        # and the pair without 515 nm does no better than 515 nm alone.
        'q1_s4_30_outside': gain[30.0]
        > 3 * (max(nominal) - min(nominal)) / min(nominal)
        and not separated(at30[0.515,], at30[0.635, 0.6748], resolution),
        **lo_hi('q1_nominal', nominal, lambda x: f'{x:.3f}'),
        'q1_s4_rows': [
            {
                'snr': f'{snr:g}',
                'single': f'{min(row[w] for w in Q1_SINGLES):.3f}',
                'single_label': label[lowest(row, Q1_SINGLES)],
                'pair': f'{min(row[w] for w in Q1_PAIRS):.3f}',
                'pair_label': label[lowest(row, Q1_PAIRS)],
                'three': f'{row[triple]:.3f}',
                'gain': pct(gain[snr]),
            }
            for snr, row in rmse.items()
        ],
        # The text: the gain peaks between 10 and 30 dB, is within the seed
        # range at 0 dB and reverses at 40 dB, where the 515 nm cell alone
        # trained into a third cycle.
        'q1_s4_gain_shape': max(gain, key=gain.__getitem__) in (10.0, 20.0, 30.0)
        and abs(gain[0.0]) < resolution
        and gain[40.0] < 0,
        'q1_s4_40_epochs': str(epochs(long_single)),
        'q1_s4_40_third_cycle': ran_third_cycle(long_single)
        and not any(ran_third_cycle(s4(w, 40.0)) for w in Q1_MULTI),
        **lo_hi('q1_s4_oracle', oracle, lambda x: sig(x, 2)),
        'q1_s4_oracle_worse': min(oracle) > zero,
    }


def spectral_values() -> dict[str, str]:
    """Numbers quoted beside the data/target mismatch figure.

    The PSDs share one bin grid, so sums over bins are power shares.
    """
    spectra = mismatch_spectra(False)
    low = spectra.freq_hz < 100

    def low_share(psd: np.ndarray) -> str:
        return pct(psd[low].sum() / psd.sum())

    return {
        'spec_shots': str(MISMATCH_SHOTS),
        'spec_bin_hz': f'{spectra.freq_hz[0]:.0f}',
        'spec_v_low_pct': low_share(spectra.velocity),
        'spec_x_low_pct': low_share(spectra.displacement),
    }


def budget_range(
    cells: list[RunRecord], per_epoch: int, shot_length: int
) -> dict[str, str]:
    """Epochs, shots and samples the ``cells`` trained on, as printed ranges."""
    n = [epochs(r) for r in cells]
    samples = [
        epochs(r) * per_epoch * (r.train_crop_length or shot_length) for r in cells
    ]
    return {
        **lo_hi('epochs', n, str),
        **lo_hi('kshots', (x * per_epoch / 1e3 for x in n), lambda x: f'{x:.0f}'),
        **lo_hi('gsamples', samples, lambda x: f'{x / 1e9:.2f}'),
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
    grids = [
        yaml.safe_load(p.read_text('utf-8')) for p in EXPERIMENTS_DIR.glob('*.yaml')
    ]
    family = yaml.safe_load((MATCHED_DIR / 'tcn-44k.yaml').read_text('utf-8'))
    convergence = family['training']['convergence']

    def one(values: Iterable[Any]) -> Any:
        (value,) = set(values)
        return value

    batch = one(r.batch_size for r in trained)
    steps = one(r.steps_per_epoch for r in trained)
    crop = one(r.train_crop_length for r in trained if r.train_crop_length is not None)
    per_epoch = batch * steps
    val_steps = one(g['synthetic']['val_steps'] for g in grids)
    test_set = one(tuple(sorted(g['test_set'].items())) for g in grids)
    full_length = sorted({r.model_type for r in trained if r.train_crop_length is None})
    # Each matched family's stopping rule, which the text states once.
    for path in MATCHED_DIR.glob('*-44k.yaml'):
        rule = yaml.safe_load(path.read_text('utf-8'))['training']['convergence']
        assert {k: rule[k] for k in convergence if k != 'monitor'} == {
            k: v for k, v in convergence.items() if k != 'monitor'
        }, path
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
                [r for r in trained if cycle == r.T_0], per_epoch, shot_length
            ).items()
        },
        'm_val_batches': str(val_steps),
        'm_val_shots': str(val_steps * batch),
        'm_val_seed': str(VAL_SEED),
        'm_test_shots': str(dict(test_set)['n_shots']),
        'm_test_seed': str(dict(test_set)['seed']),
        'm_test_sets': str(len({r.dataset for r in records.values()})),
        'm_trained_cells': str(len(trained)),
        'm_patience': str(convergence['patience_epochs']),
        'm_min_rel_delta_pct': pct(convergence['min_rel_delta']),
        'm_grace': str(convergence['min_epochs']),
        'm_lr': f'{one(r.lr for r in trained):g}',
        'm_device': one(r.device.split(':')[0] for r in trained).upper(),
        'm_precision': one(r.precision for r in trained).removesuffix('-true'),
        **lo_hi('m_epochs', map(epochs, trained), str),
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
    si = DISPLACEMENT_METRICS['velocity_rmse_sign_invariant']
    signed = DISPLACEMENT_METRICS['velocity_rmse']
    top = max(Q3_FEEDBACK_C)
    base = [mean_metric(r, signed) for r in noise_seeds(records, 'ladder-base')]
    resolution = (max(base) - min(base)) / (sum(base) / len(base))

    def trained(sim: str) -> dict[str, float]:
        return {
            s: mean_metric(r, DISPLACEMENT_METRICS[Q2_SIMS[sim]])
            for s in Q2_STEMS
            if (r := q2_record(records, s, sim)) is not None and r.converged
        }

    smi, mich = trained('smi-a3'), trained('michelson')
    tcn = smi['tcn-44k']
    ranked = sorted(smi, key=smi.__getitem__)
    lowest = [smi[s] for s in ranked[: len(DISPLACEMENT_LOWEST)]]
    # Michelson: the families the velocity text calls tied (all but the
    # offline neural operator) and their spread on displacement.
    tied = [v for s, v in mich.items() if s != 'neuraloperator-offline-44k']
    curve_rmse = [mean_metric(curve[c], signed) for c in sorted(Q3_FEEDBACK_C)]
    velocity = {
        s: mean_metric(synthetic_cell(records, f'{s}-T90', 'smi-a3'), 'velocity_rmse')
        for s in ('tcn-44k', 'mamba-44k')
    }
    return {
        'disp_pending': False,
        'disp_c_top': f'{top:g}',
        'disp_gain_top': sig(
            mean_metric(michelson, si) / mean_metric(curve[top], signed), 2
        ),
        'disp_curve_monotone': all(a > b for a, b in itertools.pairwise(curve_rmse)),
        'disp_seed_range_pct': pct(resolution),
        **lo_hi('disp_lowest', lowest, lambda x: sig(x, 2)),
        'disp_tcn': sig(tcn, 2),
        'disp_mamba': sig(smi['mamba-44k'], 2),
        # The three lowest are the named ones, unordered by the seed range,
        # and each sits below the TCN by more than it.
        'disp_lowest_named': set(ranked[: len(DISPLACEMENT_LOWEST)])
        == DISPLACEMENT_LOWEST
        and not separated(lowest[0], lowest[-1], resolution)
        and separated(lowest[-1], tcn, resolution),
        # Mamba, tied with the TCN on velocity, is last of the trained
        # families on displacement but not resolved from the TCN.
        'disp_mamba_last': ranked[-1] == 'mamba-44k'
        and not separated(tcn, smi['mamba-44k'], resolution)
        and not separated(
            velocity['tcn-44k'], velocity['mamba-44k'], seed_resolution(records)
        ),
        'disp_mich_span': sig(max(tied) / min(tied), 2),
        'disp_mich_spread': separated(min(tied), max(tied), resolution),
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
        **lo_hi('s1d_smi_ratio', map(ratio, smi), three),
        **lo_hi('s1d_mich_ratio', map(ratio, mich.values()), lambda x: sig(x, 2)),
        **lo_hi('s1d_epochs', map(epochs, [*smi, *mich.values()]), str),
        # The text: at C = 0.2 every retargeted cell, the TCN control
        # included, stalls; on Michelson every one trains.
        's1d_smi_stalled': all(
            not r.converged and ratio(r) >= 1 - STALL_TOLERANCE for r in smi
        ),
        's1d_mich_trained': all(
            r.converged and ratio(r) < 1 - STALL_TOLERANCE for r in mich.values()
        ),
        **lo_hi('s1d_lstm_disp', (disp[s] for s in lstms), three),
        **lo_hi('s1d_lstm_vel', (vel[s] for s in lstms), three),
        's1d_no_disp': three(disp['neuraloperator-offline-44k']),
        's1d_no_vel': three(vel['neuraloperator-offline-44k']),
        's1d_tcn_disp': three(disp['tcn-44k']),
        's1d_tcn_vel': three(vel['tcn-44k']),
        # The text: retargeting helps the LSTMs and costs the neural operator
        # and the TCN, and every retargeted cell stays behind the
        # velocity-trained TCN.
        's1d_shape': all(disp[s] < vel[s] for s in lstms)
        and all(disp[s] > vel[s] for s in ('neuraloperator-offline-44k', 'tcn-44k'))
        and min(disp.values()) > vel['tcn-44k'],
        's1d_mamba_velocity': sig(mean_metric(mamba, 'velocity_rmse'), 2),
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
        'pred_nominal_ratio': ratio('nominal'),
        'pred_c09_ratio': ratio('nominal-c0.9'),
        'pred_no_drift_ratio': ratio('nominal-no-phasedrift'),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output

    prior = bounds.motion_prior(
        SimConfig.nominal(SYNTHETIC_WAVELENGTHS_UM, 0.0, NOMINAL_ALPHA)
    )
    causal_k1, _ = nominal_bounds(prior, SYNTHETIC_WAVELENGTHS_UM[:1])
    causal_k3, offline_k3 = nominal_bounds(prior, SYNTHETIC_WAVELENGTHS_UM)
    info = [
        bounds.expected_information(
            SimConfig.nominal(SYNTHETIC_WAVELENGTHS_UM[:k], NOMINAL_C, NOMINAL_ALPHA)
        )
        for k in (1, 3)
    ]
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
        'prior_velocity_std': sig(prior.velocity_var() ** 0.5, 3),
        'phase_factor_c02': sig(bounds.phase_factor(0.2, NOMINAL_ALPHA), 3),
        'phase_factor_c07': sig(bounds.phase_factor(0.7, NOMINAL_ALPHA), 3),
        'causal_k3': sig(causal_k3, 2),
        'offline_k3': sig(offline_k3, 2),
        'info_ratio_k3_k1': sig(info[1] / info[0], 2),
        # The sign-evidence caption: how much more three channels carry than one.
        **lo_hi(
            'sign_k3_k1',
            (
                bounds.sign_information(
                    SimConfig.nominal(SYNTHETIC_WAVELENGTHS_UM, c, NOMINAL_ALPHA),
                    N_SHOTS,
                    SEED,
                )
                / bounds.sign_information(
                    SimConfig.nominal(SYNTHETIC_WAVELENGTHS_UM[:1], c, NOMINAL_ALPHA),
                    N_SHOTS,
                    SEED,
                )
                for c in TABLE_C
            ),
            lambda x: sig(x, 2),
        ),
        'causal_ratio_k3_k1': sig(causal_k3 / causal_k1, 2),
        'real_tcn_rmse': sig(real, 4),
        'real_gap': f'{real / causal_k3:.0f}',
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
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(values, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
