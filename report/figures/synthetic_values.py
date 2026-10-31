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
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import (
    DATA_DIR,
    NOMINAL_PHASE_DRIFT_FRINGES,
    Q3_FEEDBACK_C,
    SYNTHETIC_WAVELENGTHS_UM,
    LadderRow,
    mean_metric,
    noise_ladder,
    noise_seeds,
    synthetic_cell,
    synthetic_records,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from smi.analysis import bounds
from smi.analysis.experiments import expand_grid
from smi.analysis.models.factory import create_model
from smi.analysis.models.sliding_window import SlidingWindowModel
from smi.analysis.results_artifact import RunRecord
from smi.synthetic.forward import (
    NOMINAL_ALPHA,
    NOMINAL_PHASE_DRIFT_RAD_PER_SQRT_S,
    SimConfig,
)
from smi.synthetic.frozen import DEFAULT_ROOT, load_or_create

# Matches plot_velocity_bounds.py and plot_sign_information.py.
N_STEPS = 2000
N_SHOTS = 8
SEED = 0
NOMINAL_C = 0.2
CEILING_GRID = (
    Path(__file__).resolve().parents[2] / 'smi/analysis/experiments/q3_s0c_ceiling.yaml'
)
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


def pct(x: float) -> str:
    """``x`` as a whole-number percentage."""
    return f'{100 * x:.0f}'


def test_shots(record: RunRecord) -> int:
    """Shot count of the frozen set, from its dataset key ``<hash>_seed<s>_n<n>``.

    The record stores the set's seed but not its size; the key is written by
    smi.synthetic.frozen.frozen_set_path.
    """
    return int(record.dataset.rsplit('_n', 1)[1])


def zero_predictor_rmse(record: RunRecord) -> float:
    """Test RMSE of predicting zero velocity on the alpha = 0 control's test set.

    That is each shot's velocity RMS, averaged over shots. The control's sim
    config comes from expanding the grid that produced the record, so the
    emitter cannot drift from the runner; the hash check stops a grid edit
    since the run from scoring another set.
    """
    _, cells = expand_grid(CEILING_GRID)
    (cfg,) = [cell.sim for cell in cells if cell.key == record.cell_key]
    assert cfg.hash() == record.sim_config_hash, record.cell_key
    assert record.test_seed is not None
    path = load_or_create(cfg, record.test_seed, test_shots(record), DEFAULT_ROOT)
    with h5py.File(path, 'r') as f:
        velocity = f['velocity'][:]
    return float(np.sqrt((velocity**2).mean(axis=1)).mean())


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


def q3_values(records: dict[str, RunRecord]) -> dict[str, str | bool]:
    """Values section 2 quotes from the fixed-budget accuracy-against-C grids."""

    def rmse(family: str, sim: str, metric: str = 'velocity_rmse') -> float:
        return mean_metric(synthetic_cell(records, family, sim), metric)

    def sweep(family: str, metric: str = 'velocity_rmse') -> list[float]:
        return [rmse(family, f'smi-a3-feedback_c={c}', metric) for c in Q3_FEEDBACK_C]

    t90, t30 = 'tcn-44k-T90', 'tcn-44k'
    curve = sweep(t90)
    t30_curve = sweep(t30)
    curve_si = sweep(t90, 'velocity_rmse_sign_invariant')
    michelson = rmse(t90, 'michelson', 'velocity_rmse_sign_invariant')
    michelson_n = rmse(t90, 'michelson', 'velocity_nrmse_sign_invariant')
    top = f'smi-a3-feedback_c={Q3_FEEDBACK_C[-1]}'
    c02 = 'smi-a3-feedback_c=0.2'
    a5_gap = max(
        abs(rmse(f, 'smi-a5') - rmse(f, c02)) / rmse(f, c02) for f in (t90, t30)
    )
    a0 = [synthetic_cell(records, f, 'smi-a0') for f in (t90, t30)]
    t30_epochs = {
        c: synthetic_cell(records, t30, f'smi-a3-feedback_c={c}').epochs_run
        for c in Q3_FEEDBACK_C
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
        synthetic_cell(records, t90, f'smi-a3-feedback_c={c}').metrics[
            'velocity_nrmse'
        ]['max']
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
        sims = ('michelson', *(f'smi-a3-feedback_c={c}' for c in Q3_FEEDBACK_C))
        bins = [synthetic_cell(records, family, sim).binned for sim in sims]
        return [b['abs_velocity'][0]['rmse_sign_invariant'] for b in bins if b]

    slow = slowest(t90)
    t90_epochs = [
        synthetic_cell(records, t90, f'smi-a3-feedback_c={c}').epochs_run or 0
        for c in Q3_FEEDBACK_C
    ]
    a0_t90 = synthetic_cell(records, t90, 'smi-a0')
    zero_gap = abs(
        mean_metric(a0_t90, 'velocity_rmse') / zero_predictor_rmse(a0_t90) - 1
    )
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
        'q3_n_worst_lo': sig(min(worst), 3),
        'q3_n_worst_hi': sig(max(worst), 3),
        'q3_flip_c': f'{Q3_FEEDBACK_C[n_flip]:g}',
        'q3_slow_lo': sig(min(slow), 2),
        'q3_slow_hi': sig(max(slow), 2),
        # The text calls the slow-decile floor a budget property; the
        # document asserts that it is lower at 90 epochs in every cell.
        'q3_slow_falls_with_budget': len(slow) == len(Q3_FEEDBACK_C) + 1
        and all(a < b for a, b in zip(slow, slowest(t30), strict=True)),
        'q3_t90_epochs_lo': str(min(t90_epochs)),
        'q3_t90_epochs_hi': str(max(t90_epochs)),
        'q3_zero_gap_pct': sig(100 * zero_gap, 1),
        'q3_a5_gap_pct': pct(a5_gap),
        'q3_a0_epochs': str(max(r.epochs_run or 0 for r in a0)),
        # eq-sign-info depends on alpha only through alpha^2 / (1 + alpha^2).
        'sign_info_a5_a3': sig(alpha_factor(5.0) / alpha_factor(3.0), 3),
        # The 30-epoch curve's one inversion, which the text attributes to the
        # C = 0.4 cell's longer run; asserted by the document like q3_monotone.
        'q3_t30_c04_below': all(
            t30_curve[Q3_FEEDBACK_C.index(0.4)] < t30_curve[Q3_FEEDBACK_C.index(c)]
            for c in (0.7, 0.9)
        ),
        'q3_t30_c04_epochs': str(t30_epochs[0.4]),
        'q3_t30_other_epochs_lo': str(min(others)),
        'q3_t30_other_epochs_hi': str(max(others)),
        'q3_c02_t270': sig(rmse('tcn-44k-T270', c02), 2),
        'q3_t270_epochs': str(long.epochs_run),
        'q3_t270_cycle': str(long.T_0),
    }


def noise_values(
    records: dict[str, RunRecord], prior: bounds.StateSpace
) -> dict[str, str | bool]:
    """Values the noise-ladder section quotes, from the ladder and spread grids."""

    def rmse(record: RunRecord) -> float:
        return mean_metric(record, 'velocity_rmse')

    ladder = noise_ladder(records)
    base_runs = noise_seeds(records, 'ladder-base')
    base = [rmse(r) for r in base_runs]
    nominal = [rmse(r) for r in noise_seeds(records, 'nominal')]
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
    # The sweep runs 0.5, 1, 1.64, 2.7, 5.4 fringes per shot (forward.py):
    # first is half the published low end, the middle three the published
    # range.
    half_fringe, *published_rows, _ = (row.record for row in ladder[phase])
    published = [rmse(r) for r in published_rows]
    (c09,) = noise_seeds(records, 'nominal-c0.9')
    (nominal0,) = [r for r in noise_seeds(records, 'nominal') if r.seed == 0]
    # The causal bound at the 30 dB row's own config (unit visibility, no
    # drift), so it is the floor for exactly what that row adds to the base.
    snr_row = at_nominal(snr)
    snr_bound = float(bounds.pcrb_filter(prior, snr_row.sim, N_STEPS)[-1]) ** 0.5
    snr_cost = rmse(snr_row.record) - base_mean
    # The DC-offset rows that score below every base seed; the text names them
    # as the +/-0.3 and +/-1 rows and reports how long they trained.
    dc_low = [row for row in ladder['dc_offset_range'] if rmse(row.record) < min(base)]
    base_epochs = [r.epochs_run or 0 for r in base_runs]
    rows = [row for rows in ladder.values() for row in rows]
    # Rows that trained past their second warm restart, i.e. into a third
    # cosine cycle, unlike every base seed.
    long_rows = [
        row for row in rows if (row.record.epochs_run or 0) > 2 * (row.record.T_0 or 0)
    ]
    assert all(
        n <= 2 * (r.T_0 or 0) for n, r in zip(base_epochs, base_runs, strict=True)
    )
    short_rows = [row for row in rows if row not in long_rows]

    def above_base(field: str) -> list[object]:
        return [row.value for row in ladder[field] if rmse(row.record) > max(base)]

    snr_nominal = rmse(snr_row.record)
    return {
        # The text: only 8 bits and a 3 kHz bandwidth leave the base spread.
        'noise_electronics_flat': above_base('adc_bits') == [8]
        and above_base('pd_bandwidth_hz') == [3e3],
        # The text: at 20% each drift costs about what 30 dB white noise does.
        'noise_drift_20_like_snr': all(
            abs(rmse(row.record) / snr_nominal - 1) < 0.1
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
        # The text calls the error over the published range nearly flat.
        'noise_phase_flat': max(published) / min(published) < 1.1,
        'noise_phase_half_above_base': rmse(half_fringe) > max(base),
        'noise_phase_half_kept': pct(
            (rmse(half_fringe) - base_mean)
            / (rmse(at_nominal(phase).record) - base_mean)
        ),
        'noise_c09_ratio': sig(rmse(nominal0) / rmse(c09), 2),
        'noise_snr_cost': sig(snr_cost, 2),
        'noise_snr_bound': sig(snr_bound, 2),
        'noise_snr_cost_over_bound': f'{snr_cost / snr_bound:.0f}',
        'noise_snr_above_bound': snr_cost > snr_bound,
        'noise_dc_low_named': sorted(row.value[1] for row in dc_low) == [0.3, 1.0],
        'noise_dc_low_epochs': '/'.join(
            sorted({str(row.record.epochs_run) for row in dc_low})
        ),
        'noise_base_epochs_lo': str(min(base_epochs)),
        'noise_base_epochs_hi': str(max(base_epochs)),
        'noise_n_long': str(len(long_rows)),
        'noise_n_rows': str(len(rows)),
        'noise_long_epochs_lo': str(min(r.record.epochs_run or 0 for r in long_rows)),
        'noise_long_epochs_hi': str(max(r.record.epochs_run or 0 for r in long_rows)),
        'noise_short_epochs_lo': str(min(r.record.epochs_run or 0 for r in short_rows)),
        'noise_short_epochs_hi': str(max(r.record.epochs_run or 0 for r in short_rows)),
    }


# Section Q2's families as (runner family stem, table label), causal column
# first. Each ran as ``<stem>-T90`` from configs/matched/<stem>.yaml. Mamba's
# stage (q3_s1m_mamba.yaml) may not have run yet: its row reads "pending"
# until its records exist, and the text's per-family claims name only the
# other families, so its arrival fills the table without contradicting them.
# Claims over all families (the best at C = 0.2, the oracle comparisons)
# include it once it exists, so a Mamba row that overturns one fails the build.
Q2_CAUSAL = (
    ('tcn-44k', 'TCN'),
    ('tcan-44k', 'TCAN'),
    ('patchtransformer-causal-44k', 'PatchTransformer'),
    ('unettransformer-causal-44k', 'UNet-Transformer'),
    ('neuraloperator-causal-44k', 'Neural operator'),
    ('lstm-44k', 'LSTM'),
    ('mamba-44k', 'Mamba'),
)
Q2_OFFLINE = (
    ('scnn-44k', 'SCNN'),
    ('barland-44k', 'Barland CNN'),
    ('mlp-44k', 'MLP'),
    ('patchtransformer-offline-44k', 'PatchTransformer'),
    ('unettransformer-offline-44k', 'UNet-Transformer'),
    ('neuraloperator-offline-44k', 'Neural operator'),
    ('lstm-bidir-44k', 'Bidirectional LSTM'),
)
Q2_OPTIONAL = ('mamba-44k',)
Q2_STEMS = [stem for stem, _ in Q2_CAUSAL + Q2_OFFLINE]
Q2_NAMED = [stem for stem in Q2_STEMS if stem not in Q2_OPTIONAL]
# Each sim's score: Michelson's global sign is unobservable (section 1.1).
Q2_SIMS = {'michelson': 'velocity_rmse_sign_invariant', 'smi-a3': 'velocity_rmse'}
MATCHED_DIR = (
    Path(__file__).resolve().parents[2] / 'smi/analysis/models/configs/matched'
)
# Long enough to hold Barland's widened field with margin on both sides.
SPAN_PROBE_LENGTH = 2048
# The text calls a cell stalled when its best validation error stays within
# this fraction of the zero predictor's (the runner's own test, min_rel_delta).
STALL_TOLERANCE = 0.01
# A converged cell stopped well before its first restart; the rest end at it.
EARLY_STOP_FRACTION = 0.95
# The text says no family comes near the Michelson oracle; a ratio this large
# is well clear of the seed range, which a bare `oracle < best` would not be.
ORACLE_MARGIN = 1.5
# The text calls the top offline families' span close to the seed range.
CLOSE_FACTOR = 1.5


def sliding_span(stem: str) -> int | None:
    """Samples one output of a sliding-window model depends on; else None.

    The record's ``receptive_field`` counts one window, but every output
    averages all the windows that cover it, so the field is wider. Measured by
    a gradient probe of a seeded init, exact here because the windows have
    hard edges. Other families' formulas are checked in
    tests/test_receptive_field.py.
    """
    config = yaml.safe_load((MATCHED_DIR / f'{stem}.yaml').read_text('utf-8'))
    torch.manual_seed(SEED)
    model = create_model(config['model']).eval()
    if not isinstance(model, SlidingWindowModel):
        return None
    x = torch.randn(1, config['model']['in_channels'], SPAN_PROBE_LENGTH)
    x.requires_grad_(True)
    model(x)[0, 0, SPAN_PROBE_LENGTH // 2].backward()
    assert x.grad is not None
    hit = (x.grad.abs().sum(dim=1)[0] > 0).nonzero().flatten()
    return int(hit.max() - hit.min()) + 1


def q2_record(records: dict[str, RunRecord], stem: str, sim: str) -> RunRecord | None:
    """The family's seed-0 cell, or None for a family whose stage has not run."""
    key = f'{stem}-T90__ch635-675-515__{sim}__seed0'
    if key not in records and stem in Q2_OPTIONAL:
        return None
    return records[key]


def q2_table(records: dict[str, RunRecord]) -> dict[str, object]:
    """The rows of section Q2's table, and checks on what they are labeled."""

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
        if r.receptive_field is None:
            return 'unbounded' if r.causal else 'whole shot'
        span = sliding_span(stem)
        return str(r.receptive_field if span is None else span)

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
        mean_metric(records[f'inverse_fit__ch635-675-515__{sim}__oracle'], metric)
        for sim, metric in Q2_SIMS.items()
    ]
    stalled = sorted(
        (stem, sim)
        for stem in Q2_STEMS
        for sim in Q2_SIMS
        if (r := q2_record(records, stem, sim)) is not None and not r.converged
    )
    sliding = {s: span for s in Q2_NAMED if (span := sliding_span(s)) is not None}
    mlp = records['mlp-44k-T90__ch635-675-515__smi-a3__seed0']
    # The families whose context is a fixed window, which the text says are
    # matched; the sliding-window families' wider spans are named apart.
    matched = {
        stem: r.receptive_field
        for stem in Q2_NAMED
        if stem not in sliding
        and (r := q2_record(records, stem, 'smi-a3')) is not None
        and r.receptive_field is not None
    }
    return {
        'q2_causal_rows': rows(Q2_CAUSAL, True),
        'q2_offline_rows': rows(Q2_OFFLINE, False),
        'q2_oracle_cells': [f'{v:.3f}' for v in oracle],
        'q2_mamba_pending': q2_record(records, 'mamba-44k', 'smi-a3') is None,
        'q2_stalled_named': stalled
        == [
            ('lstm-44k', 'michelson'),
            ('lstm-44k', 'smi-a3'),
            ('lstm-bidir-44k', 'michelson'),
            ('lstm-bidir-44k', 'smi-a3'),
            ('neuraloperator-offline-44k', 'smi-a3'),
        ],
        'q2_stall_tol_pct': pct(STALL_TOLERANCE),
        'q2_stall_epochs': '/'.join(
            sorted(
                {
                    str(records[f'{s}-T90__ch635-675-515__{m}__seed0'].epochs_run)
                    for s, m in stalled
                }
            )
        ),
        'q2_sliding_named': sorted(sliding) == ['barland-44k', 'mlp-44k'],
        'q2_mlp_span': str(sliding['mlp-44k']),
        'q2_barland_span': str(sliding['barland-44k']),
        'q2_window_field': str(mlp.receptive_field),
        'q2_matched_named': sorted(matched)
        == sorted(
            [
                'tcn-44k',
                'tcan-44k',
                'patchtransformer-causal-44k',
                'unettransformer-causal-44k',
                'neuraloperator-causal-44k',
                'scnn-44k',
                'patchtransformer-offline-44k',
                'unettransformer-offline-44k',
            ]
        ),
        'q2_matched_lo': str(min(matched.values())),
        'q2_matched_hi': str(max(matched.values())),
    }


def q2_values(records: dict[str, RunRecord]) -> dict[str, object]:
    """Values section Q2 quotes from the S1 and S1m family grids."""

    def trained(sim: str, among: list[str]) -> dict[str, float]:
        """Test score of each converged cell of ``among`` on ``sim``."""
        return {
            s: mean_metric(r, Q2_SIMS[sim])
            for s in among
            if (r := q2_record(records, s, sim)) is not None and r.converged
        }

    # The ranking resolution: the noise-free base's three-seed range over its
    # mean (section 2B's grid, at the 30-epoch budget). Two cells closer than
    # this are not ranked apart.
    base = [
        mean_metric(r, 'velocity_rmse') for r in noise_seeds(records, 'ladder-base')
    ]
    resolution = (max(base) - min(base)) / (sum(base) / len(base))

    def apart(a: float, b: float) -> bool:
        """Whether ``b`` exceeds ``a`` by more than the seed range."""
        return b / a - 1 > resolution

    mich, smi = trained('michelson', Q2_STEMS), trained('smi-a3', Q2_STEMS)
    mich_named, smi_named = trained('michelson', Q2_NAMED), trained('smi-a3', Q2_NAMED)
    oracle_mich, oracle_smi = (
        records[f'inverse_fit__ch635-675-515__{sim}__oracle'] for sim in Q2_SIMS
    )
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
    below_tcn = sorted(s for s, v in smi_named.items() if apart(v, tcn_smi))
    above_tcn = sorted(s for s, v in smi_named.items() if apart(tcn_smi, v))
    epochs = [
        (r.epochs_run or 0, r.T_0 or 0)
        for s in Q2_NAMED
        for sim in Q2_SIMS
        if (r := q2_record(records, s, sim)) is not None and r.converged
    ]
    early = [n for n, t in epochs if n < EARLY_STOP_FRACTION * t]
    late = [n for n, t in epochs if n >= EARLY_STOP_FRACTION * t]
    second = [n for n, t in epochs if n > t]
    # The zero predictor's test RMSE: the alpha = 0 control's frozen set shares
    # the oracle's test seed, so it scores the same velocities.
    a0 = synthetic_cell(records, 'tcn-44k-T90', 'smi-a0')
    assert a0.test_seed == oracle_smi.test_seed
    zero = zero_predictor_rmse(a0)
    oracle_signed = mean_metric(oracle_smi, 'velocity_rmse')
    return {
        'q2_seed_range_pct': pct(resolution),
        # Michelson: every trained family but the offline neural operator lies
        # within the seed range, and that one lies above the rest.
        'q2_mich_tied': weakest == 'neuraloperator-offline-44k'
        and not apart(min(rest), max(rest))
        and apart(max(rest), mich_named[weakest]),
        'q2_mich_lo': sig(min(rest), 2),
        'q2_mich_hi': sig(max(rest), 2),
        'q2_mich_span_pct': pct(max(rest) / min(rest) - 1),
        'q2_no_offline_mich': sig(mich_named[weakest], 2),
        'q2_oracle_mich': sig(oracle, 2),
        'q2_oracle_gap': sig(min(mich.values()) / oracle, 2),
        'q2_oracle_beats_all': ORACLE_MARGIN * oracle <= min(mich.values()),
        # C = 0.2: the families the text names as beating or trailing the TCN.
        'q2_smi_vs_tcn': below_tcn
        == sorted(
            [
                'tcan-44k',
                'patchtransformer-causal-44k',
                'scnn-44k',
                'barland-44k',
                'patchtransformer-offline-44k',
                'unettransformer-offline-44k',
            ]
        )
        and above_tcn
        == sorted(
            ['unettransformer-causal-44k', 'neuraloperator-causal-44k', 'mlp-44k']
        ),
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
        'q2_early_lo': str(min(early)),
        'q2_early_hi': str(max(early)),
        'q2_late_lo': str(min(late)),
        'q2_late_hi': str(max(late)),
        'q2_restart': str(max(t for _, t in epochs)),
        'q2_second_n': str(len(second)),
        'q2_second_lo': str(min(second)),
    }


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
    records = synthetic_records()

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
        **exit_check_values(records),
        **q3_values(records),
        **noise_values(records, prior),
        **q2_table(records),
        **q2_values(records),
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
