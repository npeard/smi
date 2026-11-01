"""Shared matplotlib settings and helpers for report figures.

Every ``plot_*.py`` in this directory imports from here rather than setting its
own rcParams, so figures share the same build-reproducibility settings and the
same channel-color mapping. Everything else -- fonts, sizes, spines, grid,
tick direction, legend frame -- is left at matplotlib's own defaults, the same
as loading a figure into a LaTeX document: no house theme is imposed here.

Contract each figure script honors (relied on by the Snakefile and by
``tests/test_report_build.py``):

- it accepts ``--output PATH`` and writes exactly that path;
- it is deterministic, so re-running produces the same bytes and the DAG's
  timestamp-based caching stays trustworthy.

Determinism note: matplotlib stamps a creation date into PDF metadata by
default, which would make otherwise-identical runs differ byte-for-byte.
``save`` pins ``pdf.compression`` and clears the date so that does not happen.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.typing import RcKeyType
from scipy import signal

from smi.analysis.experiments import (
    ANALYSIS_DIR,
    ORACLE_FAMILY,
    Cell,
    cell_key,
    expand_grid,
    training_config,
)
from smi.analysis.results_artifact import RunRecord, read_results
from smi.core.acquisition import STEMLAB_125_14
from smi.data.frozen import RIG_WAVELENGTHS_UM
from smi.data.schema import DRIVE_KEY, PD_WAVELENGTHS_UM, read_sample_rate
from smi.theory import bounds
from smi.theory.forward import (
    NOMINAL_ALPHA,
    NOMINAL_FEEDBACK_C,
    SimConfig,
    simulate_batch,
)

# Resolved by the package, not the cwd (pytest runs scripts from the repo
# root, Snakemake from report/).
DATA_DIR = ANALYSIS_DIR / 'data'
EXPERIMENTS_DIR = ANALYSIS_DIR / 'experiments'
MATCHED_DIR = ANALYSIS_DIR / 'models' / 'configs' / 'matched'

FREE_SPACE_H5 = DATA_DIR / 'free-space-synchro_10k.h5'
MMFIBER_H5 = DATA_DIR / 'mmfiber-synchro_10k.h5'

# Baseline results, written by smi.experiments.baselines and committed. Figures
# read this rather than recomputing: method 2's Viterbi decode is ~20 minutes
# per acquisition on a GPU, which is not work a document build should be
# repeating, and reading a fixed artifact is what makes the figures
# deterministic. (Method 1's fits are ~1 minute on CPU at N = 1000; it is
# method 2 that makes regeneration expensive.)
RESULTS_JSON = DATA_DIR / 'baseline_results.json'

# Experiment-runner records behind synthetic.typ's learned-model results.
SYNTHETIC_RESULTS_JSON = DATA_DIR / 'synthetic_results.json'

# The same lasers as the synthetic simulator spells them, for figures that
# compute from smi.theory.
SYNTHETIC_WAVELENGTHS_UM = RIG_WAVELENGTHS_UM

# The feedback values the Q3 grids sweep at alpha = 3.
Q3_FEEDBACK_C = (0.05, 0.1, 0.2, 0.4, 0.7, 0.9)


def c_sim(c: float) -> str:
    """The Q3 grids' sim name for feedback ``c`` at alpha = 3."""
    return f'smi-a3-feedback_c={c}'


# The sign-collapse controls of synthetic.typ sections 1.1 (the exit check's
# signed Michelson run) and 2A (alpha = 0 at both schedules). These are the
# only records without zero_predictor_monitor (written before the min_epochs
# grace) that patience stopped inside the grace, at epoch 11 on the
# zero-output plateau. The report quotes exactly that collapse, so they are
# exempt from check_grace_protocol; every other such record ran past the grace
# or to its epoch cap.
SIGN_COLLAPSE_CONTROLS = frozenset(
    cell_key(family, SYNTHETIC_WAVELENGTHS_UM, sim, 0)
    for family, sim in (
        ('tcn-44k-E20', 'michelson-signed'),
        ('tcn-44k-T90', 'smi-a0'),
        ('tcn-44k', 'smi-a0'),
    )
)


def check_grace_protocol(
    records: Iterable[RunRecord],
    min_epochs: Mapping[str, int],
    expected_collapses: Collection[str],
) -> None:
    """Reject a record trained before the min_epochs grace that the grace would change.

    A record without ``zero_predictor_monitor`` was trained before
    ``training.convergence.min_epochs`` existed, under the converged rule that
    did not compare against the zero predictor. It is equivalent to a record
    of today's protocol when it ran past the grace, hit its epoch cap (no
    patience stop to withdraw), or was never trained (``epochs_run`` None).
    One that patience stopped inside the grace is not: today the run would
    have continued and might have escaped, so mixing it with new records
    would silently mix protocols.

    Args:
        records: Records to check.
        min_epochs: The grace of each cell key under the current configs; a
            legacy record whose key is absent has no grace to compare against
            and is rejected like one stopped inside it.
        expected_collapses: Cell keys of such records the caller quotes as
            zero-output collapses stopped by patience under the old protocol,
            which is what the caller reports them as.

    Raises:
        ValueError: Naming each offending cell key.
    """
    stale = [
        record.cell_key
        for record in records
        if record.zero_predictor_monitor is None
        and record.epochs_run is not None
        and record.epochs_run != record.max_epochs
        and record.cell_key not in expected_collapses
        and (
            record.cell_key not in min_epochs
            or record.epochs_run <= min_epochs[record.cell_key]
        )
    ]
    if stale:
        raise ValueError(
            'records trained before the min_epochs grace stopped inside it, or '
            'match no cell of a shipped grid; rerun them or name them as '
            f'expected collapses: {stale}'
        )


def grace_by_cell() -> dict[str, int]:
    """``training.convergence.min_epochs`` of every trained cell of every grid."""
    grace = {}
    for path in sorted(EXPERIMENTS_DIR.glob('*.yaml')):
        grid, cells = expand_grid(path)
        for cell in cells:
            if cell.seed is not None:
                config = training_config(cell, grid)
                grace[cell.key] = config.training_config['convergence']['min_epochs']
    return grace


def synthetic_records() -> dict[str, RunRecord]:
    """The synthetic results artifact's records, keyed by runner cell key.

    Pre-grace early-stopped cells other than SIGN_COLLAPSE_CONTROLS raise (see
    ``check_grace_protocol``).
    """
    records = read_results(SYNTHETIC_RESULTS_JSON)
    # Only pre-grace records can fail the check, and resolving every grid's
    # training config is the slow part, so skip it when none remain.
    legacy = [r for r in records if r.zero_predictor_monitor is None]
    check_grace_protocol(
        legacy, grace_by_cell() if legacy else {}, SIGN_COLLAPSE_CONTROLS
    )
    return {r.cell_key: r for r in records if r.cell_key is not None}


def synthetic_cell(records: dict[str, RunRecord], family: str, sim: str) -> RunRecord:
    """The seed-0, three-channel record of ``family`` on ``sim``.

    Every synthetic.typ figure reads three-channel seed-0 cells; a missing one
    raises here, naming the key, rather than plotting a gap.
    """
    return records[cell_key(family, SYNTHETIC_WAVELENGTHS_UM, sim, 0)]


def oracle_cell(records: dict[str, RunRecord], sim: str) -> RunRecord:
    """The three-channel inverse-fit oracle's record on ``sim``."""
    return records[cell_key(ORACLE_FAMILY, SYNTHETIC_WAVELENGTHS_UM, sim, None)]


def mean_metric(record: RunRecord, metric: str) -> float:
    """Mean over test shots of a per-shot metric of ``record``."""
    return record.metrics[metric]['mean']


# The noise ladder (each knob swept alone from the noise-free base) and the
# grid that reruns its base and nominal cells at more seeds.
CEILING_GRID = EXPERIMENTS_DIR / 'q3_s0c_ceiling.yaml'
NOISE_LADDER_GRID = EXPERIMENTS_DIR / 'q3_s3_noise_ladder.yaml'
NOISE_SPREAD_GRID = EXPERIMENTS_DIR / 'q3_s3_spread.yaml'
LEAVE_ONE_OUT_GRID = EXPERIMENTS_DIR / 'q3_s3_leave_one_out.yaml'
MATCHED_LENGTH_GRID = EXPERIMENTS_DIR / 'q3_s3_matched_length.yaml'
# The noise knobs, as (SimConfig field, display label), in the order the
# noise-ladder figure and the leave-one-out table show them.
NOISE_KNOBS = {
    'phase_drift_rad_per_sqrt_s': 'phase drift',
    'snr_db': 'white noise',
    'visibility_range': 'visibility',
    'amplitude_drift_rel': 'amplitude drift',
    'contrast_drift_rel': 'contrast drift',
    'dc_offset_range': 'DC offset',
    'pd_bandwidth_hz': 'PD bandwidth',
    'adc_bits': 'ADC',
}


@dataclass(frozen=True)
class LadderRow:
    """One single-knob cell of the noise ladder.

    Attributes:
        field: The swept SimConfig field.
        value: Its value in this cell.
        nominal: Whether that value is SimConfig.nominal's.
        sim: The cell's sim config.
        record: The cell's run record.
    """

    field: str
    value: object
    nominal: bool
    sim: SimConfig
    record: RunRecord


_NOMINAL_SIM = SimConfig.nominal(
    STEMLAB_125_14, SYNTHETIC_WAVELENGTHS_UM, NOMINAL_FEEDBACK_C, NOMINAL_ALPHA
)


def _is_nominal(field: str, value: object) -> bool:
    nominal = getattr(_NOMINAL_SIM, field)
    if isinstance(value, float) and isinstance(nominal, float):
        return math.isclose(value, nominal)
    return value == nominal


def grid_cells(
    records: dict[str, RunRecord], grid: Path
) -> list[tuple[Cell, RunRecord]]:
    """``(cell, record)`` per cell of ``grid``, hash-checked.

    The cell's sim config comes from expanding the grid, so a grid edited after
    its run fails here instead of labelling a record with the wrong knob value.

    Raises:
        ValueError: Naming the cell whose record's sim hash differs.
    """
    _, cells = expand_grid(grid)
    out = []
    for cell in cells:
        record = records[cell.key]
        if record.sim_config_hash != cell.sim.hash():
            raise ValueError(f'{cell.key}: record was run under another sim config')
        out.append((cell, record))
    return out


def noise_ladder(records: dict[str, RunRecord]) -> dict[str, list[LadderRow]]:
    """The ladder's single-knob rows, keyed by swept field, in sweep order."""
    rows: dict[str, list[LadderRow]] = {}
    for cell, record in grid_cells(records, NOISE_LADDER_GRID):
        if cell.sweep is None:
            continue
        field, _ = cell.sweep
        value = getattr(cell.sim, field)
        rows.setdefault(field, []).append(
            LadderRow(field, value, _is_nominal(field, value), cell.sim, record)
        )
    return rows


def noise_seeds(records: dict[str, RunRecord], sim_name: str) -> list[RunRecord]:
    """Every seed of ``sim_name`` across the ladder, spread and leave-one-out grids."""
    found = {
        record.cell_key: record
        for grid in (NOISE_LADDER_GRID, NOISE_SPREAD_GRID, LEAVE_ONE_OUT_GRID)
        for cell, record in grid_cells(records, grid)
        if cell.sim_name == sim_name
    }
    return list(found.values())


def matched_length_cells(
    records: dict[str, RunRecord],
) -> tuple[dict[str, RunRecord], dict[tuple[str, object], RunRecord]]:
    """The equal-length grid's records, keyed by sim name and by sweep point.

    The second mapping is keyed by ``Cell.sweep``, so a ladder row's
    ``(field, value)`` finds its 60-epoch rerun of the same knob setting.
    """
    cells = grid_cells(records, MATCHED_LENGTH_GRID)
    by_name = {cell.sim_name: record for cell, record in cells}
    by_sweep = {cell.sweep: record for cell, record in cells if cell.sweep}
    return by_name, by_sweep


# Section Q1's noise-free channel-subset grid, and the per-sample binned
# errors smi.analysis.q1_binned writes from the Q1 checkpoints, which are
# gitignored and so cannot be read at build time.
Q1_S2_GRID = EXPERIMENTS_DIR / 'q1_s2_subsets.yaml'
Q1_BINNED_JSON = DATA_DIR / 'q1_binned_errors.json'
# The seven subsets of the Q1 grids, singles first, as (wavelengths, label).
Q1_SUBSETS: tuple[tuple[tuple[float, ...], str], ...] = (
    ((0.635,), '635'),
    ((0.6748,), '674.8'),
    ((0.515,), '515'),
    ((0.635, 0.6748), '635 + 674.8'),
    ((0.635, 0.515), '635 + 515'),
    ((0.6748, 0.515), '674.8 + 515'),
    ((0.635, 0.6748, 0.515), 'all three'),
)
# The record length the causal bound has converged by and the smoother's
# mid-record value is stationary at (see plot_velocity_bounds.py).
BOUND_STEPS = 2000


def velocity_bounds(prior: bounds.StateSpace, cfg: SimConfig) -> tuple[float, float]:
    """Stationary RMS velocity bounds of ``cfg``: causal, then offline."""
    causal = float(bounds.pcrb_filter(prior, cfg, BOUND_STEPS)[-1]) ** 0.5
    offline = float(bounds.bcrb_smoother(prior, cfg, BOUND_STEPS)[BOUND_STEPS // 2])
    return causal, offline**0.5


# Shots averaged per spectrum in the data/target mismatch figure and the
# values quoted beside it.
MISMATCH_SHOTS = 256


@dataclass(frozen=True)
class MismatchSpectra:
    """Shot-averaged one-sided PSDs of the targets and of the traces' fringe rate.

    Each PSD is divided by its own maximum over the nonzero bins, so the
    curves compare shapes, not units.

    Attributes:
        freq_hz: Bin frequencies, DC excluded.
        velocity: PSD of the velocity target.
        displacement: PSD of the displacement target.
        fringe_rate: ``[C, F]`` PSD of each channel's instantaneous fringe
            frequency, in SYNTHETIC_WAVELENGTHS_UM order.
    """

    freq_hz: np.ndarray
    velocity: np.ndarray
    displacement: np.ndarray
    fringe_rate: np.ndarray


def mismatch_spectra(noise_free: bool) -> MismatchSpectra:
    """Spectra of MISMATCH_SHOTS seeded shots at the nominal operating point.

    The fringe rate is estimated from the analytic signal: the phase of
    ``scipy.signal.hilbert`` of the mean-removed trace, unwrapped and
    differentiated with ``np.gradient``, divided by 2 pi. For a clean fringe
    this tracks 2|v|/lambda. Every series is mean-removed and Hann-windowed
    over the whole shot (``scipy.signal.periodogram``), so the lowest nonzero
    bin is 1 / shot length.

    Args:
        noise_free: Simulate with every detector knob at its identity value
            instead of at nominal. The trajectory draws come first, so both
            settings see the same motion.
    """
    cfg = _NOMINAL_SIM
    if noise_free:
        cfg = SimConfig.noise_free(
            acquisition=STEMLAB_125_14,
            wavelengths_um=cfg.wavelengths_um,
            sample_rate=cfg.sample_rate,
            start_freq_hz=cfg.start_freq_hz,
            end_freq_hz=cfg.end_freq_hz,
            drive_gain=cfg.drive_gain,
            feedback_c=cfg.feedback_c,
            alpha=cfg.alpha,
        )
    batch = simulate_batch(
        cfg,
        MISMATCH_SHOTS,
        torch.device('cpu'),
        torch.Generator().manual_seed(FIGURE_SEED),
    )
    fs = cfg.sample_rate

    def psd(x: np.ndarray) -> np.ndarray:
        _, p = signal.periodogram(x, fs=fs, window='hann', detrend='constant', axis=-1)
        p = p.mean(axis=0)[..., 1:]
        return p / p.max(axis=-1, keepdims=True)

    traces = batch.signals.double().numpy()
    analytic = signal.hilbert(traces - traces.mean(axis=-1, keepdims=True), axis=-1)
    phase = np.unwrap(np.angle(analytic), axis=-1)
    fringe_rate = np.gradient(phase, axis=-1) * fs / (2 * np.pi)
    freq = np.fft.rfftfreq(traces.shape[-1], d=1 / fs)[1:]
    return MismatchSpectra(
        freq_hz=freq,
        velocity=psd(batch.velocity.double().numpy()),
        displacement=psd(batch.displacement.double().numpy()),
        fringe_rate=psd(fringe_rate),
    )


# Photodiode channels, in the order the dataset feeds them to the model, with
# the laser wavelength each one detects (microns).
PD_CHANNELS: tuple[tuple[str, float], ...] = tuple(PD_WAVELENGTHS_UM.items())
VOLTAGE_CHANNEL = DRIVE_KEY

# The two acquisitions, as (key in baseline_results.json, axis label). Shared
# so that adding a third cannot leave one figure silently showing two.
BASELINE_DATASETS: tuple[tuple[str, str], ...] = (
    ('free-space', 'free space'),
    ('mmfiber', 'mm fiber'),
)

# The leave-one-out grid's knobs, as (SimConfig field, sim-name stem); each
# cell is named nominal-no-<stem> (q3_s3_leave_one_out.yaml).
LEAVE_ONE_OUT = (
    ('phase_drift_rad_per_sqrt_s', 'phasedrift'),
    ('snr_db', 'snr'),
    ('visibility_range', 'visibility'),
    ('amplitude_drift_rel', 'ampdrift'),
    ('contrast_drift_rel', 'contrastdrift'),
    ('dc_offset_range', 'dc'),
    ('pd_bandwidth_hz', 'pdbw'),
    ('adc_bits', 'adc'),
)


# Section Q2's families as (runner family stem, label), causal first. Each ran
# as ``<stem>-T90`` from configs/matched/<stem>.yaml. Mamba ran as its own
# stage (q3_s1m_mamba.yaml); it stays optional so a fresh artifact without that
# stage still builds, with Mamba marked pending.
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
# A cell is called stalled when its best validation error stays within
# this fraction of the zero predictor's (the runner's own test, min_rel_delta).
STALL_TOLERANCE = 0.01


def q2_record(records: dict[str, RunRecord], stem: str, sim: str) -> RunRecord | None:
    """The family's seed-0 cell, or None for a family whose stage has not run."""
    try:
        return synthetic_cell(records, f'{stem}-T90', sim)
    except KeyError:
        if stem in Q2_OPTIONAL:
            return None
        raise


# Displacement RMSE beside each velocity metric: the predicted velocity
# integrated over the shot, each start offset removed, as LitModule's auxiliary
# displacement loss compares them. In um. smi.analysis.evaluate does not write
# them yet; the rerun plan's backfill re-scores every checkpoint to add them.
DISPLACEMENT_METRICS = {
    'velocity_rmse': 'displacement_rmse',
    'velocity_rmse_sign_invariant': 'displacement_rmse_sign_invariant',
}


# The S1d grid (q3_s1d_displacement.yaml): the families that stalled on the
# velocity target, retrained on displacement, plus a TCN control. Each ran as
# ``<stem>-disp-T90`` on the same sims and test shots as its S1 cell.
S1D_FAMILIES = (
    ('tcn-44k', 'TCN'),
    ('lstm-44k', 'LSTM'),
    ('lstm-bidir-44k', 'Bidirectional LSTM'),
    ('neuraloperator-offline-44k', 'Neural operator (off.)'),
)


def s1d_record(records: dict[str, RunRecord], stem: str, sim: str) -> RunRecord:
    """The displacement-target seed-0 cell of ``stem`` on ``sim``."""
    return synthetic_cell(records, f'{stem}-disp-T90', sim)


def displacement_pending(records: Iterable[RunRecord]) -> bool:
    """Whether none of ``records`` carries displacement RMSE yet.

    Raises:
        ValueError: Naming the records that lack it, when only some do. A
            part-backfilled artifact would otherwise plot a silent mix.
    """
    records = list(records)
    missing = sorted(
        str(r.cell_key) for r in records if 'displacement_rmse' not in r.metrics
    )
    if missing and len(missing) < len(records):
        raise ValueError(f'displacement RMSE missing from only some records: {missing}')
    return bool(missing)


def pending_panel(ax: plt.Axes, what: str) -> None:
    """Mark ``ax`` as awaiting data, so a pending figure cannot pass for a result."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.5,
        0.5,
        f'PENDING\n{what}',
        transform=ax.transAxes,
        ha='center',
        va='center',
        fontsize=10,
        fontweight='bold',
        color='C3',
    )


# Section Q1's two families and their scores (Michelson's global sign is
# unobservable, section 1.1). The S2 grid also expands an oracle cell for its
# SMI sim, which its header leaves unrun: the oracle decodes under the
# Michelson model, so only its Michelson cells are read.
Q1_FAMILIES = ('tcn-44k', 'tcan-44k')
Q1_S2_SIMS = {
    's2-michelson': 'velocity_rmse_sign_invariant',
    's2-smi-a3': 'velocity_rmse',
}
Q1_SNR_DB = (0.0, 10.0, 20.0, 30.0, 40.0)
Q1_SINGLES = [w for w, _ in Q1_SUBSETS if len(w) == 1]
Q1_MULTI = [w for w, _ in Q1_SUBSETS if len(w) > 1]
Q1_PAIRS = [w for w in Q1_MULTI if len(w) == 2]


# Seed used by any figure that needs randomness, so runs are reproducible.
FIGURE_SEED = 20250910

# Named colors for the whole document set, drawn from matplotlib's default
# property cycle (`C0`, `C1`, ...) rather than a hand-picked palette -- see the
# module docstring. The names are what make 635/675/515 nm stable and distinct
# across figures: every figure that plots a channel looks it up by name here
# instead of pulling the next color off its own axes' cycle, which would let
# the same wavelength land on different colors in different figures.
COLORS: dict[str, str] = {
    'drive': 'C5',
    'ch635': 'C3',
    'ch675': 'C1',
    'ch515': 'C2',
    'accent': 'C0',
    'muted': 'C7',
    'highlight': 'C4',
}
CHANNEL_COLORS: tuple[str, ...] = (COLORS['ch635'], COLORS['ch675'], COLORS['ch515'])


@dataclass(frozen=True)
class Budget:
    """One cosine schedule of the Q3 grids, and how its curves are drawn.

    Attributes:
        family: Runner family name.
        epochs: Epochs per cosine cycle.
        color: Line color.
        linestyle: Line style, so the curves differ without color.
        marker: Marker.
    """

    family: str
    epochs: int
    color: str
    linestyle: str
    marker: str


Q3_BUDGETS = (
    Budget('tcn-44k-T90', 90, COLORS['accent'], '-', 'o'),
    Budget('tcn-44k', 30, COLORS['muted'], '--', 's'),
)

# Slide-friendly default: wide enough for a 16:9 body, small enough that 9 pt
# labels stay legible when the PDF is placed at ~90% of the slide width.
FIGSIZE_WIDE = (7.0, 3.4)
FIGSIZE_TALL = (7.0, 4.6)
FIGSIZE_HALF = (3.5, 2.8)

# Keyed by RcKeyType (matplotlib's literal union of valid rcParam names) so a
# typo in a key is a type error rather than a silently ignored setting. Values
# are heterogeneous by key and matplotlib validates each one itself, so Any is
# what its own stub declares.
#
# Deliberately short: these are build-reproducibility and layout settings, not
# a look. Fonts, sizes, spines, grid, legend frame, tick direction and line
# width are all left at matplotlib's own defaults -- see the module docstring.
_RCPARAMS: dict[RcKeyType, Any] = {
    'figure.figsize': FIGSIZE_WIDE,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    # PDF text as real glyphs (Type 42) rather than paths, so the text stays
    # selectable and searchable in the compiled document.
    'pdf.fonttype': 42,
    'pdf.compression': 6,
}


def apply_style() -> None:
    """Install the shared rcParams. Called by ``parse_args``.

    Assigns key by key rather than calling ``rcParams.update``, whose stub has
    no overload matching a plain dict.
    """
    for key, value in _RCPARAMS.items():
        plt.rcParams[key] = value


def parse_args(description: str) -> argparse.Namespace:
    """Parse the standard figure-script CLI and apply the shared style.

    Args:
        description: Short help text for the script.

    Returns:
        Namespace with an ``output`` attribute (a ``Path``).
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--output', type=Path, required=True, help='Exact path of the PDF to write.'
    )
    args = parser.parse_args()
    apply_style()
    return args


def rng() -> np.random.Generator:
    """Return the seeded generator every figure must use for randomness."""
    return np.random.default_rng(FIGURE_SEED)


def save(fig: plt.Figure, output: Path) -> None:
    """Write ``fig`` to ``output``, creating parent directories as needed.

    Metadata is emptied so two runs of the same script differ nowhere,
    including in the PDF's CreationDate.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={'CreationDate': None})
    plt.close(fig)


def read_shot(
    h5_path: Path, channel: str, index: int, decimate: int = 1
) -> tuple[np.ndarray, float]:
    """Read one shot from one channel of an acquisition file.

    A single channel of these files is 0.66 GB, so read exactly one row -- never
    the whole dataset.

    Args:
        h5_path: Path to the acquisition HDF5 file.
        channel: Dataset name, e.g. ``RP1_CH1``.
        index: Row (shot) index.
        decimate: Keep every ``decimate``-th sample of the returned trace.

    Returns:
        Tuple of (trace, sample_rate). ``sample_rate`` is the rate of the
        returned trace, i.e. already divided by ``decimate``.
    """
    with h5py.File(h5_path, 'r') as f:
        trace = np.asarray(f[channel][index, :], dtype=np.float64)
        sample_rate = read_sample_rate(f.attrs)
    if decimate > 1:
        trace = trace[::decimate]
        sample_rate /= decimate
    return trace, sample_rate


def time_axis(n: int, sample_rate: float) -> np.ndarray:
    """Time axis in milliseconds for ``n`` samples at ``sample_rate`` Hz."""
    return np.arange(n) / sample_rate * 1e3


def panel_label(ax: plt.Axes, text: str) -> None:
    """Put a bold ``(a)``-style label in an axes' top-left corner."""
    ax.text(
        0.012,
        0.97,
        text,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=9,
        fontweight='bold',
    )
