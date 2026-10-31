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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.typing import RcKeyType

from smi.analysis.experiments import expand_grid
from smi.analysis.results_artifact import RunRecord, read_results
from smi.synthetic.forward import NOMINAL_ALPHA, NOMINAL_FEEDBACK_C, SimConfig
from smi.synthetic.frozen import RIG_WAVELENGTHS_UM

# Repository root, resolved from this file so scripts work from any cwd
# (pytest runs them from the repo root, Snakemake from report/).
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / 'smi' / 'analysis' / 'data'

FREE_SPACE_H5 = DATA_DIR / 'free-space-synchro_10k.h5'
MMFIBER_H5 = DATA_DIR / 'mmfiber-synchro_10k.h5'

# Baseline results, written by smi.analysis.baseline_fit and committed. Figures
# read this rather than recomputing: method 2's Viterbi decode is ~20 minutes
# per acquisition on a GPU, which is not work a document build should be
# repeating, and reading a fixed artifact is what makes the figures
# deterministic. (Method 1's fits are ~1 minute on CPU at N = 1000; it is
# method 2 that makes regeneration expensive.)
RESULTS_JSON = DATA_DIR / 'baseline_results.json'

# Experiment-runner records behind synthetic.typ's learned-model results.
SYNTHETIC_RESULTS_JSON = DATA_DIR / 'synthetic_results.json'

# The feedback values the Q3 grids sweep at alpha = 3, and the two cosine
# schedules they ran under, as (runner family name, epochs per cosine cycle).
Q3_FEEDBACK_C = (0.05, 0.1, 0.2, 0.4, 0.7, 0.9)
Q3_BUDGETS = (('tcn-44k-T90', 90), ('tcn-44k', 30))


def synthetic_records() -> dict[str, RunRecord]:
    """The synthetic results artifact's records, keyed by runner cell key."""
    records = read_results(SYNTHETIC_RESULTS_JSON)
    return {r.cell_key: r for r in records if r.cell_key is not None}


def synthetic_cell(records: dict[str, RunRecord], family: str, sim: str) -> RunRecord:
    """The seed-0, three-channel record of ``family`` on ``sim``.

    Every synthetic.typ figure reads three-channel seed-0 cells; a missing one
    raises here, naming the key, rather than plotting a gap.
    """
    return records[f'{family}__ch635-675-515__{sim}__seed0']


def mean_metric(record: RunRecord, metric: str) -> float:
    """Mean over test shots of a per-shot metric of ``record``."""
    return record.metrics[metric]['mean']


# The noise ladder (each knob swept alone from the noise-free base) and the
# grid that reruns its base and nominal cells at more seeds.
EXPERIMENTS_DIR = REPO_ROOT / 'smi' / 'analysis' / 'experiments'
NOISE_LADDER_GRID = EXPERIMENTS_DIR / 'q3_s3_noise_ladder.yaml'
NOISE_SPREAD_GRID = EXPERIMENTS_DIR / 'q3_s3_spread.yaml'
LEAVE_ONE_OUT_GRID = EXPERIMENTS_DIR / 'q3_s3_leave_one_out.yaml'
MATCHED_LENGTH_GRID = EXPERIMENTS_DIR / 'q3_s3_matched_length.yaml'
# forward.py sets the nominal phase drift at the geometric middle of the
# published 1 to 2.7 fringes per shot; drift in fringes per shot is linear in
# the rate, so this converts a swept rate to fringes.
NOMINAL_PHASE_DRIFT_FRINGES = math.sqrt(2.7)


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


def _is_nominal(field: str, value: object) -> bool:
    nominal = getattr(
        SimConfig.nominal(SYNTHETIC_WAVELENGTHS_UM, NOMINAL_FEEDBACK_C, NOMINAL_ALPHA),
        field,
    )
    if isinstance(value, float) and isinstance(nominal, float):
        return math.isclose(value, nominal)
    return value == nominal


def _grid_records(
    records: dict[str, RunRecord], grid: Path
) -> list[tuple[str, SimConfig, RunRecord]]:
    """``(sim name, sim, record)`` per cell of ``grid``, hash-checked.

    The sim config comes from expanding the grid, so a grid edited after its
    run fails here instead of labelling a record with the wrong knob value.
    """
    _, cells = expand_grid(grid)
    out = []
    for cell in cells:
        record = records[cell.key]
        assert record.sim_config_hash == cell.sim.hash(), cell.key
        out.append((cell.sim_name, cell.sim, record))
    return out


def noise_ladder(records: dict[str, RunRecord]) -> dict[str, list[LadderRow]]:
    """The ladder's single-knob rows, keyed by swept field, in sweep order."""
    rows: dict[str, list[LadderRow]] = {}
    for name, sim, record in _grid_records(records, NOISE_LADDER_GRID):
        if not name.startswith('ladder-'):
            continue
        # Sweep cells are named <entry>-<field>=<value> (experiments._build_sims).
        field = name.partition('=')[0].split('-', 2)[2]
        value = getattr(sim, field)
        rows.setdefault(field, []).append(
            LadderRow(field, value, _is_nominal(field, value), sim, record)
        )
    return rows


def noise_seeds(records: dict[str, RunRecord], sim_name: str) -> list[RunRecord]:
    """Every seed of ``sim_name`` across the ladder, spread and leave-one-out grids."""
    found = {
        record.cell_key: record
        for grid in (NOISE_LADDER_GRID, NOISE_SPREAD_GRID, LEAVE_ONE_OUT_GRID)
        for name, _, record in _grid_records(records, grid)
        if name == sim_name
    }
    return list(found.values())


def matched_length_cells(records: dict[str, RunRecord]) -> dict[str, RunRecord]:
    """The equal-length grid's records, keyed by sim name."""
    return {
        name: record for name, _, record in _grid_records(records, MATCHED_LENGTH_GRID)
    }


# Photodiode channels, in the order the dataset feeds them to the model, with
# the laser wavelength each one detects (microns).
PD_CHANNELS: tuple[tuple[str, float], ...] = (
    ('RP1_CH2', 0.635),
    ('RP2_CH1', 0.675),
    ('RP2_CH2', 0.515),
)
VOLTAGE_CHANNEL = 'RP1_CH1'

# The same lasers as the synthetic simulator spells them, for figures that
# compute from smi.synthetic.
SYNTHETIC_WAVELENGTHS_UM = RIG_WAVELENGTHS_UM

# The two acquisitions, as (key in baseline_results.json, axis label). Shared
# so that adding a third cannot leave one figure silently showing two.
BASELINE_DATASETS: tuple[tuple[str, str], ...] = (
    ('free-space', 'free space'),
    ('mmfiber', 'mm fiber'),
)

# Fallback sample rate; every real file carries `sample_rate` in its attrs.
DEFAULT_SAMPLE_RATE = 488281.25

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
        sample_rate = float(f.attrs.get('sample_rate', DEFAULT_SAMPLE_RATE))
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
