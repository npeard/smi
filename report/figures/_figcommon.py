"""Shared matplotlib style and helpers for report figures.

Every ``plot_*.py`` in this directory imports from here rather than setting its
own rcParams, so panels in the slides and the report read as one system.

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
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.typing import RcKeyType

# Repository root, resolved from this file so scripts work from any cwd
# (pytest runs them from the repo root, Snakemake from report/).
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / 'smi' / 'analysis' / 'data'

FREE_SPACE_H5 = DATA_DIR / 'free-space-synchro_10k.h5'
MMFIBER_H5 = DATA_DIR / 'mmfiber-synchro_10k.h5'

# Baseline results, written by smi.analysis.baseline_fit and committed. Figures
# read this rather than recomputing: the fits are a GPU job of ~20 minutes per
# acquisition, which is not work a document build should be repeating, and
# reading a fixed artifact is what makes the figures deterministic.
RESULTS_JSON = DATA_DIR / 'baseline_results.json'

# Photodiode channels, in the order the dataset feeds them to the model, with
# the laser wavelength each one detects (microns).
PD_CHANNELS: tuple[tuple[str, float], ...] = (
    ('RP1_CH2', 0.635),
    ('RP2_CH1', 0.675),
    ('RP2_CH2', 0.515),
)
VOLTAGE_CHANNEL = 'RP1_CH1'

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

# One qualitative palette for the whole document set. Chosen to stay
# distinguishable in grayscale and for the common red-green deficiencies.
COLORS: dict[str, str] = {
    'drive': '#4c4c4c',
    'ch635': '#c1442b',
    'ch675': '#d99a1c',
    'ch515': '#2f7d4f',
    'accent': '#2a5d9f',
    'muted': '#8a8a8a',
    'highlight': '#7b4ea8',
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
_RCPARAMS: dict[RcKeyType, Any] = {
    'figure.figsize': FIGSIZE_WIDE,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.color': '#dcdcdc',
    'grid.linewidth': 0.6,
    'legend.frameon': False,
    'legend.fontsize': 8,
    'lines.linewidth': 1.2,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
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
