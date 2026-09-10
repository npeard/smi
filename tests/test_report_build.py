"""Tests for the report/ figure-and-document build.

Scope: the figure layer and the Snakefile's registry, both of which are cheap
to check and easy to silently break. The full document build is deliberately
out of scope -- it needs the `report` pixi env (typst, snakemake), which the
test env does not have, and it is slow. Build it by hand with
``pixi run report``.

Two failure modes these tests exist to catch:

1. A figure script that no longer honors the ``--output PATH`` contract the
   Snakefile relies on to pass it a target.
2. Registry drift: the Snakefile's FIGURES list and the ``plot_*.py`` files on
   disk falling out of sync. A file-per-figure layout invites exactly this,
   and the symptom otherwise is a confusing Snakemake error at build time
   rather than a test failure.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / 'report'
FIGURE_DIR = REPORT_DIR / 'figures'
SNAKEFILE = REPORT_DIR / 'Snakefile'

FIGURE_SCRIPTS = sorted(FIGURE_DIR.glob('plot_*.py'))

# The figure scripts read the acquisition HDF5 files, which are gitignored, so
# a fresh clone cannot run them.
DATA_DIR = REPO_ROOT / 'smi' / 'analysis' / 'data'
REQUIRED_DATA = (
    DATA_DIR / 'free-space-synchro_10k.h5',
    DATA_DIR / 'mmfiber-synchro_10k.h5',
)
data_available = all(p.exists() for p in REQUIRED_DATA)
requires_data = pytest.mark.skipif(
    not data_available, reason='acquisition HDF5 files are not present'
)


def _snakefile_list(name: str) -> list[str]:
    """Read a top-level list-of-strings assignment out of the Snakefile.

    Parsing the literal rather than importing keeps this test free of a
    snakemake dependency, which the test env does not have.
    """
    source = SNAKEFILE.read_text(encoding='utf-8')
    match = re.search(rf'^{name}\s*=\s*(\[.*?\])', source, re.MULTILINE | re.DOTALL)
    if match is None:
        pytest.fail(f'No top-level `{name} = [...]` assignment in {SNAKEFILE}')
    return ast.literal_eval(match.group(1))


def _snakefile_document_figures() -> dict[str, list[str]]:
    """Read the DOCUMENT_FIGURES mapping out of the Snakefile."""
    source = SNAKEFILE.read_text(encoding='utf-8')
    match = re.search(
        r'^DOCUMENT_FIGURES\s*=\s*(\{.*?\n\})', source, re.MULTILINE | re.DOTALL
    )
    if match is None:
        pytest.fail(f'No top-level `DOCUMENT_FIGURES = {{...}}` in {SNAKEFILE}')
    return ast.literal_eval(match.group(1))


def _script_for(figure_name: str) -> Path:
    return FIGURE_DIR / f'plot_{figure_name}.py'


def test_figure_scripts_exist():
    """There is at least one figure script, so the parametrized tests run.

    Without this, an empty glob would make every parametrized test below
    silently vanish and the suite would pass while checking nothing.
    """
    assert FIGURE_SCRIPTS, f'No plot_*.py scripts found in {FIGURE_DIR}'


@pytest.mark.parametrize('script', FIGURE_SCRIPTS, ids=lambda p: p.stem)
@requires_data
def test_figure_script_writes_requested_output(script: Path, tmp_path: Path):
    """Each script writes a non-empty PDF at exactly the ``--output`` path.

    This is the contract the Snakefile depends on: it hands the script a
    target path and expects that path -- and only that path -- to appear.
    """
    output = tmp_path / 'nested' / 'requested-name.pdf'
    result = subprocess.run(
        [sys.executable, str(script), '--output', str(output)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, (
        f'{script.name} exited {result.returncode}\n'
        f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
    )
    assert output.exists(), (
        f'{script.name} did not write the requested path {output}. '
        f'Files created: {sorted(p.name for p in tmp_path.rglob("*") if p.is_file())}'
    )
    assert output.stat().st_size > 0, f'{script.name} wrote an empty file'
    assert output.read_bytes()[:4] == b'%PDF', (
        f'{script.name} wrote something that is not a PDF'
    )


@pytest.mark.parametrize('script', FIGURE_SCRIPTS, ids=lambda p: p.stem)
def test_figure_script_requires_output(script: Path):
    """A script invoked without ``--output`` fails rather than guessing a path.

    If a script silently defaulted to some path, Snakemake would mark the rule
    successful while the declared output never appeared.
    """
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode != 0, (
        f'{script.name} succeeded with no --output; it must require one'
    )
    assert '--output' in result.stderr


def test_every_registered_figure_has_a_script():
    """Every name in the Snakefile's FIGURES maps to a script on disk."""
    missing = [
        name for name in _snakefile_list('FIGURES') if not _script_for(name).exists()
    ]
    assert not missing, (
        f'Snakefile FIGURES names with no plot_<name>.py in {FIGURE_DIR}: {missing}'
    )


def test_every_script_is_registered():
    """Every ``plot_*.py`` on disk appears in the Snakefile's FIGURES.

    The other direction of the same drift: an unregistered script is dead
    weight that never gets built and never gets noticed.
    """
    registered = set(_snakefile_list('FIGURES'))
    on_disk = {script.stem.removeprefix('plot_') for script in FIGURE_SCRIPTS}
    unregistered = sorted(on_disk - registered)
    assert not unregistered, (
        f'plot_*.py scripts absent from the Snakefile FIGURES list: {unregistered}. '
        f'Add them, or delete them.'
    )


def test_document_figure_references_are_registered():
    """Figures a document declares as inputs are all in the FIGURES registry."""
    registered = set(_snakefile_list('FIGURES'))
    unknown = {
        document: sorted(set(names) - registered)
        for document, names in _snakefile_document_figures().items()
        if set(names) - registered
    }
    assert not unknown, f'DOCUMENT_FIGURES names not in FIGURES: {unknown}'


def test_documents_declare_the_figures_they_include():
    """A document's Snakefile inputs match the figures its .typ actually uses.

    A figure used by a document but not declared as its input builds by luck
    (some other document pulled it in) and breaks the moment that changes;
    a declared-but-unused figure makes the document rebuild for no reason.
    """
    document_figures = _snakefile_document_figures()
    mismatches = {}
    for document, declared in document_figures.items():
        source_path = REPORT_DIR / f'{document}.typ'
        assert source_path.exists(), f'DOCUMENT_FIGURES lists missing {source_path}'
        used = set(re.findall(r'#fig\(\s*"([^"]+)"', source_path.read_text('utf-8')))
        if used != set(declared):
            mismatches[document] = {
                'used_but_not_declared': sorted(used - set(declared)),
                'declared_but_not_used': sorted(set(declared) - used),
            }
    assert not mismatches, f'Snakefile/document figure mismatch: {mismatches}'
