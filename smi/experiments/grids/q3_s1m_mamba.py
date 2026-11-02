"""Q2 stage S1m: Mamba, deferred from S1 until its scan was parallel.

Same cells and budget as q3_s1_families; its rows join the S1 table, noted as
run on the chunked scan and the Python 3.14 environment.
Run: pixi run experiments q3_s1m_mamba
"""

from dataclasses import replace

from smi.experiments.grid import Family
from smi.experiments.grids.q3_s1_families import GRID as S1

GRID = replace(
    S1, families=(Family('mamba-44k-T90', 'matched/mamba-44k'),), oracle=False
)
