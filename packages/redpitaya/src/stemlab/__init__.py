"""Arbitrary waveform generation and acquisition on Red Pitaya STEMlab boards.

Signal-agnostic: callers hand in sample arrays and get raw channel arrays back.
Importing the package opens no connection; boards are reached only when a
``RedPitayaManager`` is built with a transport factory.
"""

from .board import STEMLAB_125_14_SPEC, BoardSpec
from .manager import (
    Acquisition,
    AcquisitionMetadata,
    AcquisitionTrigger,
    RedPitayaManager,
    Transport,
)
from .scpi import SCPI

__all__ = [
    'SCPI',
    'STEMLAB_125_14_SPEC',
    'Acquisition',
    'AcquisitionMetadata',
    'AcquisitionTrigger',
    'BoardSpec',
    'RedPitayaManager',
    'Transport',
]
