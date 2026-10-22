"""Red Pitaya hardware interface module.

``RedPitayaManager`` is intentionally not re-exported here. ``manager`` imports
from ``smi.synthetic``, whose ``waveform`` module imports
``RedPitayaConfig`` from this package; eagerly importing ``manager`` in this
``__init__`` would close that cycle during package initialization. Import the
manager directly from its module instead::

    from smi.redpitaya.manager import RedPitayaManager
"""

from .redpitaya_config import RedPitayaConfig
from .scpi import SCPI

__all__ = ['SCPI', 'RedPitayaConfig']
