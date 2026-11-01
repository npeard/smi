"""Red Pitaya hardware interface module.

``RedPitayaManager`` is intentionally not re-exported here: ``manager`` pulls
in ``smi.synthetic`` and matplotlib, which importing ``SCPI`` should not cost.
Import the manager directly from its module instead::

    from smi.redpitaya.manager import RedPitayaManager
"""

from .scpi import SCPI

__all__ = ['SCPI']
