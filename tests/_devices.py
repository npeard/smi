"""Devices a device-parametrized test runs on."""

import pytest
import torch


def devices() -> list:
    """Cpu always; mps where present, reported as skipped elsewhere."""
    return [
        'cpu',
        pytest.param(
            'mps',
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason='requires MPS'
            ),
        ),
    ]
