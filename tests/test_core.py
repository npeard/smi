import numpy as np
import pytest
import torch

from smi.core.acquisition import (
    NOMINAL_ACQ_DECIMATION,
    STEMLAB_125_14,
    AcquisitionConfig,
)
from smi.core.kinematics import derivative_displacement, integrate_velocity

SAMPLE_RATE = STEMLAB_125_14.sample_rate_dec1_hz / NOMINAL_ACQ_DECIMATION


def _sine_velocity() -> np.ndarray:
    # Displacement d(t) = A (1 - cos(w t)) starts at zero; velocity is um/ms.
    n, amplitude, freq = 4096, 3.0, 100.0
    t = np.arange(n) / SAMPLE_RATE
    w = 2 * np.pi * freq
    return amplitude * w * np.sin(w * t) * 1e-3


def _midpoint(velocity):
    # The cumulative sum counts v[k] over the step ending at k, so the central
    # difference of its integral returns the average of two neighbours, exactly.
    return 0.5 * (velocity[..., 1:-2] + velocity[..., 2:-1])


def test_integrate_then_differentiate_round_trips_numpy():
    velocity = _sine_velocity()
    recovered = derivative_displacement(
        integrate_velocity(velocity, SAMPLE_RATE), SAMPLE_RATE
    )
    np.testing.assert_allclose(
        recovered[1:-2], _midpoint(velocity), atol=1e-6 * np.abs(velocity).max(), rtol=0
    )
    # And it tracks the velocity itself to the half-sample shift's size.
    np.testing.assert_allclose(
        recovered[2:-2], velocity[2:-2], atol=1e-3 * np.abs(velocity).max(), rtol=0
    )


def test_integrate_then_differentiate_round_trips_torch():
    velocity = torch.from_numpy(_sine_velocity())[None].double()
    displacement = integrate_velocity(velocity, SAMPLE_RATE)
    assert isinstance(displacement, torch.Tensor)
    recovered = derivative_displacement(displacement, SAMPLE_RATE)
    assert isinstance(recovered, torch.Tensor)
    torch.testing.assert_close(
        recovered[:, 1:-2],
        _midpoint(velocity),
        atol=1e-6 * velocity.abs().max().item(),
        rtol=0,
    )


def test_acquisition_config_has_no_defaults():
    no_fields: dict[str, float] = {}  # keeps the omission from the type checker
    with pytest.raises(TypeError):
        AcquisitionConfig(**no_fields)


def test_stemlab_constants():
    assert STEMLAB_125_14.buffer_size == 16384
    assert STEMLAB_125_14.sample_rate_dec1_hz == 125e6
