"""Numerical equivalence of the batched torch coil-driver physics.

The torch port exists so the drive-voltage -> velocity/displacement transform
can run on-device over a whole batch. That is only useful if it computes the
same thing as the numpy path, so that is what these tests check: same input,
same answer to float32 tolerance, over a non-trivial batch.
"""

import numpy as np
import pytest
import torch

from smi.synthetic.coil_driver import CoilDriver

SAMPLE_RATE = 488281.25
SIGNAL_LENGTH = 4096
BATCH_SIZE = 16


def _make_batch(seed: int = 20260910) -> np.ndarray:
    """Build a batch of band-limited drive-voltage-like waveforms."""
    rng = np.random.default_rng(seed)
    t = np.arange(SIGNAL_LENGTH) / SAMPLE_RATE
    waveforms = np.zeros((BATCH_SIZE, SIGNAL_LENGTH))
    for row in range(BATCH_SIZE):
        # A handful of tones spanning the coil resonance (f0 ~ 259 Hz), plus
        # broadband noise, so the transfer function is exercised on and off
        # resonance rather than at a single benign frequency.
        for freq in rng.uniform(50.0, 2000.0, size=6):
            waveforms[row] += rng.uniform(0.1, 1.0) * np.sin(
                2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi)
            )
        waveforms[row] += 0.05 * rng.standard_normal(SIGNAL_LENGTH)
    return waveforms


def _numpy_reference(
    driver: CoilDriver, batch: np.ndarray, max_freq: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row velocity/displacement via the original numpy methods."""
    velocity = np.stack(
        [driver.get_velocity(row, SAMPLE_RATE, max_freq)[0] for row in batch]
    )
    displacement = np.stack(
        [driver.get_displacement(row, SAMPLE_RATE, max_freq)[0] for row in batch]
    )
    return velocity, displacement


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    """Max absolute error scaled by the reference signal's RMS."""
    scale = float(np.sqrt(np.mean(expected**2)))
    return float(np.max(np.abs(actual - expected)) / scale)


@pytest.mark.parametrize('max_freq', [None, 5000.0])
def test_torch_batch_matches_numpy_on_cpu(max_freq):
    """Batched torch physics reproduces the per-row numpy result."""
    driver = CoilDriver()
    batch = _make_batch()
    expected_v, expected_d = _numpy_reference(driver, batch, max_freq)

    tensor = torch.from_numpy(batch.astype(np.float32))
    velocity, displacement = driver.get_velocity_and_displacement_torch(
        tensor, SAMPLE_RATE, max_freq
    )

    assert velocity.shape == tensor.shape
    assert velocity.dtype == torch.float32

    # float32 round-trip through a 4096-point FFT: relative error is set by
    # float32 epsilon (~1.2e-7) times the dynamic range of the transform.
    assert _relative_error(velocity.numpy(), expected_v) < 1e-5
    assert _relative_error(displacement.numpy(), expected_d) < 1e-5


def test_torch_1d_input_matches_numpy():
    """The port accepts an unbatched (time,) vector as well."""
    driver = CoilDriver()
    row = _make_batch()[0]
    expected_v, _, _ = driver.get_velocity(row, SAMPLE_RATE)
    expected_d, _, _ = driver.get_displacement(row, SAMPLE_RATE)

    velocity, displacement = driver.get_velocity_and_displacement_torch(
        torch.from_numpy(row.astype(np.float32)), SAMPLE_RATE
    )

    assert velocity.shape == (SIGNAL_LENGTH,)
    assert _relative_error(velocity.numpy(), expected_v) < 1e-5
    assert _relative_error(displacement.numpy(), expected_d) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_torch_batch_on_gpu_matches_numpy():
    """The on-device path agrees with numpy-on-CPU, which is the whole point."""
    driver = CoilDriver()
    batch = _make_batch()
    expected_v, expected_d = _numpy_reference(driver, batch)

    tensor = torch.from_numpy(batch.astype(np.float32)).cuda()
    velocity, displacement = driver.get_velocity_and_displacement_torch(
        tensor, SAMPLE_RATE
    )

    assert velocity.device.type == 'cuda'
    assert _relative_error(velocity.cpu().numpy(), expected_v) < 1e-5
    assert _relative_error(displacement.cpu().numpy(), expected_d) < 1e-5


def test_shared_spectrum_matches_separate_calls():
    """The de-duplicated numpy helper is bit-for-bit the two separate calls.

    This is the fix for the dataset computing the same FFT twice per sample,
    so it must not change any number the pipeline already produces.
    """
    driver = CoilDriver()
    row = _make_batch()[0]
    expected_v, _, _ = driver.get_velocity(row, SAMPLE_RATE)
    expected_d, _, _ = driver.get_displacement(row, SAMPLE_RATE)

    velocity, displacement = driver.get_velocity_and_displacement(row, SAMPLE_RATE)

    np.testing.assert_array_equal(velocity, expected_v)
    np.testing.assert_array_equal(displacement, expected_d)
