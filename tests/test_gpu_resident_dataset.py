"""Tests for the opt-in GPU-resident data path.

The property worth testing is not "the resident mode exists" but that it
produces the *same batches* as the per-sample DataLoader path it replaces, and
that it refuses rather than OOMs when the data does not fit. A mode that is
faster but returns different numbers is a silent training bug.
"""

from pathlib import Path

import pytest
import torch

from smi.analysis.datamodule import VelocityDataModule, _same_device
from smi.analysis.datasets import GPUResidentDataset, VelocityDataset

DATASET = (
    Path(__file__).parent.parent
    / 'smi'
    / 'analysis'
    / 'data'
    / 'free-space-synchro_10k.h5'
)
SIGNAL_LENGTH = 16384
NUM_PD_CHANNELS = 3
MAX_SHOTS = 8

pytestmark = pytest.mark.skipif(
    not DATASET.exists(), reason=f'dataset {DATASET.name} not present'
)


def test_resident_samples_match_the_lazy_dataset():
    """Slicing resident tensors gives the same numbers as per-sample loading.

    Held on CPU so this runs in CI. The torch physics is device-independent,
    and the CUDA equivalence is covered in ``test_coil_driver_torch.py``.
    """
    lazy = VelocityDataset(DATASET, num_pd_channels=NUM_PD_CHANNELS)
    resident = GPUResidentDataset(
        DATASET, device='cpu', num_pd_channels=NUM_PD_CHANNELS, max_shots=MAX_SHOTS
    )
    assert len(resident) == MAX_SHOTS

    for idx in (0, 3, MAX_SHOTS - 1):
        lazy_signals, lazy_velocity, lazy_displacement = lazy[idx]
        signals, velocity, displacement = resident[idx]

        assert signals.shape == (NUM_PD_CHANNELS, SIGNAL_LENGTH)
        torch.testing.assert_close(signals, lazy_signals)
        # The resident path runs the physics batched in float64 and the lazy
        # path per-sample in numpy float64, so they agree to float32 rounding
        # rather than bit-for-bit.
        scale = float(lazy_velocity.abs().max())
        torch.testing.assert_close(velocity, lazy_velocity, rtol=0, atol=1e-5 * scale)
        scale_d = float(lazy_displacement.abs().max())
        torch.testing.assert_close(
            displacement, lazy_displacement, rtol=0, atol=1e-5 * scale_d
        )


def test_chunked_physics_crosses_a_chunk_boundary_cleanly():
    """Loading more than one PHYSICS_CHUNK gives the same answer as one chunk.

    Construction runs the transform in chunks to bound the complex128
    workspace. Chunking is an allocation detail, so it must not be visible in
    the numbers -- an off-by-one in the slice assignment would be.
    """
    n = GPUResidentDataset.PHYSICS_CHUNK + 3
    chunked = GPUResidentDataset(
        DATASET, device='cpu', num_pd_channels=NUM_PD_CHANNELS, max_shots=n
    )
    assert len(chunked) == n

    lazy = VelocityDataset(DATASET, num_pd_channels=NUM_PD_CHANNELS)
    # One index inside the first chunk, one straddling the boundary, one after.
    for idx in (
        0,
        GPUResidentDataset.PHYSICS_CHUNK - 1,
        GPUResidentDataset.PHYSICS_CHUNK,
        n - 1,
    ):
        _lazy_signals, lazy_velocity, lazy_displacement = lazy[idx]
        _signals, velocity, displacement = chunked[idx]
        scale = float(lazy_velocity.abs().max())
        torch.testing.assert_close(velocity, lazy_velocity, rtol=0, atol=1e-5 * scale)
        scale_d = float(lazy_displacement.abs().max())
        torch.testing.assert_close(
            displacement, lazy_displacement, rtol=0, atol=1e-5 * scale_d
        )


def test_required_bytes_matches_what_is_allocated():
    """The pre-flight size estimate is the size actually held."""
    expected = GPUResidentDataset.required_bytes(
        DATASET, num_pd_channels=NUM_PD_CHANNELS, max_shots=MAX_SHOTS
    )
    resident = GPUResidentDataset(
        DATASET, device='cpu', num_pd_channels=NUM_PD_CHANNELS, max_shots=MAX_SHOTS
    )
    assert resident.resident_bytes() == expected
    # 3 photodiode channels + velocity + displacement, float32.
    assert expected == 4 * MAX_SHOTS * SIGNAL_LENGTH * (NUM_PD_CHANNELS + 2)


def test_full_file_working_set_is_reported_honestly():
    """The whole 10k file needs 3.28 GB resident, not the 1 GB on-disk size."""
    total = GPUResidentDataset.required_bytes(DATASET, num_pd_channels=3)
    assert 3.2e9 < total < 3.4e9


def test_datamodule_defaults_to_the_dataloader_path():
    """No preload_device means nothing about existing behaviour changes."""
    dm = VelocityDataModule(
        dataset_path=str(DATASET), batch_size=2, num_workers=0, num_pd_channels=3
    )
    dm.setup()
    assert dm.preload_device is None
    assert dm.preloaded is False
    assert isinstance(dm.train_dataset.dataset, VelocityDataset)


def test_datamodule_falls_back_when_the_data_does_not_fit():
    """An impossible headroom demand degrades to the DataLoader, not an OOM."""
    dm = VelocityDataModule(
        dataset_path=str(DATASET),
        batch_size=2,
        num_workers=0,
        preload_device='cuda:0' if torch.cuda.is_available() else 'cpu',
        # Larger than any GPU on the planet, so the fit check must fail.
        preload_headroom_bytes=1 << 50,
        num_pd_channels=3,
    )
    if not torch.cuda.is_available():
        pytest.skip('fallback is only triggered for CUDA devices')
    dm.setup()
    assert dm.preloaded is False
    assert isinstance(dm.train_dataset.dataset, VelocityDataset)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_device_mismatch_warns_instead_of_silently_copying(caplog):
    """Preloading to a device the trainer is not on must not pass unremarked.

    It still works, but every batch pays a copy, which makes the resident mode
    slower than the DataLoader it replaced -- with no visible symptom.
    """
    dm = VelocityDataModule(
        dataset_path=str(DATASET),
        batch_size=2,
        num_workers=0,
        preload_device='cuda:0',
        num_pd_channels=3,
    )
    dm.setup()
    if not dm.preloaded:
        pytest.skip('GPU does not have room for the dataset plus headroom')

    batch = next(iter(dm.train_dataloader()))
    with caplog.at_level('WARNING'):
        moved = dm.transfer_batch_to_device(batch, torch.device('cpu'), 0)
    assert 'being copied between devices' in caplog.text
    assert moved[0].device.type == 'cpu'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_datamodule_preload_serves_device_resident_batches():
    """With preload_device set, batches arrive already on the device."""
    dm = VelocityDataModule(
        dataset_path=str(DATASET),
        batch_size=2,
        num_workers=4,
        preload_device='cuda:0',
        num_pd_channels=3,
    )
    dm.setup()
    if not dm.preloaded:
        pytest.skip('GPU does not have room for the dataset plus headroom')

    signals, velocity, displacement = next(iter(dm.train_dataloader()))
    assert signals.device.type == 'cuda'
    assert velocity.device.type == 'cuda'
    assert displacement.device.type == 'cuda'
    assert signals.shape == (2, 3, SIGNAL_LENGTH)
    # num_workers is forced to 0: workers cannot fork CUDA tensors.
    assert dm.train_dataloader().num_workers == 0


def test_same_device_treats_bare_cuda_as_the_current_device():
    """A Trainer on 'cuda' must not be reported as a device mismatch.

    torch.device('cuda') != torch.device('cuda:0') even though tensors sent
    to the former land on the latter. Comparing them naively fires the
    "every batch is being copied" warning on every batch of a correctly
    configured run, which is the false alarm the warning exists to avoid.
    """
    assert _same_device(torch.device('cuda'), torch.device('cuda:0'))
    assert _same_device(torch.device('cuda:0'), torch.device('cuda'))
    assert _same_device(torch.device('cpu'), torch.device('cpu'))
    assert not _same_device(torch.device('cpu'), torch.device('cuda:0'))
    assert not _same_device(torch.device('cuda:0'), torch.device('cuda:1'))


def test_resident_path_rejects_kwargs_it_would_otherwise_drop():
    """An unsupported dataset kwarg must raise, not be silently ignored.

    The two dataset classes take different arguments, so **dataset_kwargs
    cannot simply be forwarded. Dropping the difference would let a caller
    pass an argument that changes nothing, while the same call on the
    DataLoader path raises -- the two paths must agree about what was asked.
    """
    dm = VelocityDataModule(
        dataset_path=str(DATASET), preload_device='cpu', not_a_real_dataset_kwarg=1
    )
    with pytest.raises(TypeError, match='not_a_real_dataset_kwarg'):
        dm._build_full_dataset()
