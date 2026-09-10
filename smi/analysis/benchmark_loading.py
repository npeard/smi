#!/usr/bin/env python3
"""End-to-end training-throughput benchmark for the data-loading path.

Answers one question: has training throughput actually been bottlenecked by the
DataLoader, or by the model? There are three candidate regimes and they need
different fixes, so the benchmark must be able to tell them apart:

1. Disk/IO bound  -- the HDF5 read dominates.
2. CPU-compute bound -- the per-sample FFT physics in ``VelocityDataset``
   dominates (it computed the displacement spectrum twice per sample).
3. GPU bound -- the model dominates and loading is already hidden behind it.

Every configuration is timed against a *real* forward + backward pass of a real
architecture at a realistic size. A loader benchmark whose baseline does not do
production's job cannot distinguish regime 3 from the others, and regime 3 is
the answer that would make all the loader work pointless.

Configurations (see ``CONFIG_NAMES``):

- ``dataloader-dup-w{0,4,8}``  -- DataLoader over the original per-sample path,
  which computes ``get_displacement_spectrum`` twice per sample.
- ``dataloader-w{0,4,8}``      -- DataLoader over the de-duplicated path
  (``get_velocity_and_displacement``), one spectrum per sample.
- ``gpu-resident``             -- inputs and both targets precomputed and held
  on the GPU; batches are slices of a shuffled index. No DataLoader.
- ``gpu-resident-torch-physics`` -- only the raw channels are held on the GPU;
  velocity/displacement are recomputed on-device each epoch with
  ``get_velocity_and_displacement_torch``. Trades compute for ~1.3 GB of VRAM.

Two isolation runs bracket those, so the regime can be named rather than
inferred:

- ``loader-only-*``  -- the loader with no model: its supply ceiling.
- ``model-only``     -- the model on synthetic in-VRAM tensors with no data
  path at all: the GPU ceiling.

If the end-to-end numbers sit at ``model-only`` while ``loader-only`` is far
above them, the workload is GPU bound and no loader change can help.

Timing rules: warm up before measuring, ``torch.cuda.synchronize()`` around any
GPU timing, report the median of ``--repeats`` (>= 3) epochs, and pin to one
explicit device. Peak GPU memory is reported for the resident modes.

Caveat for this machine: the 3090 Ti is also the display GPU, so the desktop
compositor holds a few GB and takes a variable slice of the SMs. The absolute
samples/s therefore shifts between sessions (212 and 393 have both been
measured for ``model-only``). What is stable, and what the conclusion rests
on, is the *ratio* within a single run -- so compare configurations only
against others from the same invocation, never across runs.

The ``w0`` configurations are the noisy ones: with no worker processes the
loader runs inline on the training thread, so it competes with the same
Python process that is driving the GPU and picks up whatever else the desktop
is doing. Run-to-run spread there has been ~20%, against under 1.5% for
every ``w4``/``w8`` configuration. They are kept because they show what the
duplicate-FFT fix was worth when the loader is on the critical path, but no
conclusion should rest on a ``w0`` number.

Run with::

    pixi run -e dev python -m smi.analysis.benchmark_loading --shots 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from smi.analysis.datasets import VelocityDataset
from smi.analysis.models.tcn import TCN, TCNConfig
from smi.synthetic.coil_driver import CoilDriver

logger = logging.getLogger(__name__)

DEFAULT_DATA_FILE = (
    Path(__file__).resolve().parent / 'data' / 'free-space-synchro_10k.h5'
)
PD_CHANNEL_KEYS = ('RP1_CH2', 'RP2_CH1', 'RP2_CH2')
VOLTAGE_KEY = 'RP1_CH1'
DEFAULT_SAMPLE_RATE = 488281.25

CONFIG_NAMES = (
    'dataloader-dup-w0',
    'dataloader-dup-w4',
    'dataloader-dup-w8',
    'dataloader-w0',
    'dataloader-w4',
    'dataloader-w8',
    'gpu-resident',
    'gpu-resident-torch-physics',
    # Isolation runs. These are not candidate training paths; they bracket the
    # end-to-end numbers so the regime can be named rather than guessed. If
    # 'model-only' is close to the end-to-end configurations and
    # 'loader-only-*' is far above them, the workload is GPU bound.
    'loader-only-dup-w0',
    'loader-only-dup-w8',
    'loader-only-w0',
    'loader-only-w8',
    'model-only',
)


# --------------------------------------------------------------------------
# Model under test
# --------------------------------------------------------------------------


def build_model(sequence_length: int) -> nn.Module:
    """Build the TCN at the size used in ``models/configs/tcn-config.yaml``.

    Using the checked-in production configuration (6 dilated temporal blocks,
    16..64 channels, kernel 5) rather than a toy net is what makes the GPU-bound
    regime detectable.
    """
    config = TCNConfig(
        sequence_length=sequence_length,
        in_channels=3,
        activation='GELU',
        use_layer_norm=True,
        use_weight_norm=True,
        kernel_size=5,
        temporal_channels=[16, 16, 32, 32, 64, 64],
        dilation_base=2,
        dropout=0.0,
    )
    return TCN(config)


# --------------------------------------------------------------------------
# Dataset variants
# --------------------------------------------------------------------------


class DuplicateSpectrumDataset(VelocityDataset):
    """The pre-fix ``__getitem__``, kept so the fix can be measured.

    ``VelocityDataset`` now shares one displacement spectrum between velocity
    and displacement. This subclass restores the original two-call behaviour so
    the benchmark can quantify what that duplication actually cost, instead of
    asserting a speedup that was never measured.
    """

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load one sample, computing the displacement spectrum twice."""
        self.open_hdf5()
        pd_signals = [self.pd_data[i][idx] for i in range(len(self.pd_data))]
        voltage = self.voltage_data[idx]
        velocity, _, _ = self.coil_driver.get_velocity(voltage, self.sample_rate)
        displacement, _, _ = self.coil_driver.get_displacement(
            voltage, self.sample_rate
        )
        signals = np.stack(pd_signals)
        return (
            torch.FloatTensor(signals),
            torch.FloatTensor(velocity),
            torch.FloatTensor(displacement),
        )


class _SubsetDataset(Dataset):
    """First ``n`` samples of ``base``, so a run can be shortened honestly."""

    def __init__(self, base: Dataset, n: int) -> None:
        self.base = base
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        return self.base[idx]


# --------------------------------------------------------------------------
# Result record
# --------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """One configuration's measurement."""

    name: str
    shots: int
    batch_size: int
    epoch_times_s: list[float] = field(default_factory=list)
    peak_gpu_bytes: int | None = None
    resident_gpu_bytes: int | None = None
    setup_s: float | None = None
    error: str | None = None

    @property
    def median_epoch_s(self) -> float | None:
        """Median wall-clock seconds per epoch."""
        if not self.epoch_times_s:
            return None
        return statistics.median(self.epoch_times_s)

    @property
    def samples_per_s(self) -> float | None:
        """Median throughput in samples/s."""
        median = self.median_epoch_s
        if median is None or median <= 0:
            return None
        return self.shots / median

    def as_dict(self) -> dict[str, object]:
        """JSON-serializable view."""
        return {
            'name': self.name,
            'shots': self.shots,
            'batch_size': self.batch_size,
            'epoch_times_s': self.epoch_times_s,
            'median_epoch_s': self.median_epoch_s,
            'samples_per_s': self.samples_per_s,
            'peak_gpu_bytes': self.peak_gpu_bytes,
            'resident_gpu_bytes': self.resident_gpu_bytes,
            'setup_s': self.setup_s,
            'error': self.error,
        }


# --------------------------------------------------------------------------
# The training step, shared by every configuration
# --------------------------------------------------------------------------


def _train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    signals: torch.Tensor,
    velocity: torch.Tensor,
) -> None:
    """One forward + backward + optimizer step, identical across configs."""
    optimizer.zero_grad(set_to_none=True)
    prediction = model(signals)
    loss = nn.functional.mse_loss(prediction.squeeze(1), velocity)
    loss.backward()
    optimizer.step()


# --------------------------------------------------------------------------
# Configuration runners
# --------------------------------------------------------------------------


def _run_dataloader_config(
    name: str,
    dataset_cls: type[VelocityDataset],
    num_workers: int,
    *,
    file_path: Path,
    shots: int,
    batch_size: int,
    repeats: int,
    device: torch.device,
    sequence_length: int,
) -> BenchmarkResult:
    """Time an epoch served by a torch DataLoader."""
    result = BenchmarkResult(name=name, shots=shots, batch_size=batch_size)
    setup_start = time.perf_counter()
    base = dataset_cls(file_path, num_pd_channels=3, cache_size=0)
    dataset = _SubsetDataset(base, shots)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    model = build_model(sequence_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    result.setup_s = time.perf_counter() - setup_start

    def one_epoch() -> float:
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for cpu_signals, cpu_velocity, _displacement in loader:
            signals = cpu_signals.to(device, non_blocking=True)
            velocity = cpu_velocity.to(device, non_blocking=True)
            _train_step(model, optimizer, signals, velocity)
        torch.cuda.synchronize(device)
        return time.perf_counter() - start

    one_epoch()  # warm up: page cache, worker spawn, cuDNN autotune
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(repeats):
        result.epoch_times_s.append(one_epoch())
    result.peak_gpu_bytes = torch.cuda.max_memory_allocated(device)

    return result


def _load_raw_channels(
    file_path: Path, shots: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Read the photodiode channels and drive voltage for the first ``shots``."""
    with h5py.File(file_path, 'r') as f:
        sample_rate = float(f.attrs.get('sample_rate', DEFAULT_SAMPLE_RATE))
        signals = np.stack(
            [np.asarray(f[key][:shots], dtype=np.float32) for key in PD_CHANNEL_KEYS],
            axis=1,
        )
        voltage = np.asarray(f[VOLTAGE_KEY][:shots], dtype=np.float32)
    return signals, voltage, sample_rate


def _free_gpu_bytes(device: torch.device) -> int:
    """Currently free memory on ``device``, in bytes."""
    free, _total = torch.cuda.mem_get_info(device)
    return int(free)


def _run_gpu_resident(
    name: str,
    *,
    file_path: Path,
    shots: int,
    batch_size: int,
    repeats: int,
    device: torch.device,
    sequence_length: int,
    torch_physics: bool,
) -> BenchmarkResult:
    """Time an epoch served by slicing GPU-resident tensors.

    With ``torch_physics=False`` the velocity and displacement targets are
    precomputed once on the CPU and parked on the GPU alongside the inputs.
    With ``torch_physics=True`` only the raw drive voltage is resident and the
    targets are recomputed on-device per batch, which costs GPU time but saves
    the two target tensors' worth of VRAM.
    """
    result = BenchmarkResult(name=name, shots=shots, batch_size=batch_size)
    setup_start = time.perf_counter()
    signals_np, voltage_np, sample_rate = _load_raw_channels(file_path, shots)
    driver = CoilDriver()

    # Refuse rather than OOM: report the shortfall as the measurement.
    bytes_per_shot = 4 * sequence_length * (3 + (1 if torch_physics else 2))
    needed = bytes_per_shot * shots
    free = _free_gpu_bytes(device)
    headroom = 3 << 30  # leave room for the model, activations and workspace
    if needed + headroom > free:
        result.error = (
            f'infeasible: needs {needed / 1e9:.2f} GB resident plus '
            f'~{headroom / 1e9:.1f} GB working headroom, but only '
            f'{free / 1e9:.2f} GB is free on {torch.cuda.get_device_name(device)}'
        )
        return result

    torch.cuda.reset_peak_memory_stats(device)
    signals_gpu = torch.from_numpy(signals_np).to(device)
    if torch_physics:
        voltage_gpu = torch.from_numpy(voltage_np).to(device)
        velocity_gpu = None
    else:
        velocity_np = np.empty_like(voltage_np)
        for i in range(shots):
            velocity_np[i], _ = driver.get_velocity_and_displacement(
                voltage_np[i], sample_rate
            )
        voltage_gpu = None
        velocity_gpu = torch.from_numpy(velocity_np).to(device)
    torch.cuda.synchronize(device)
    result.resident_gpu_bytes = torch.cuda.memory_allocated(device)

    model = build_model(sequence_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator(device=device).manual_seed(0)
    result.setup_s = time.perf_counter() - setup_start

    def one_epoch() -> float:
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        order = torch.randperm(shots, device=device, generator=generator)
        for begin in range(0, shots, batch_size):
            idx = order[begin : begin + batch_size]
            signals = signals_gpu[idx]
            if torch_physics:
                velocity, _displacement = driver.get_velocity_and_displacement_torch(
                    voltage_gpu[idx], sample_rate
                )
            else:
                velocity = velocity_gpu[idx]
            _train_step(model, optimizer, signals, velocity)
        torch.cuda.synchronize(device)
        return time.perf_counter() - start

    one_epoch()
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(repeats):
        result.epoch_times_s.append(one_epoch())
    result.peak_gpu_bytes = torch.cuda.max_memory_allocated(device)

    # The GPU tensors and the loader's worker processes are released when this
    # frame goes away; the caller calls empty_cache() between configurations.
    return result


def _run_loader_only(
    name: str,
    dataset_cls: type[VelocityDataset],
    num_workers: int,
    *,
    device: torch.device,
    file_path: Path,
    shots: int,
    batch_size: int,
    repeats: int,
) -> BenchmarkResult:
    """Time the loader alone, with no model, to find its supply ceiling.

    This is the upper bound on what the loader can deliver. Compared against
    the end-to-end numbers it says whether the loader has headroom.

    The host-to-device copy is included even though no model consumes the
    result. Supplying a batch means supplying it *on the device* -- that copy
    is work the production path pays every iteration, and leaving it out
    would measure the delivery of pinned host tensors and call it the
    loader's ceiling, overstating the headroom against configurations that
    do pay it.
    """
    result = BenchmarkResult(name=name, shots=shots, batch_size=batch_size)
    setup_start = time.perf_counter()
    base = dataset_cls(file_path, num_pd_channels=3, cache_size=0)
    loader = DataLoader(
        _SubsetDataset(base, shots),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    result.setup_s = time.perf_counter() - setup_start

    def one_epoch() -> float:
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for cpu_signals, cpu_velocity, _displacement in loader:
            cpu_signals.to(device, non_blocking=True)
            cpu_velocity.to(device, non_blocking=True)
        torch.cuda.synchronize(device)
        return time.perf_counter() - start

    one_epoch()
    for _ in range(repeats):
        result.epoch_times_s.append(one_epoch())
    return result


def _run_model_only(
    name: str,
    *,
    shots: int,
    batch_size: int,
    repeats: int,
    device: torch.device,
    sequence_length: int,
) -> BenchmarkResult:
    """Time the model alone on synthetic in-VRAM tensors: the GPU ceiling.

    No data path of any kind is involved, so this is the fastest an epoch of
    ``shots`` samples can possibly be. Any end-to-end configuration close to
    this number is GPU bound and cannot be improved by touching the loader.
    """
    result = BenchmarkResult(name=name, shots=shots, batch_size=batch_size)
    setup_start = time.perf_counter()
    model = build_model(sequence_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    signals = torch.randn(batch_size, 3, sequence_length, device=device)
    velocity = torch.randn(batch_size, sequence_length, device=device)
    n_batches = -(-shots // batch_size)
    result.setup_s = time.perf_counter() - setup_start

    def one_epoch() -> float:
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(n_batches):
            _train_step(model, optimizer, signals, velocity)
        torch.cuda.synchronize(device)
        return time.perf_counter() - start

    one_epoch()
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(repeats):
        result.epoch_times_s.append(one_epoch())
    result.peak_gpu_bytes = torch.cuda.max_memory_allocated(device)
    return result


def run_configuration(
    name: str,
    *,
    file_path: Path,
    shots: int,
    batch_size: int,
    repeats: int,
    device: torch.device,
    sequence_length: int,
) -> BenchmarkResult:
    """Dispatch one named configuration."""
    if name == 'model-only':
        return _run_model_only(
            name,
            shots=shots,
            batch_size=batch_size,
            repeats=repeats,
            device=device,
            sequence_length=sequence_length,
        )
    if name.startswith('loader-only'):
        return _run_loader_only(
            name,
            DuplicateSpectrumDataset if '-dup-' in name else VelocityDataset,
            int(name.rsplit('w', 1)[1]),
            device=device,
            file_path=file_path,
            shots=shots,
            batch_size=batch_size,
            repeats=repeats,
        )
    if name.startswith('dataloader'):
        num_workers = int(name.rsplit('w', 1)[1])
        dataset_cls = DuplicateSpectrumDataset if '-dup-' in name else VelocityDataset
        return _run_dataloader_config(
            name,
            dataset_cls,
            num_workers,
            file_path=file_path,
            shots=shots,
            batch_size=batch_size,
            repeats=repeats,
            device=device,
            sequence_length=sequence_length,
        )
    if name.startswith('gpu-resident'):
        return _run_gpu_resident(
            name,
            file_path=file_path,
            shots=shots,
            batch_size=batch_size,
            repeats=repeats,
            device=device,
            sequence_length=sequence_length,
            torch_physics=name.endswith('torch-physics'),
        )
    raise ValueError(f'Unknown configuration: {name}')


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def format_table(results: list[BenchmarkResult]) -> str:
    """Render the results as a fixed-width table."""
    header = (
        f'{"configuration":<30} {"samples/s":>10} {"epoch s":>9} '
        f'{"peak GPU GB":>12} {"resident GB":>12}'
    )
    lines = [header, '-' * len(header)]
    for r in results:
        if r.error is not None:
            lines.append(f'{r.name:<30} {r.error}')
            continue
        peak = '' if r.peak_gpu_bytes is None else f'{r.peak_gpu_bytes / 1e9:.2f}'
        resident = (
            '' if r.resident_gpu_bytes is None else f'{r.resident_gpu_bytes / 1e9:.2f}'
        )
        lines.append(
            f'{r.name:<30} {r.samples_per_s:>10.1f} {r.median_epoch_s:>9.2f} '
            f'{peak:>12} {resident:>12}'
        )
    return '\n'.join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data-file', type=Path, default=DEFAULT_DATA_FILE, help='HDF5 dataset.'
    )
    parser.add_argument(
        '--shots',
        type=int,
        default=2000,
        help='Number of shots per epoch. State this N when reporting.',
    )
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument(
        '--repeats', type=int, default=3, help='Timed epochs per configuration (>=3).'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Explicit device. Pin to the 24 GB card, not whichever is first.',
    )
    parser.add_argument(
        '--require-device-name',
        type=str,
        default='3090',
        help=(
            'Substring the chosen device name must contain, so a run cannot '
            'silently land on the wrong card. torch and nvidia-smi do not '
            'agree on GPU ordering on this machine. Pass an empty string to '
            'skip the check.'
        ),
    )
    parser.add_argument(
        '--configs',
        nargs='+',
        default=list(CONFIG_NAMES),
        choices=list(CONFIG_NAMES),
        help='Subset of configurations to run.',
    )
    parser.add_argument('--json-out', type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and print the throughput table."""
    logging.basicConfig(level=logging.WARNING)
    args = _parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark measures GPU training throughput.')
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    # torch orders CUDA devices by compute capability by default, while
    # nvidia-smi orders them by PCI bus id -- on this machine those disagree,
    # so 'cuda:0' and 'GPU 0' are different cards. Fail loudly rather than
    # publish numbers from the 8 GB Quadro thinking they came from the 3090 Ti.
    device_name = torch.cuda.get_device_name(device)
    if args.require_device_name and args.require_device_name not in device_name:
        raise RuntimeError(
            f'{args.device} is "{device_name}", which does not contain '
            f'"{args.require_device_name}". Pick the right --device or pass '
            f'--require-device-name "" to override.'
        )

    # Another process sharing the GPU makes every number below meaningless.
    other = torch.cuda.memory_reserved(device)
    free, total = torch.cuda.mem_get_info(device)
    in_use = total - free - other
    if in_use > (1 << 30):
        logger.warning(
            'Another process is holding %.2f GB on %s. Throughput numbers '
            'from a shared GPU are not comparable; wait for it to be idle.',
            in_use / 1e9,
            device_name,
        )

    with h5py.File(args.data_file, 'r') as f:
        sequence_length = int(f[VOLTAGE_KEY].shape[1])
        available = int(f[VOLTAGE_KEY].shape[0])
    shots = min(args.shots, available)

    print(f'device        : {device_name} ({args.device})')  # noqa: T201
    print(f'data file     : {args.data_file}')  # noqa: T201
    print(f'shots (N)     : {shots} of {available}')  # noqa: T201
    print(f'sequence len  : {sequence_length}')  # noqa: T201
    print(f'batch size    : {args.batch_size}')  # noqa: T201
    print(f'timed epochs  : {args.repeats} (median reported), 1 warmup epoch')  # noqa: T201
    print()  # noqa: T201

    results: list[BenchmarkResult] = []
    for name in args.configs:
        print(f'running {name} ...', flush=True)  # noqa: T201
        result = run_configuration(
            name,
            file_path=args.data_file,
            shots=shots,
            batch_size=args.batch_size,
            repeats=args.repeats,
            device=device,
            sequence_length=sequence_length,
        )
        results.append(result)
        torch.cuda.empty_cache()

    print()  # noqa: T201
    print(format_table(results))  # noqa: T201

    if args.json_out is not None:
        payload = {
            'device': torch.cuda.get_device_name(device),
            'data_file': str(args.data_file),
            'shots': shots,
            'sequence_length': sequence_length,
            'batch_size': args.batch_size,
            'repeats': args.repeats,
            'results': [r.as_dict() for r in results],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f'\nwrote {args.json_out}')  # noqa: T201

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
