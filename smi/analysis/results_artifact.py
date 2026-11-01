"""Read and write the learned-model results artifact.

Mirrors the shape of ``smi/analysis/data/baseline_results.json``: a
``schema_version`` at the top and a ``datasets`` mapping keyed by dataset name,
so a figure or a report table can index the analytic and the learned artifact
identically.
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

#: Bump when the on-disk layout changes incompatibly.
SCHEMA_VERSION = 1


@dataclass
class RunRecord:
    """One trained model evaluated on one dataset.

    ``total_params``, ``receptive_field`` and ``causal`` are required, not
    optional: a causal and a non-causal model at equal parameter count are not
    solving the same problem, and a comparison table that cannot express the
    difference silently implies they are.

    Attributes:
        model_type: Architecture name as passed to ``create_model``.
        config: The model hyperparameters the run used.
        total_params: Trainable parameter count.
        receptive_field: Receptive field in samples; ``None`` when the model
            has no finite lookback (recurrent state, whole-shot estimators).
        causal: Whether the model uses only samples at or before the
            predicted one.
        seed: Random seed the run was trained with; records from
            :mod:`smi.analysis.evaluate` use ``-1`` when the estimator has none.
        device: Device name, e.g. the CUDA device string.
        dataset: Dataset key, matching baseline_results.json's ('free-space'
            or 'mmfiber').
        metrics: Metric name -> summary, each summary using the six keys from
            :func:`smi.analysis.metrics.summarize_per_shot`.
        estimator: Estimator name when the record comes from
            :mod:`smi.analysis.evaluate` (``model_type`` then names the
            architecture or ``inverse_fit``); absent from older records.
        wavelengths_um: Channel subset, in the order the estimator saw it.
        precision: Numeric precision of the evaluation, e.g. ``'32-true'``.
        converged: Whether training converged, as reported by the caller; the
            evaluator cannot know it from a checkpoint.
        train_seed: Seed the evaluated model was trained with; ``None`` for a
            classical estimator.
        test_seed: Seed of the evaluated frozen set; ``None`` for a file that
            records none (frozen sets always do).
        finite_predictions: Whether every prediction was finite.
        sim_config_hash: ``SimConfig.hash()`` of the evaluated frozen set.
        simulator_version: ``SIMULATOR_VERSION`` that wrote that set; the
            hash covers config values only, not simulator code.
        binned: Per-sample errors binned by ``|v|`` quantile and by distance to
            the nearest fringe extremum, keyed by binning name.
        cell_key: Experiment-runner cell identifier; its presence in the
            artifact is what makes a grid resumable.
        sim_name: Name of the runner's sim entry, with any swept value.
        wall_clock_s: Seconds the runner spent on the cell (training plus
            evaluation), for sizing a full grid from a pilot.
        epochs_run: Epochs trained; with ``best_monitor`` and ``final_monitor``
            (the monitored validation value of the best and of the last
            epoch) it shows how far past its best a converged run went.
        zero_predictor_monitor: The monitored validation value for a zero
            prediction; a ``best_monitor`` not below it by ``min_rel_delta``
            marks a run stalled at the zero-output plateau, which is recorded
            not converged, as distinct from one that hit ``max_epochs``.
        train_crop_length: Training window length (samples) of the cell.
        steps_per_epoch: Training batches per epoch of the cell.
        batch_size: Training batch size of the cell.
        max_epochs: Epoch cap of the cell.
        T_0: First cosine-restart period (epochs) of the cell's scheduler.
        lr: Peak learning rate of the cell; with the five above, the training
            budget behind the cell, so cells are comparable and a resumed grid
            can tell a recorded cell from a changed one.
        run_fingerprint: Hash of the cell's whole resolved training config,
            test set and ``SIMULATOR_VERSION``; resume skips a cell only when
            it matches. ``fingerprint_blocks`` holds the per-block hashes so a
            mismatch names the changed blocks.

    ``sees_future`` is a property, so it cannot disagree with ``causal``; a
    receptive field is unbounded exactly when it is ``None``.
    """

    model_type: str
    config: dict[str, Any]
    total_params: int
    receptive_field: int | None
    causal: bool
    seed: int
    device: str
    dataset: str
    metrics: dict[str, dict[str, float]]
    # Optional so learned_results.json (schema 1, without them) still reads.
    estimator: str | None = None
    wavelengths_um: list[float] | None = None
    precision: str | None = None
    converged: bool | None = None
    train_seed: int | None = None
    test_seed: int | None = None
    finite_predictions: bool | None = None
    sim_config_hash: str | None = None
    simulator_version: int | None = None
    binned: dict[str, list[dict[str, float]]] | None = None
    cell_key: str | None = None
    sim_name: str | None = None
    wall_clock_s: float | None = None
    epochs_run: int | None = None
    best_monitor: float | None = None
    final_monitor: float | None = None
    zero_predictor_monitor: float | None = None
    train_crop_length: int | None = None
    steps_per_epoch: int | None = None
    batch_size: int | None = None
    max_epochs: int | None = None
    T_0: int | None = None
    lr: float | None = None
    run_fingerprint: str | None = None
    fingerprint_blocks: dict[str, str] | None = None

    @property
    def sees_future(self) -> bool:
        """Whether the estimator uses samples after the predicted one."""
        return not self.causal


def write_results(path: str | Path, records: list[RunRecord]) -> None:
    """Write records to JSON, grouped by dataset.

    Keys are sorted so re-running with identical results produces a
    byte-identical file: a diff should mean the numbers moved, not that dict
    ordering did.

    Args:
        path: Destination file.
        records: Runs to store; records sharing a dataset are grouped under it
            in the order given.
    """
    datasets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        datasets.setdefault(record.dataset, []).append(asdict(record))

    payload = {'schema_version': SCHEMA_VERSION, 'datasets': datasets}
    # Rename over the target so a killed runner never leaves a truncated
    # artifact that resume would then fail to read.
    tmp = Path(f'{path}.tmp')
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    tmp.replace(path)


def append_results(path: str | Path, records: list[RunRecord]) -> None:
    """Add records to the artifact at ``path``, creating it if absent."""
    path = Path(path)
    existing = read_results(path) if path.exists() else []
    write_results(path, [*existing, *records])


def read_results(path: str | Path) -> list[RunRecord]:
    """Read records back, flattened across datasets.

    Args:
        path: Artifact written by :func:`write_results`.

    Returns:
        The records, dataset groups concatenated in the file's key order.

    Raises:
        ValueError: If the file's schema version is not :data:`SCHEMA_VERSION`.
    """
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    version = payload.get('schema_version')
    if version != SCHEMA_VERSION:
        raise ValueError(
            f'Unsupported results schema version {version!r}; expected {SCHEMA_VERSION}'
        )

    known = {f.name for f in fields(RunRecord)}
    records = []
    for entries in payload['datasets'].values():
        for entry in entries:
            # learned_results.json predates the stored ``causal`` flag.
            if 'causal' not in entry:
                entry['causal'] = not entry['sees_future']
            records.append(RunRecord(**{k: v for k, v in entry.items() if k in known}))
    return records
