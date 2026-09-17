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

    ``total_params``, ``receptive_field`` and ``sees_future`` are required, not
    optional: a causal and a non-causal model at equal parameter count are not
    solving the same problem, and a comparison table that cannot express the
    difference silently implies they are.

    Attributes:
        model_type: Architecture name as passed to ``create_model``.
        config: The model hyperparameters the run used.
        total_params: Trainable parameter count.
        receptive_field: Receptive field in samples.
        sees_future: Whether the model uses samples after the predicted one.
        seed: Random seed the run was trained with.
        device: Device name, e.g. the CUDA device string.
        dataset: Dataset key, matching baseline_results.json's ('free-space'
            or 'mmfiber').
        metrics: Metric name -> summary, each summary using the six keys from
            :func:`smi.analysis.metrics.summarize_per_shot`.
    """

    model_type: str
    config: dict[str, Any]
    total_params: int
    receptive_field: int
    sees_future: bool
    seed: int
    device: str
    dataset: str
    metrics: dict[str, dict[str, float]]


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
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )


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
    return [
        RunRecord(**{k: v for k, v in entry.items() if k in known})
        for entries in payload['datasets'].values()
        for entry in entries
    ]
