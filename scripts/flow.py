"""R8 flow matching: train a FlowLitModule config, or score the trained flows.

The CLI over ``smi.experiments.flow_runner``, which holds the budget and the
rationale.

    pixi run python scripts/flow.py train flow/denoise-unettransformer-offline-44k
    pixi run python scripts/flow.py score
    pixi run python scripts/flow.py sign --device mps
    pixi run python scripts/flow.py score --variant _enc
"""

import argparse
from pathlib import Path

import torch

from smi.core.cli import add_verbosity, setup_logging
from smi.core.paths import RESULTS_DIR
from smi.data.frozen import DEFAULT_ROOT
from smi.experiments.flow_runner import VARIANTS, score, score_sign, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_verbosity(parser)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train', help='train one flow config')
    t.add_argument(
        'config', choices=[n for d, p in VARIANTS.values() for n in (d, *p.values())]
    )
    t.add_argument('--accelerator', default='cpu')
    t.add_argument('--seed', type=int, default=0)
    s = sub.add_parser('score', help='score the trained flows')
    s.add_argument('--device', default='cpu')
    s.add_argument('--frozen-root', type=Path, default=DEFAULT_ROOT)
    s.add_argument('--out', type=Path, default=RESULTS_DIR / 'flow_results.json')
    g = sub.add_parser('sign', help="add the posteriors' sign scores to --out")
    g.add_argument('--device', default='cpu')
    g.add_argument('--frozen-root', type=Path, default=DEFAULT_ROOT)
    g.add_argument('--out', type=Path, default=RESULTS_DIR / 'flow_results.json')
    for p in (s, g):
        p.add_argument(
            '--variant',
            choices=list(VARIANTS),
            default='',
            help="'' for the original flows, '_enc' for the encoder ones",
        )
    args = parser.parse_args()
    setup_logging(args.verbosity)
    torch.set_float32_matmul_precision('high')
    if args.command == 'train':
        train(args.config, args.accelerator, args.seed)
    elif args.command == 'sign':
        score_sign(args.device, args.frozen_root, args.out, args.variant)
    else:
        score(args.device, args.frozen_root, args.out, args.variant)


if __name__ == '__main__':
    main()
