"""Unified training entry point.

Usage:
  python train.py --model OURS --variant motive --cond_data dataset/APS_burst.pt
  python train.py --model OURS --variant uncond  --data dataset/APS_burst.pt
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from trainers import TRAINERS


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--model', default='OURS', choices=list(TRAINERS),
                     help='Which model to train: ' + ', '.join(TRAINERS))
    known, _ = pre.parse_known_args()

    trainer = TRAINERS[known.model]

    parser = argparse.ArgumentParser(
        description=f'Train {known.model}',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', default=known.model, choices=list(TRAINERS))
    trainer.add_args(parser)

    args = parser.parse_args()
    trainer.train(args)


if __name__ == '__main__':
    main()
