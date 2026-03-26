"""Unified test entry point for cascade sequence generation models.

Usage (standalone):
    python test.py --model CASFLOW --outdir runs_casflow/reddit_123 \
        --data dataset/RedditM_burst.pt ...

All model-specific arguments are the same as those used during training.
The checkpoint is loaded from <outdir>/best.pt by default.

Three evaluation metrics (reported for all models):
    W1   - Wasserstein-1 distance on normalized length distributions
    MAE  - Mean Absolute Error on sequence lengths (sort-matched pairs)
    MSLE - Mean Squared Log Error on sequence lengths (sort-matched pairs)

Results are printed to stdout and saved to <outdir>/test_results.json.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from trainers import TRAINERS


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--model', default='OURS', choices=list(TRAINERS),
                     help='Model to test: ' + ', '.join(TRAINERS))
    known, _ = pre.parse_known_args()

    trainer = TRAINERS[known.model]

    parser = argparse.ArgumentParser(
        description=f'Test {known.model}',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', default=known.model, choices=list(TRAINERS))
    trainer.add_args(parser)

    args = parser.parse_args()
    trainer.test(args)


if __name__ == '__main__':
    main()
