"""Detailed statistical description of cascade .pt datasets."""
import sys
import os
import torch
import numpy as np
from collections import Counter

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
TARGETS = ['APS.pt', 'WeiboCov.pt', 'RedditM.pt']


def describe(path):
    name = os.path.basename(path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"\n{'='*70}")
    print(f"  {name}  ({size_mb:.1f} MB)")
    print(f"{'='*70}")

    d = torch.load(path, weights_only=False, map_location='cpu')
    cascades = d['cascades']
    vocab    = d.get('vocab', {})
    stats    = d.get('stats', {})

    n = len(cascades)
    print(f"\n[Top-level]")
    print(f"  n_cascades : {n:,}")
    print(f"  vocab_size : {len(vocab):,}")
    print(f"  stored stats: {stats}")

    # ---- cascade lengths ----
    lens = np.array([len(c['times']) for c in cascades])
    print(f"\n[Cascade length  (# events, excluding root)]")
    print(f"  min    : {lens.min()}")
    print(f"  max    : {lens.max()}")
    print(f"  mean   : {lens.mean():.2f}")
    print(f"  median : {np.median(lens):.1f}")
    print(f"  std    : {lens.std():.2f}")
    pcts = [25, 50, 75, 90, 95, 99]
    pct_vals = np.percentile(lens, pcts)
    print(f"  percentiles: " + "  ".join(f"p{p}={v:.0f}" for p, v in zip(pcts, pct_vals)))

    # distribution buckets
    buckets = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    print(f"  length distribution:")
    prev = 0
    for b in buckets:
        cnt = int(((lens > prev) & (lens <= b)).sum())
        print(f"    ({prev:>5}, {b:>5}] : {cnt:>8,}  ({100*cnt/n:.1f}%)")
        prev = b
    cnt = int((lens > prev).sum())
    print(f"    ({prev:>5},  inf] : {cnt:>8,}  ({100*cnt/n:.1f}%)")

    # ---- depths ----
    all_depths = np.concatenate([c['depths'].numpy() for c in cascades])
    print(f"\n[Tree depth  (event-level, root=0 excluded)]")
    print(f"  max depth (D_max): {all_depths.max()}")
    depth_counter = Counter(all_depths.tolist())
    for d_val in sorted(depth_counter):
        cnt = depth_counter[d_val]
        print(f"  depth={d_val:2d} : {cnt:>10,}  ({100*cnt/len(all_depths):.1f}%)")

    # ---- times ----
    all_times = np.concatenate([c['times'].numpy() for c in cascades])
    print(f"\n[Normalized times  (event-level, in (0,1])]")
    print(f"  min    : {all_times.min():.6f}")
    print(f"  max    : {all_times.max():.6f}")
    print(f"  mean   : {all_times.mean():.4f}")
    print(f"  median : {np.median(all_times):.4f}")
    print(f"  std    : {all_times.std():.4f}")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(all_times, bins=bins)
    total_events = len(all_times)
    print(f"  time histogram (deciles):")
    for i in range(len(hist)):
        bar = '#' * int(40 * hist[i] / hist.max())
        print(f"    [{bins[i]:.1f}, {bins[i+1]:.1f}] : {hist[i]:>10,} ({100*hist[i]/total_events:.1f}%)  {bar}")

    # ---- t_max (observation window) ----
    t_maxs = np.array([c['t_max'] for c in cascades])
    print(f"\n[t_max  (observation window, seconds)]")
    print(f"  min    : {t_maxs.min():.1f}  ({t_maxs.min()/86400:.1f} days)")
    print(f"  max    : {t_maxs.max():.1f}  ({t_maxs.max()/86400:.1f} days)")
    print(f"  mean   : {t_maxs.mean():.1f}  ({t_maxs.mean()/86400:.1f} days)")
    print(f"  median : {np.median(t_maxs):.1f}  ({np.median(t_maxs)/86400:.1f} days)")

    # ---- parent_times ----
    all_ptimes = np.concatenate([c['parent_times'].numpy() for c in cascades])
    frac_root_parent = (all_ptimes == 0.0).sum() / len(all_ptimes)
    print(f"\n[parent_times]")
    print(f"  events with parent=root (pt=0.0): {frac_root_parent*100:.1f}%  "
          f"(depth-1 events or ambiguous)")
    nonzero = all_ptimes[all_ptimes > 0]
    if len(nonzero):
        print(f"  non-zero parent_times: mean={nonzero.mean():.4f}  "
              f"median={np.median(nonzero):.4f}  std={nonzero.std():.4f}")

    # ---- text (abstract/post) ----
    texts = [c['text'] for c in cascades]
    text_lens = np.array([len(t) for t in texts])
    print(f"\n[text  (root cascade text)]")
    print(f"  min chars  : {text_lens.min()}")
    print(f"  max chars  : {text_lens.max()}")
    print(f"  mean chars : {text_lens.mean():.1f}")
    print(f"  median chars: {np.median(text_lens):.1f}")
    print(f"  sample text: {texts[0][:120]!r}")

    # ---- vocab ----
    if vocab:
        print(f"\n[vocab  (char-level)]")
        print(f"  total chars: {len(vocab):,}  (incl. <PAD>, <UNK>)")
        printable_chars = [ch for ch in vocab if len(ch) == 1 and ch.isprintable()]
        print(f"  printable single-char entries: {len(printable_chars)}")
        sample = ''.join(sorted(printable_chars)[:80])
        print(f"  sample chars: {sample!r}")

    print()


def main():
    for fname in TARGETS:
        path = os.path.join(DATASET_DIR, fname)
        if not os.path.exists(path):
            print(f"\n[SKIP] {fname} not found at {path}")
            continue
        try:
            describe(path)
        except Exception as e:
            print(f"\n[ERROR] {fname}: {e}")
            import traceback; traceback.print_exc()


if __name__ == '__main__':
    main()
