"""Shared evaluation metrics for cascade sequence generation.

Three metrics, identical to test_editflow.py:
  MMD    - Maximum Mean Discrepancy on event-time sequences
           (counting-distance Gaussian kernel, median heuristic for sigma)
  W1(l)  - Wasserstein-1 on cascade length (event count) distribution,
           normalized by the reference mean length
  W1(t)  - Wasserstein-1 on pooled inter-event time (gap) distribution

All three operate on sequences of absolute times in [0, t_max].
"""
import os
import json
import numpy as np
from scipy.stats import wasserstein_distance
import torch
from torch.utils.data import random_split


# ---- Metric helpers (same as test_editflow.py) ----

def _match_shapes(X, Y, t_max):
    max_x = max((x < t_max).sum() for x in X) if X else 0
    max_y = max((y < t_max).sum() for y in Y) if Y else 0
    max_size = max(max_x, max_y, 1)
    new_X = np.ones((len(X), max_size)) * t_max
    new_Y = np.ones((len(Y), max_size)) * t_max
    for i, x in enumerate(X):
        x = x[x < t_max]; new_X[i, :len(x)] = x
    for i, y in enumerate(Y):
        y = y[y < t_max]; new_Y[i, :len(y)] = y
    return new_X, new_Y


def _counting_dist_matrix(A, B, t_max=1.0):
    """Vectorized pairwise counting distance: A(Na,L) × B(Nb,L) → (Na,Nb).

    Replaces the original Python for-loop over counting_distance(x, Y).
    Uses chunked numpy broadcasting to cap per-chunk memory at ~256 MB.
    """
    Na, Nb, L = len(A), len(B), A.shape[1]
    # chunk_size rows/cols so that chunk_a * chunk_b * L * 8 bytes ≤ 256 MB
    chunk_size = max(1, int((256 * 1024 ** 2 / (L * 8)) ** 0.5))

    a_len = (A < t_max).sum(1)
    b_len = (B < t_max).sum(1)
    D = np.empty((Na, Nb), dtype=np.float64)

    for i0 in range(0, Na, chunk_size):
        Ai  = A[i0: i0 + chunk_size]
        ali = a_len[i0: i0 + chunk_size]
        for j0 in range(0, Nb, chunk_size):
            Bj  = B[j0: j0 + chunk_size]
            blj = b_len[j0: j0 + chunk_size]

            # (ca, cb, 1) swap mask: ensure shorter sequence is S, longer is T
            swap = (ali[:, None] > blj[None, :])[:, :, None]
            Ai3  = Ai[:, None, :]
            Bj3  = Bj[None, :, :]
            S    = np.where(swap, Bj3, Ai3)
            T_   = np.where(swap, Ai3, Bj3)

            mask_s = S < t_max
            mask_t = T_ < t_max
            d  = (np.abs(S - T_) * mask_s).sum(-1)
            d += ((t_max - T_) * (~mask_s & mask_t)).sum(-1)
            D[i0: i0 + len(Ai), j0: j0 + len(Bj)] = d

    return D


def compute_mmd(X, Y, t_max=1.0, sigma=None, n_subsample=2000):
    """MMD with counting-distance Gaussian kernel (median heuristic).

    n_subsample: randomly subsample each set to at most this many sequences
    before computing pairwise distances, making O(N²) cost tractable.
    """
    rng = np.random.default_rng(0)
    if len(X) > n_subsample:
        X = [X[i] for i in rng.choice(len(X), n_subsample, replace=False)]
    if len(Y) > n_subsample:
        Y = [Y[i] for i in rng.choice(len(Y), n_subsample, replace=False)]

    X, Y = _match_shapes(X, Y, t_max)
    X, Y = X / t_max, Y / t_max

    Dxx = _counting_dist_matrix(X, X, 1.0).ravel()
    Dxy = _counting_dist_matrix(X, Y, 1.0).ravel()
    Dyy = _counting_dist_matrix(Y, Y, 1.0).ravel()

    if sigma is None:
        sigma = np.median(np.concatenate([Dxx, Dxy, Dyy]))
    s2 = sigma ** 2 + 1e-8
    val = float(np.sqrt(max(
        np.mean(np.exp(-Dxx / (2 * s2)))
        - 2 * np.mean(np.exp(-Dxy / (2 * s2)))
        + np.mean(np.exp(-Dyy / (2 * s2))),
        0
    )))
    return val, float(sigma)


def compute_w1_length(X, Y, t_max=1.0, mean_n=None):
    """W1 on cascade length distribution, normalized by mean reference length."""
    X_lens = np.array([(s < t_max).sum() for s in X], dtype=float)
    Y_lens = np.array([(s < t_max).sum() for s in Y], dtype=float)
    if mean_n is None:
        mean_n = max(Y_lens.mean(), 1.0)
    return float(wasserstein_distance(X_lens / mean_n, Y_lens / mean_n))


def compute_w1_intertime(X, Y, t_max=1.0):
    """W1 on pooled inter-event gap distribution."""
    def pool_gaps(seqs):
        gaps = []
        for seq in seqs:
            valid = np.sort(seq[seq < t_max])
            if len(valid) > 1:
                gaps.extend(np.diff(valid).tolist())
        return np.array(gaps, dtype=float)

    gen_gaps = pool_gaps(X)
    ref_gaps = pool_gaps(Y)
    if len(gen_gaps) == 0 or len(ref_gaps) == 0:
        return 1.0
    return float(wasserstein_distance(gen_gaps, ref_gaps))


# ---- Unified evaluation ----

def eval_metrics(gen_seqs, ref_seqs, t_max=1.0):
    """
    Compute all three metrics.

    Args:
        gen_seqs : list of 1-D numpy arrays of absolute event times
        ref_seqs : list of 1-D numpy arrays of absolute event times
        t_max    : time horizon (sequences are in [0, t_max])
    Returns:
        dict with keys: mmd, mmd_sigma, w1_l, w1_t,
                        gen_mean_len, gen_med_len, ref_mean_len, ref_med_len, n
    """
    gen_lens = np.array([(s < t_max).sum() for s in gen_seqs], dtype=float)
    ref_lens = np.array([(s < t_max).sum() for s in ref_seqs], dtype=float)
    mean_n   = max(ref_lens.mean(), 1.0)

    print(f'  Generated: mean={gen_lens.mean():.1f}  median={np.median(gen_lens):.0f}')
    print(f'  Reference: mean={ref_lens.mean():.1f}  median={np.median(ref_lens):.0f}')

    print('  Computing MMD...')
    mmd, sigma = compute_mmd(gen_seqs, ref_seqs, t_max)

    print('  Computing W1(l)...')
    w1_l = compute_w1_length(gen_seqs, ref_seqs, t_max, mean_n)

    print('  Computing W1(t)...')
    w1_t = compute_w1_intertime(gen_seqs, ref_seqs, t_max)

    return {
        'mmd':          mmd,
        'mmd_sigma':    sigma,
        'w1_l':         w1_l,
        'w1_t':         w1_t,
        'gen_mean_len': float(gen_lens.mean()),
        'gen_med_len':  float(np.median(gen_lens)),
        'ref_mean_len': float(ref_lens.mean()),
        'ref_med_len':  float(np.median(ref_lens)),
        'n':            len(ref_seqs),
    }


def print_and_save(metrics: dict, outdir: str, prefix: str = 'test'):
    """Print metrics table and save to JSON."""
    print(f'\n{"=" * 44}')
    print(f'  {prefix.upper()} RESULTS  (n={metrics.get("n", "?")})')
    print(f'  MMD    = {metrics["mmd"]:.6f}  (sigma={metrics.get("mmd_sigma", 0):.4f})')
    print(f'  W1(l)  = {metrics["w1_l"]:.6f}')
    print(f'  W1(t)  = {metrics["w1_t"]:.6f}')
    print(f'  gen_mean_len = {metrics.get("gen_mean_len", float("nan")):.1f}')
    print(f'  ref_mean_len = {metrics.get("ref_mean_len", float("nan")):.1f}')
    print(f'{"=" * 44}\n')

    path = os.path.join(outdir, f'{prefix}_results.json')
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Results saved to {path}')


# ---- Split helper ----

def get_test_split(dataset, args):
    """Return the same deterministic test split used during training."""
    n       = len(dataset)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    n_test  = n - n_train - n_val
    _, _, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )
    return test_set


# ---- Sequence conversion helpers ----

def to_numpy_seq(t):
    """Convert a tensor or list of times to a 1-D numpy float array."""
    if isinstance(t, torch.Tensor):
        return t.cpu().numpy().astype(float)
    return np.array(t, dtype=float)
