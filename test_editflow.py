"""Evaluate depth-aware Edit Flow model."""
import argparse
import json
import os
import time
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from torch.utils.data import random_split

from editflow import DataBatch, EditFlow
from editflow_model import EditFlowTransformer
from train_editflow import CascadeSeqDataset, TreeNoise


def counting_distance(x, Y, t_max):
    x, Y = x.copy(), Y.copy()
    x = x[None].repeat(Y.shape[0], 0)
    x_len = (x < t_max).sum(-1)
    y_len = (Y < t_max).sum(-1)
    to_swap = x_len > y_len
    x[to_swap], Y[to_swap] = Y[to_swap].copy(), x[to_swap].copy()
    mask_x = x < t_max
    mask_y = Y < t_max
    result = (np.abs(x - Y) * mask_x).sum(-1)
    result += ((t_max - Y) * (~mask_x & mask_y)).sum(-1)
    return result


def match_shapes(X, Y, t_max):
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


def compute_mmd(X, Y, t_max, sigma=None):
    X, Y = match_shapes(X, Y, t_max)
    X, Y = X / t_max, Y / t_max
    x_x_d = np.concatenate([counting_distance(x, X, 1.0) for x in X])
    x_y_d = np.concatenate([counting_distance(x, Y, 1.0) for x in X])
    y_y_d = np.concatenate([counting_distance(y, Y, 1.0) for y in Y])
    if sigma is None:
        sigma = np.median(np.concatenate([x_x_d, x_y_d, y_y_d]))
    s2 = sigma ** 2 + 1e-8
    return float(np.sqrt(max(
        np.mean(np.exp(-x_x_d/(2*s2))) - 2*np.mean(np.exp(-x_y_d/(2*s2))) + np.mean(np.exp(-y_y_d/(2*s2))), 0
    ))), float(sigma)


def compute_count_w1(X, Y, t_max, mean_n):
    X_lens = np.array([(s < t_max).sum() for s in X])
    Y_lens = np.array([(s < t_max).sum() for s in Y])
    return float(wasserstein_distance(X_lens / max(mean_n, 1), Y_lens / max(mean_n, 1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data', default='dataset/processed.pt')
    p.add_argument('--outdir', default=None)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--n_steps', type=int, default=100)
    a = p.parse_args()

    if a.outdir is None:
        a.outdir = os.path.join(os.path.dirname(a.ckpt), 'eval')
    os.makedirs(a.outdir, exist_ok=True)

    print(f'Loading {a.ckpt}')
    ckpt = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    args = argparse.Namespace(**ckpt['args'])
    d_max = ckpt.get('d_max', 2)

    dataset = CascadeSeqDataset(a.data, max_events=args.max_events)
    n = len(dataset)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    n_test = n - n_train - n_val
    _, _, test_set = random_split(dataset, [n_train, n_val, n_test],
                                   generator=torch.Generator().manual_seed(args.seed))
    print(f'Test set: {n_test} cascades')

    model = EditFlowTransformer(
        hidden_dim=args.hidden_dim, n_heads=args.n_heads, n_layers=args.n_layers,
        n_ins_bins=args.n_ins_bins, d_max=d_max, t_max=dataset.t_max,
    ).to(a.device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Model loaded (step {ckpt["step"]})')

    noise = TreeNoise(dataset.count_dist, dataset.t_max, d_max)
    ef = EditFlow(n_ins_bins=args.n_ins_bins, delta=args.delta,
                  rate_penalty=getattr(args, 'rate_penalty', 0.5),
                  gamma=args.gamma, d_max=d_max)
    gen = torch.Generator(device=a.device)
    gen.manual_seed(12345)

    print('Generating...')
    t0 = time.time()
    x0 = noise.sample(n_test, generator=gen).to(a.device).wrap(prefix=0.0, suffix=dataset.t_max)
    result, n_ins, n_del = ef.sample(model, x0, n_steps=a.n_steps, generator=gen)
    gen_time = time.time() - t0
    print(f'Generated in {gen_time:.1f}s (ins={n_ins}, del={n_del})')

    # Collect results
    gen_seqs = list(zip(result.split_sequences(), result.split_depth(), result.split_parent_time()))
    ref_items = [dataset.items[i] for i in test_set.indices]

    gen_lens = [len(s[0]) for s in gen_seqs]
    ref_lens = [len(r[0]) for r in ref_items]
    print(f'Generated: mean={np.mean(gen_lens):.1f}, median={np.median(gen_lens):.0f}')
    print(f'Reference: mean={np.mean(ref_lens):.1f}, median={np.median(ref_lens):.0f}')

    # Metrics
    gen_time_arrays = [s[0].cpu().numpy() for s in gen_seqs]
    ref_time_arrays = [np.array(r[0]) for r in ref_items]
    t_max = 1.0
    mean_n = np.mean(ref_lens)

    print('Computing MMD...')
    mmd, sigma = compute_mmd(gen_time_arrays, ref_time_arrays, t_max)
    print(f'  MMD = {mmd:.6f}')

    print('Computing W1...')
    w1 = compute_count_w1(gen_time_arrays, ref_time_arrays, t_max, mean_n)
    print(f'  W1 = {w1:.6f}')

    # Save metrics
    results = {
        'mmd': mmd, 'mmd_sigma': sigma, 'w1_count': w1,
        'gen_mean_len': float(np.mean(gen_lens)), 'ref_mean_len': float(np.mean(ref_lens)),
        'gen_time_sec': gen_time, 'step': ckpt['step'],
    }
    with open(os.path.join(a.outdir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save cascades with tree structure
    cascades = []
    for i, (gen_s, ref_r) in enumerate(zip(gen_seqs, ref_items)):
        g_times = gen_s[0].cpu().tolist()
        g_depths = gen_s[1].cpu().tolist()
        g_pt = gen_s[2].cpu().tolist()
        g_parents = EditFlow.reconstruct_tree(gen_s[0].cpu(), gen_s[2].cpu())

        r_times, r_depths, r_pt = ref_r
        r_parents = EditFlow.reconstruct_tree(
            torch.tensor(r_times), torch.tensor(r_pt))

        cascades.append({
            'gen_times': g_times, 'gen_depths': g_depths,
            'gen_parent_times': g_pt, 'gen_tree_parents': g_parents,
            'ref_times': r_times, 'ref_depths': r_depths,
            'ref_parent_times': r_pt, 'ref_tree_parents': r_parents,
        })

    with open(os.path.join(a.outdir, 'generated_cascades.json'), 'w') as f:
        json.dump(cascades, f)

    print(f'\nResults saved to {a.outdir}/')
    print(f'  MMD: {mmd:.6f}  W1: {w1:.6f}')


if __name__ == '__main__':
    main()
