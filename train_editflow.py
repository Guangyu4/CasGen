"""Training script for depth-aware Edit Flow cascade model."""
import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.stats import wasserstein_distance
from editflow import DataBatch, EditFlow, D_MAX, GAMMA
from editflow_model import EditFlowTransformer


# ---- Data ----

class CascadeSeqDataset(Dataset):
    def __init__(self, processed_path, max_events=500):
        data = torch.load(processed_path, map_location='cpu', weights_only=False)
        self.items = []
        for c in data['cascades']:
            if len(c['times']) == 0:
                continue
            t = c['times'][:max_events]
            d = c['depths'][:max_events]
            p = c['parent_times'][:max_events]
            self.items.append((t.tolist(), d.tolist(), p.tolist()))
        self.stats = data['stats']
        counts = [len(item[0]) for item in self.items]
        max_c = max(counts)
        hist = torch.zeros(max_c + 1)
        for c in counts:
            hist[c] += 1
        self.count_dist = hist / hist.sum()
        self.t_max = 1.0
        self.d_max = self.stats['D_max']

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):
    ts = [item[0] for item in batch]
    ds = [item[1] for item in batch]
    ps = [item[2] for item in batch]
    return DataBatch.from_sequences(ts, ds, ps)


class TreeNoise:
    """Generates structurally valid random trees as noise.

    Each noise sample is a valid cascade tree grown by a random attachment process:
    - Nodes are added one at a time, each attaching to a uniformly random earlier node as parent.
    - Times are uniformly sampled and sorted, so parent_time < child_time is always satisfied.
    - Depths are derived from the tree structure (parent_depth + 1), clamped to d_max.

    This produces noise that is structurally coherent (valid trees), so the EditFlow alignment
    between noise and data involves tree-like edit operations (leaf insertions/deletions) rather
    than wholesale structural rewrites.
    """
    def __init__(self, count_dist, t_max, d_max):
        self.count_dist = count_dist
        self.t_max = t_max
        self.d_max = d_max

    def _build_tree(self, count, rng):
        if count == 0:
            return np.empty(0), np.empty(0, dtype=np.int64), np.empty(0)
        times = np.sort(rng.random(count)) * self.t_max
        depths = np.zeros(count, dtype=np.int64)
        parent_times = np.zeros(count)
        for i in range(1, count):
            parent = rng.integers(0, i)
            depths[i] = min(int(depths[parent]) + 1, self.d_max)
            parent_times[i] = times[parent]
        return times, depths, parent_times

    def sample(self, n, generator=None):
        dev = generator.device if generator else 'cpu'
        counts = torch.multinomial(
            self.count_dist.to(dev), n, replacement=True, generator=generator
        )
        # Build trees on CPU with numpy (tree structure requires Python loops)
        cpu_gen = torch.Generator(device='cpu')
        seed = int(torch.randint(0, 2**31, (1,), generator=cpu_gen).item())
        rng = np.random.default_rng(seed)
        t_seqs, d_seqs, p_seqs = [], [], []
        for count in counts.cpu().tolist():
            t, d, p = self._build_tree(int(count), rng)
            t_seqs.append(torch.tensor(t, dtype=torch.float32))
            d_seqs.append(torch.tensor(d, dtype=torch.long))
            p_seqs.append(torch.tensor(p, dtype=torch.float32))
        return DataBatch.from_sequences(t_seqs, d_seqs, p_seqs).to(dev)


def compute_w1(gen_lens, ref_lens):
    g = np.array(gen_lens, dtype=float)
    r = np.array(ref_lens, dtype=float)
    if len(g) == 0 or len(r) == 0:
        return 1.0
    mean_r = max(r.mean(), 1)
    return float(wasserstein_distance(g / mean_r, r / mean_r))


# ---- Training ----

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='dataset/processed.pt')
    p.add_argument('--outdir', default='runs_editflow')
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--n_ins_bins', type=int, default=64)
    p.add_argument('--delta', type=float, default=0.05)
    p.add_argument('--gamma', type=float, default=0.3)
    p.add_argument('--rate_penalty', type=float, default=1.0)
    p.add_argument('--max_events', type=int, default=500)
    p.add_argument('--max_steps', type=int, default=30000)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--eval_every', type=int, default=1000)
    p.add_argument('--n_sample_steps', type=int, default=100)
    p.add_argument('--n_val_samples', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    print('Loading data...')
    dataset = CascadeSeqDataset(args.data, max_events=args.max_events)
    n = len(dataset)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)

    ref_lens = [len(dataset.items[i][0]) for i in test_set.indices]
    d_max = dataset.d_max
    print(f'Data: {n} cascades, train={n_train}, val={n_val}, test={n_test}, D_max={d_max}')
    print(f'Ref lengths: mean={np.mean(ref_lens):.1f}, median={np.median(ref_lens):.0f}')

    noise = TreeNoise(dataset.count_dist, dataset.t_max, d_max)
    ef = EditFlow(n_ins_bins=args.n_ins_bins, delta=args.delta,
                  rate_penalty=args.rate_penalty, gamma=args.gamma, d_max=d_max)

    model = EditFlowTransformer(
        hidden_dim=args.hidden_dim, n_heads=args.n_heads, n_layers=args.n_layers,
        n_ins_bins=args.n_ins_bins, d_max=d_max, t_max=dataset.t_max,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Cosine LR decay: helps model stabilize around the good balance it finds early on
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.lr * 0.05
    )
    gen = torch.Generator(device=args.device)
    gen.manual_seed(args.seed)

    step = 0
    best_wd = float('inf')
    train_iter = iter(train_loader)

    print('Training...')
    while step < args.max_steps:
        try:
            x1_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x1_batch = next(train_iter)

        x1_batch = x1_batch.to(args.device)
        x0_batch = noise.sample(x1_batch.batch_size, generator=gen).to(args.device)
        # Length coupling: sort both by count and pair similar-length sequences.
        # This reduces average edit distance and creates more balanced ins/del signal.
        idx0 = sorted(range(x0_batch.batch_size), key=lambda i: x0_batch.seq_lens[i])
        idx1 = sorted(range(x1_batch.batch_size), key=lambda i: x1_batch.seq_lens[i])
        inv1 = [0] * x1_batch.batch_size
        for rank, orig in enumerate(idx1):
            inv1[orig] = rank
        seqs0 = x0_batch.split_sequences()
        deps0 = x0_batch.split_depth()
        pts0 = x0_batch.split_parent_time()
        paired = [None] * x0_batch.batch_size
        for rank, orig1 in enumerate(idx1):
            orig0 = idx0[rank]
            paired[orig1] = (seqs0[orig0], deps0[orig0], pts0[orig0])
        x0_batch = DataBatch.from_sequences(
            [p[0] for p in paired], [p[1] for p in paired], [p[2] for p in paired]
        )
        x1_wrapped = x1_batch.wrap(prefix=0.0, suffix=dataset.t_max)
        x0_wrapped = x0_batch.wrap(prefix=0.0, suffix=dataset.t_max)

        loss = ef.compute_loss(model, x0_wrapped, x1_wrapped, generator=gen)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % 100 == 0:
            print(f'[Step {step:5d}] loss={loss.item():.4f}')

        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                x0_sample = noise.sample(args.n_val_samples, generator=gen)
                x0_sample = x0_sample.to(args.device).wrap(prefix=0.0, suffix=dataset.t_max)
                result, n_ins, n_del = ef.sample(model, x0_sample,
                                                  n_steps=args.n_sample_steps, generator=gen)

            gen_lens = list(result.seq_lens)
            wd = compute_w1(gen_lens, ref_lens)

            gen_depths = result.depth.cpu().numpy()
            depth_dist = np.bincount(gen_depths.clip(0, d_max), minlength=d_max + 1)
            depth_str = '/'.join(str(int(d)) for d in depth_dist)
            print(f'  [Eval] gen_mean={np.mean(gen_lens):.1f} gen_median={np.median(gen_lens):.0f} '
                  f'ref_mean={np.mean(ref_lens):.1f} W1={wd:.4f} ins={n_ins} del={n_del} '
                  f'depth_dist=[{depth_str}]')

            if wd < best_wd:
                best_wd = wd
                torch.save({
                    'step': step, 'model': model.state_dict(), 'args': vars(args),
                    'best_wd': best_wd, 'stats': dataset.stats, 'd_max': d_max,
                }, os.path.join(args.outdir, 'best.pt'))

            model.train()

    print(f'Training complete. Best W1: {best_wd:.4f}')


if __name__ == '__main__':
    main()
