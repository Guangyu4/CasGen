#!/bin/bash
#SBATCH --job-name=describe-reddit
#SBATCH --partition=sfscai
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/describe_reddit_%j.out
#SBATCH --error=logs/describe_reddit_%j.err

source ~/.bashrc
conda activate T2S

cd /scratch/gw2556/ODE/CasGen
mkdir -p logs

echo "=== RedditM Dataset Description ==="
python - <<'EOF'
import os, torch, numpy as np
from collections import Counter

path = 'dataset/RedditM.pt'
size_mb = os.path.getsize(path) / 1024 / 1024
print(f"\n{'='*70}")
print(f"  RedditM.pt  ({size_mb:.1f} MB)")
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

buckets = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
print(f"  length distribution:")
prev = 0
for b in buckets:
    cnt = int(((lens > prev) & (lens <= b)).sum())
    print(f"    ({prev:>5}, {b:>5}] : {cnt:>8,}  ({100*cnt/n:.1f}%)")
    prev = b
cnt = int((lens > prev).sum())
print(f"    ({prev:>5},  inf] : {cnt:>8,}  ({100*cnt/n:.1f}%)")

all_depths = np.concatenate([c['depths'].numpy() for c in cascades])
print(f"\n[Tree depth  (event-level)]")
print(f"  max depth: {all_depths.max()}")
depth_counter = Counter(all_depths.tolist())
for dv in sorted(depth_counter)[:15]:
    cnt = depth_counter[dv]
    print(f"  depth={dv:2d} : {cnt:>10,}  ({100*cnt/len(all_depths):.1f}%)")
if all_depths.max() >= 15:
    print(f"  ... (truncated at depth 14)")

all_times = np.concatenate([c['times'].numpy() for c in cascades])
print(f"\n[Normalized times  (event-level)]")
print(f"  min={all_times.min():.6f}  max={all_times.max():.6f}  "
      f"mean={all_times.mean():.4f}  median={np.median(all_times):.4f}  std={all_times.std():.4f}")
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
hist, _ = np.histogram(all_times, bins=bins)
total_ev = len(all_times)
print(f"  time histogram (deciles):")
for i in range(len(hist)):
    bar = '#' * int(40 * hist[i] / hist.max())
    print(f"    [{bins[i]:.1f},{bins[i+1]:.1f}] : {hist[i]:>10,} ({100*hist[i]/total_ev:.1f}%)  {bar}")

t_maxs = np.array([c['t_max'] for c in cascades])
print(f"\n[t_max  (observation window, seconds)]")
print(f"  min={t_maxs.min():.1f}s ({t_maxs.min()/3600:.1f}h)  "
      f"max={t_maxs.max():.1f}s ({t_maxs.max()/86400:.1f}d)  "
      f"mean={t_maxs.mean():.1f}s ({t_maxs.mean()/3600:.1f}h)  "
      f"median={np.median(t_maxs):.1f}s ({np.median(t_maxs)/3600:.1f}h)")

all_ptimes = np.concatenate([c['parent_times'].numpy() for c in cascades])
frac_root = (all_ptimes == 0.0).sum() / len(all_ptimes)
print(f"\n[parent_times]")
print(f"  pt=0.0 (parent is root): {frac_root*100:.1f}%")
nonzero = all_ptimes[all_ptimes > 0]
if len(nonzero):
    print(f"  non-zero: mean={nonzero.mean():.4f}  median={np.median(nonzero):.4f}  std={nonzero.std():.4f}")

texts = [c['text'] for c in cascades]
text_lens = np.array([len(t) for t in texts])
print(f"\n[text]")
print(f"  min={text_lens.min()}  max={text_lens.max()}  "
      f"mean={text_lens.mean():.1f}  median={np.median(text_lens):.1f}")
print(f"  sample: {texts[0][:120]!r}")

if vocab:
    print(f"\n[vocab]")
    print(f"  size: {len(vocab):,}  (incl. <PAD>, <UNK>)")
    chars = sorted(ch for ch in vocab if len(ch)==1 and ch.isprintable())
    print(f"  printable chars ({len(chars)}): {''.join(chars[:80])!r}")

print()
EOF
echo "=== Done ==="
