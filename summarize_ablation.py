"""Collect metrics.json from all ablation run dirs and print a summary table.

Usage:
    python summarize_ablation.py runs_ablation/
"""
import json
import os
import sys
import glob

base = sys.argv[1] if len(sys.argv) > 1 else 'runs_ablation'

VARIANT_LABEL = {
    'uncond': 'W/o Cond (baseline)',
    'full':   'W/o Cond (baseline)',   # backward compat alias
    'bert':   'W/ BERT Cond',
    'motive': 'W/ Motive Cond',
    'flat':   'W/o Tree',
    'ddpm':   'W/o Flowmatching',
}

rows = []
for metrics_path in sorted(glob.glob(os.path.join(base, '*/eval/metrics.json'))):
    with open(metrics_path) as f:
        m = json.load(f)
    variant = m.get('variant', 'unknown')
    rows.append({
        'variant': variant,
        'label': VARIANT_LABEL.get(variant, variant),
        'mmd': m['mmd'],
        'w1_l': m['w1_l'],
        'w1_t': m['w1_t'],
        'gen_mean_len': m.get('gen_mean_len', float('nan')),
        'ref_mean_len': m.get('ref_mean_len', float('nan')),
        'step': m.get('step', '?'),
        'path': metrics_path,
    })

if not rows:
    print(f'No metrics.json found under {base}/')
    print('Jobs may still be running. Check: squeue -u $USER')
    sys.exit(0)

# Sort by canonical order
order = ['uncond', 'full', 'bert', 'motive', 'flat', 'ddpm']
rows.sort(key=lambda r: order.index(r['variant']) if r['variant'] in order else 99)

header = f"{'Model':<20} {'MMD':>10} {'W1(l)':>10} {'W1(t)':>10}  {'GenLen':>8} {'RefLen':>8}  {'Step':>7}"
sep = '-' * len(header)
print(sep)
print(header)
print(sep)
for r in rows:
    print(f"{r['label']:<20} {r['mmd']:>10.6f} {r['w1_l']:>10.6f} {r['w1_t']:>10.6f}  "
          f"{r['gen_mean_len']:>8.1f} {r['ref_mean_len']:>8.1f}  {r['step']:>7}")
print(sep)
print(f"(lower is better for all metrics)")
