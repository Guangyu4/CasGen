"""Preprocess cascades.pt -> processed.pt
Computes depth, parent_time for each node, builds char-level text vocabulary.
Run via SLURM: sbatch run_preprocess.slurm
"""
import os
import torch
from collections import Counter
from tqdm import tqdm

RAW_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'socialnet_cascades.pt')
OUT_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'socialnet_processed.pt')

PAD_IDX = 0
UNK_IDX = 1
MIN_CHAR_FREQ = 2


def process_cascade(cascade):
    root_user = cascade['root_user']
    nodes = cascade['nodes']
    t_max = cascade['t_max']
    if t_max <= 0:
        t_max = 1.0

    user_times = {}
    times, depths, parent_times = [], [], []

    for node in nodes:
        uid = node['user_id']
        t = node['time'] / t_max
        depth = len(node['path']) - 1

        puid = node['parent_user_id']
        if puid is None or puid == root_user:
            pt = 0.0
        elif puid in user_times:
            candidates = [tp for tp in user_times[puid] if tp <= t]
            pt = max(candidates) if candidates else 0.0
        else:
            pt = 0.0

        times.append(t)
        depths.append(depth)
        parent_times.append(pt)

        user_times.setdefault(uid, []).append(t)

    return {
        'times': torch.tensor(times, dtype=torch.float32),
        'depths': torch.tensor(depths, dtype=torch.long),
        'parent_times': torch.tensor(parent_times, dtype=torch.float32),
        'text': cascade.get('text', ''),
        't_max': t_max,
    }


def build_vocab(texts):
    counter = Counter()
    for t in texts:
        counter.update(t)
    char2idx = {'<PAD>': PAD_IDX, '<UNK>': UNK_IDX}
    for ch, freq in counter.most_common():
        if freq >= MIN_CHAR_FREQ:
            char2idx[ch] = len(char2idx)
    return char2idx


def main():
    print(f'Loading {RAW_PATH} ...')
    raw = torch.load(RAW_PATH, map_location='cpu', weights_only=False)
    print(f'Loaded {len(raw)} cascades')

    processed = []
    skipped = 0
    for c in tqdm(raw, desc='Processing'):
        if len(c['nodes']) == 0:
            skipped += 1
            continue
        processed.append(process_cascade(c))

    print(f'Processed {len(processed)} cascades (skipped {skipped} empty)')

    texts = [c['text'] for c in processed]
    vocab = build_vocab(texts)
    print(f'Vocabulary size: {len(vocab)}')

    all_depths = torch.cat([c['depths'] for c in processed])
    all_lens = [len(c['times']) for c in processed]
    D_max = int(all_depths.max().item())
    n_max = max(all_lens)

    stats = {
        'D_max': D_max,
        'n_max': n_max,
        'n_cascades': len(processed),
        'avg_len': sum(all_lens) / len(all_lens),
    }
    print(f'Stats: D_max={D_max}, n_max={n_max}, avg_len={stats["avg_len"]:.1f}')

    out = {'cascades': processed, 'vocab': vocab, 'stats': stats}
    torch.save(out, OUT_PATH)
    print(f'Saved to {OUT_PATH}')


if __name__ == '__main__':
    main()
