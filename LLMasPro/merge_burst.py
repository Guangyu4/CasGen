"""
Merge burst_scores from *_texts.json into *_pt dataset files.

For each dataset, adds a 'burst_scores' tensor (shape [9], float32) to every
cascade dict and saves the result as {name}_burst.pt alongside the original.

The mapping uses the 'idx' field in the JSON, which equals the cascade's
position in pt['cascades'].

Usage:
  python merge_burst.py [APS WeiboCov RedditM]
"""

import json
import sys
import time
from pathlib import Path

import torch

DIR        = Path(__file__).parent
TEXTS_DIR  = DIR / 'texts'
DATASET    = DIR.parent / 'dataset'

TARGETS    = ['APS', 'WeiboCov', 'RedditM']
N_KEYS     = 9


def stream_idx_scores(json_path: Path) -> dict[int, list]:
    """Stream JSON, return {idx: burst_scores} for scored records only."""
    decoder = json.JSONDecoder()
    mapping: dict[int, list] = {}
    buf = ''
    in_array = False
    with open(json_path, encoding='utf-8') as f:
        for raw_line in f:
            if not in_array:
                if '[' in raw_line:
                    in_array = True
                    buf = raw_line[raw_line.index('[') + 1:]
                continue
            buf += raw_line
            buf = buf.lstrip()
            if buf.startswith(','):
                buf = buf[1:].lstrip()
            if not buf or buf == ']':
                continue
            while buf.lstrip().startswith('{'):
                buf = buf.lstrip()
                try:
                    obj, end = decoder.raw_decode(buf)
                    bs = obj.get('burst_scores')
                    if bs is not None:
                        mapping[int(obj['idx'])] = bs
                    buf = buf[end:].lstrip()
                    if buf.startswith(','):
                        buf = buf[1:].lstrip()
                except json.JSONDecodeError:
                    break
    return mapping


def merge(name: str) -> None:
    pt_in   = DATASET / f'{name}.pt'
    pt_out  = DATASET / f'{name}_burst.pt'
    json_path = TEXTS_DIR / f'{name}_texts.json'

    if not pt_in.exists():
        print(f'[{name}] SKIP — {pt_in} not found')
        return
    if not json_path.exists():
        print(f'[{name}] SKIP — {json_path} not found')
        return

    print(f'[{name}] Loading scores from JSON ...', flush=True)
    t0 = time.time()
    scores_map = stream_idx_scores(json_path)
    print(f'  {len(scores_map):,} records with scores ({time.time()-t0:.1f}s)')

    print(f'[{name}] Loading {pt_in.name} ({pt_in.stat().st_size/1e9:.1f} GB) ...', flush=True)
    t0 = time.time()
    data = torch.load(pt_in, map_location='cpu', weights_only=False)
    cascades = data['cascades']
    print(f'  {len(cascades):,} cascades loaded ({time.time()-t0:.1f}s)')

    ZERO = [0.0] * 9
    n_ok = n_zero = 0
    for i, c in enumerate(cascades):
        if i in scores_map:
            c['burst_scores'] = torch.tensor(scores_map[i], dtype=torch.float32)
            n_ok += 1
        else:
            c['burst_scores'] = torch.tensor(ZERO, dtype=torch.float32)
            n_zero += 1

    total = n_ok + n_zero
    print(f'  scored={n_ok:,}/{total:,}  zero-filled={n_zero:,}  '
          f'({n_ok/total*100:.3f}% normal, {n_zero/total*100:.3f}% zero)')

    print(f'[{name}] Saving to {pt_out.name} ...', flush=True)
    t0 = time.time()
    torch.save(data, pt_out)
    size_gb = pt_out.stat().st_size / 1e9
    print(f'  Saved ({time.time()-t0:.1f}s)  {size_gb:.1f} GB')
    print(f'[{name}] Done.\n')


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else TARGETS
    for name in targets:
        merge(name)


if __name__ == '__main__':
    main()
