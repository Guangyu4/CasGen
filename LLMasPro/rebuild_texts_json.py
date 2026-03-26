"""
Rebuild *_texts.json from the original .pt file + burst_cache/*_out.jsonl.

Used when the _texts.json was accidentally truncated.
Reads cascades from pt (to get idx/id/text), reads cache for burst_scores,
writes a fresh _texts.json.

Usage:
  python rebuild_texts_json.py WeiboCov
"""

import json
import sys
import time
from pathlib import Path

import torch

DIR       = Path(__file__).parent
TEXTS_DIR = DIR / 'texts'
CACHE_DIR = DIR / 'burst_cache'
DATASET   = DIR.parent / 'dataset'


def load_cache_scores(cache_path: Path) -> dict[str, list]:
    """Read burst_cache/*_out.jsonl, return {id_str: [9 floats]}."""
    import re
    scores = {}
    if not cache_path.exists():
        return scores

    FLAT_KEYS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

    def parse(response: str):
        try:
            m = re.search(r'\{[\s\S]*\}', response)
            if not m:
                return None
            data = json.loads(m.group())
            flat: dict = {}
            for v in data.values():
                if isinstance(v, dict):
                    flat.update(v)
                elif isinstance(v, (int, float)):
                    flat.update(data)
                    break
            result = []
            for key in FLAT_KEYS:
                if key not in flat:
                    return None
                val = flat[key]
                result.append(float(val['score'] if isinstance(val, dict) else val))
            return result
        except Exception:
            return None

    for line in cache_path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            sc = parse(obj['response'])
            if sc is not None:
                scores[str(obj['id'])] = sc
        except Exception:
            pass
    return scores


def rebuild(name: str) -> None:
    pt_path    = DATASET / f'{name}.pt'
    cache_path = CACHE_DIR / f'{name}_out.jsonl'
    json_path  = TEXTS_DIR / f'{name}_texts.json'

    print(f'[{name}] Loading cache scores ...', flush=True)
    t0 = time.time()
    scores_map = load_cache_scores(cache_path)
    print(f'  {len(scores_map):,} scores loaded ({time.time()-t0:.1f}s)')

    print(f'[{name}] Loading pt ({pt_path.stat().st_size/1e9:.1f} GB) ...', flush=True)
    t0 = time.time()
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    cascades = data['cascades']
    print(f'  {len(cascades):,} cascades ({time.time()-t0:.1f}s)')

    print(f'[{name}] Building records ...', flush=True)
    records = []
    missing = 0
    for idx, c in enumerate(cascades):
        rec = {'idx': idx, 'id': idx, 'text': c.get('text', '')}
        sc = scores_map.get(str(idx))
        if sc is not None:
            rec['burst_scores'] = sc
        else:
            missing += 1
        records.append(rec)

    if missing:
        print(f'  WARNING: {missing:,} cascades have no burst_scores')

    print(f'[{name}] Writing {json_path.name} ...', flush=True)
    t0 = time.time()
    json_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'  Done ({time.time()-t0:.1f}s)  scored={len(records)-missing:,}/{len(records):,}\n')


def main():
    targets = sys.argv[1:] if sys.argv[1:] else ['WeiboCov']
    for name in targets:
        rebuild(name)


if __name__ == '__main__':
    main()
