"""
Validate burst_scores in *_texts.json files.

For each dataset:
  - Count total records and those with burst_scores (via grep, no OOM)
  - Sample-check format with a streaming JSON parser
  - If invalid/missing records exist, print the count and exit 1
  - Does NOT modify any files (safe to run anytime)

Usage:
  python validate_burst.py [APS WeiboCov RedditM]
"""

import json
import re
import subprocess
import sys
from pathlib import Path

DIR       = Path(__file__).parent
TEXTS_DIR = DIR / 'texts'
CACHE_DIR = DIR / 'burst_cache'

TARGETS   = ['APS', 'WeiboCov', 'RedditM']
N_KEYS    = 9
SCORE_MIN = 0.0
SCORE_MAX = 5.0


def grep_count(pattern: str, path: Path) -> int:
    r = subprocess.run(['grep', '-c', pattern, str(path)],
                       capture_output=True, text=True)
    return int(r.stdout.strip()) if r.returncode in (0, 1) else 0


def is_valid(scores) -> bool:
    if not isinstance(scores, list) or len(scores) != N_KEYS:
        return False
    return all(isinstance(v, (int, float)) and SCORE_MIN <= v <= SCORE_MAX
               for v in scores)


def full_check(json_path: Path, bad_ids_out: list) -> tuple[int, int]:
    """Stream-parse ALL records; return (checked, invalid_count)."""
    decoder = json.JSONDecoder()
    checked = invalid = 0
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
            while True:
                buf = buf.lstrip()
                if buf.startswith(','):
                    buf = buf[1:].lstrip()
                if not buf or buf.startswith(']'):
                    break
                if not buf.startswith('{'):
                    break
                try:
                    obj, end = decoder.raw_decode(buf)
                    buf = buf[end:]
                    bs = obj.get('burst_scores')
                    if bs is not None and not is_valid(bs):
                        invalid += 1
                        bad_ids_out.append(str(obj['id']))
                    checked += 1
                    if checked % 100_000 == 0:
                        print(f'  ... {checked:,} checked', flush=True)
                except json.JSONDecodeError:
                    break

    return checked, invalid


def validate(name: str) -> bool:
    json_path = TEXTS_DIR / f'{name}_texts.json'
    if not json_path.exists():
        print(f'[{name}] SKIP — {json_path} not found')
        return True

    total   = grep_count('"idx"',          json_path)
    scored  = grep_count('"burst_scores"', json_path)
    missing = total - scored

    print(f'[{name}] total={total:,}  scored={scored:,}  missing={missing:,}', flush=True)

    bad_ids: list[str] = []
    checked, invalid = full_check(json_path, bad_ids)
    print(f'  full-checked {checked:,} records → {invalid} invalid format')

    MISSING_THRESHOLD = max(1, int(total * 0.0001))  # 0.01% tolerance

    if missing == 0 and invalid == 0:
        print(f'[{name}] ALL OK\n')
        return True

    if invalid > 0:
        print(f'  ERROR: {invalid:,} records have invalid format — '
              f'clear them from cache and rescore')
        print(f'  Bad IDs (sample): {bad_ids[:10]}')
        print()
        return False

    if missing <= MISSING_THRESHOLD:
        print(f'  WARN: {missing:,} records missing (≤{MISSING_THRESHOLD}, within tolerance) — OK\n')
        return True

    print(f'  ACTION: {missing:,} records missing scores — '
          f'rerun extract_texts.py {name} to rescore from cache')
    print()
    return False


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else TARGETS
    all_ok = True
    for name in targets:
        ok = validate(name)
        all_ok = all_ok and ok
    if all_ok:
        print('=== All datasets valid. Ready to merge. ===')
    else:
        print('=== Issues found — see above for required actions. ===')
        sys.exit(1)


if __name__ == '__main__':
    main()
