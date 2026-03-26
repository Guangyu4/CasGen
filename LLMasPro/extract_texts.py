"""Score burst potential for texts/*.json datasets using local Qwen3-8B (BatchInf)."""
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

DIR       = Path(__file__).parent
TEXTS_DIR = DIR / 'texts'
CACHE_DIR = DIR / 'burst_cache'
LOG_DIR   = DIR.parent / 'logs'
BATCH_DIR = DIR.parent.parent / 'BatchInf'
PYTHON    = BATCH_DIR / 'vllm_env' / 'bin' / 'python'
PRO_MD    = DIR / 'Pro.md'

TARGETS     = ['APS', 'WeiboCov', 'RedditM']
MAX_RETRIES = 3
SAVE_EVERY  = 40_000   # flush scores to JSON every N new output lines


def setup_logger(name: str) -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = LOG_DIR / f'score_burst_{name}_{ts}.log'

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s  %(levelname)-7s  %(message)s', datefmt='%H:%M:%S')
    fh  = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(fmt)
    ch  = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f'Log file: {log_path}')
    return logger

FLAT_KEYS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for ln in path.read_text().splitlines() if ln.strip())


def _flush_scores(out_path: Path, records: list, log) -> int:
    id_map = {str(r['id']): r for r in records}
    updated = 0
    for line in out_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            rid = str(obj['id'])
            if rid in id_map and 'burst_scores' not in id_map[rid]:
                scores = parse_scores(obj['response'])
                if scores is not None:
                    id_map[rid]['burst_scores'] = scores
                    updated += 1
        except Exception as e:
            log.warning(f'flush error: {e}')
    return updated


def _write_json(path: Path, records: list) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2))


def load_system_prompt() -> str:
    raw = PRO_MD.read_text()
    return re.split(r'\n# Input', raw)[0].strip()


def parse_scores(response: str) -> list[float] | None:
    try:
        m = re.search(r'\{[\s\S]*\}', response)
        if not m:
            return None
        data = json.loads(m.group())
        # flatten all nested dicts into one lookup
        flat: dict = {}
        for v in data.values():
            if isinstance(v, dict):
                flat.update(v)
            elif isinstance(v, (int, float)):
                flat.update(data)
                break
        scores = []
        for key in FLAT_KEYS:
            if key not in flat:
                return None
            val = flat[key]
            scores.append(float(val['score'] if isinstance(val, dict) else val))
        return scores
    except Exception:
        return None


def run(name: str):
    log = setup_logger(name)
    json_path = TEXTS_DIR / f'{name}_texts.json'
    inp_path  = CACHE_DIR / f'{name}_inp.jsonl'
    out_path  = CACHE_DIR / f'{name}_out.jsonl'

    if not json_path.exists():
        log.warning(f'[SKIP] {json_path} not found')
        return

    t_start = time.time()
    records = json.loads(json_path.read_text())
    log.info(f'{name}: {len(records):,} records loaded')

    # resume: collect already-scored ids from output cache
    done_ids: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                try:
                    done_ids.add(str(json.loads(line)['id']))
                except Exception:
                    pass

    already_in_json = {str(r['id']) for r in records if 'burst_scores' in r}
    done_ids |= already_in_json

    todo = [r for r in records if str(r['id']) not in done_ids]
    if not todo:
        log.info('All already scored.')
    else:
        log.info(f'Scoring {len(todo):,} items  (resuming {len(done_ids):,} done)')
        system_prompt = load_system_prompt()
        id_to_record  = {str(r['id']): r for r in todo}
        CACHE_DIR.mkdir(exist_ok=True)

        pending_ids = set(id_to_record)
        for attempt in range(1, MAX_RETRIES + 1):
            if attempt > 1 and out_path.exists():
                kept = [ln for ln in out_path.read_text().splitlines()
                        if ln.strip() and str(json.loads(ln)['id']) not in pending_ids]
                out_path.write_text('\n'.join(kept) + ('\n' if kept else ''))

            MAX_TEXT_CHARS = 6000  # ~1500 tokens, leaves room for system prompt + output
            with open(inp_path, 'w', encoding='utf-8') as f:
                for rid in pending_ids:
                    r = id_to_record[rid]
                    raw = r.get('text', '')[:MAX_TEXT_CHARS]
                    text = raw.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                    user_msg = f'{{\n  "post_id": "{r["id"]}",\n  "text": "{text}"\n}}'
                    f.write(json.dumps({'id': r['id'], 'system': system_prompt, 'prompt': user_msg}) + '\n')

            log.info(f'Attempt {attempt}/{MAX_RETRIES}: submitting {len(pending_ids):,} items to batch_infer...')
            t0 = time.time()
            proc = subprocess.Popen(
                [str(PYTHON), str(BATCH_DIR / 'batch_infer.py'), str(inp_path), str(out_path)],
            )
            saved_lines = _count_lines(out_path)
            while proc.poll() is None:
                time.sleep(30)
                cur_lines = _count_lines(out_path)
                new_lines  = cur_lines - saved_lines
                if new_lines >= SAVE_EVERY:
                    n = _flush_scores(out_path, records, log)
                    log.info(f'  Intermediate save: {n:,} new scores flushed '
                             f'({cur_lines:,} lines in cache)')
                    _write_json(json_path, records)
                    saved_lines = cur_lines
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, proc.args)
            log.info(f'Attempt {attempt} finished in {time.time() - t0:.1f}s')

            still_failing: set[str] = set()
            if out_path.exists():
                for line in out_path.read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        rid = str(obj['id'])
                        if rid in pending_ids and parse_scores(obj['response']) is None:
                            still_failing.add(rid)
                            log.debug(f'Parse failed ({rid}): {obj["response"][:120]}')
                    except Exception:
                        pass

            n_ok = len(pending_ids) - len(still_failing)
            log.info(f'Attempt {attempt}: {n_ok:,} parsed OK, {len(still_failing):,} failed')
            pending_ids = still_failing
            if not pending_ids:
                break
            log.warning(f'{len(pending_ids):,} items failed parsing — retrying...')

        if pending_ids:
            log.error(f'{len(pending_ids):,} items failed after {MAX_RETRIES} attempts, skipping.')

    # parse output and attach scores
    score_map: dict[str, list[float]] = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                obj    = json.loads(line)
                scores = parse_scores(obj['response'])
                if scores is not None:
                    score_map[str(obj['id'])] = scores
            except Exception as e:
                log.warning(f'Line parse error: {e}')

    updated = _flush_scores(out_path, records, log)
    _write_json(json_path, records)
    elapsed = time.time() - t_start
    log.info(f'Attached {updated:,} new scores -> {json_path}  (total {elapsed:.1f}s)')


def flush_only(name: str):
    """Re-read cache and write burst_scores into JSON without re-scoring."""
    log = setup_logger(name)
    json_path = TEXTS_DIR / f'{name}_texts.json'
    out_path  = CACHE_DIR / f'{name}_out.jsonl'
    if not json_path.exists():
        log.warning(f'[SKIP] {json_path} not found')
        return
    t_start = time.time()
    records = json.loads(json_path.read_text())
    log.info(f'{name}: {len(records):,} records loaded')
    n = _flush_scores(out_path, records, log)
    _write_json(json_path, records)
    log.info(f'Flushed {n:,} new scores -> {json_path}  ({time.time()-t_start:.1f}s)')


def main():
    args = sys.argv[1:]
    if args and args[0] == '--flush-only':
        targets = args[1:] if len(args) > 1 else TARGETS
        for name in targets:
            flush_only(name)
    else:
        targets = args if args else TARGETS
        for name in targets:
            run(name)
    print('Done.')


if __name__ == '__main__':
    main()
