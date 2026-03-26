"""Build APS.pt dataset from APS abstract JSON files and citation CSV.

Memory optimizations:
  - CSR numpy arrays for adjacency list (no Python refcount COW on fork)
  - MAX_EVENTS cap to prevent huge cascades from a handful of highly-cited papers
  - Integer-indexed nodes; numpy timestamp array for fast date comparison
  - fork-based multiprocessing (globals shared as numpy arrays → true COW sharing)
"""
import os
import json
import torch
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import datetime
from collections import Counter, deque
from tqdm import tqdm

ABS_DIR  = os.path.join(os.path.dirname(__file__), 'data', 'APS', 'aps-dataset-abs')
CIT_CSV  = os.path.join(os.path.dirname(__file__), 'data', 'APS', 'aps-dataset-citations-2022.csv')
OUT_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'APS.pt')

MIN_CASCADE   = 2
MIN_CHAR_FREQ = 2
MAX_DEPTH     = 10
MAX_EVENTS    = 5000   # cap per cascade; prevents memory explosion from highly-cited papers
N_WORKERS     = 4

# ---------------------------------------------------------------------------
# Globals shared across fork workers via numpy arrays (no Python COW overhead)
# ---------------------------------------------------------------------------
_adj_ptr   = None   # np.int32[N+1]  — CSR row pointers
_adj_data  = None   # np.int32[E]    — CSR column indices (citing paper per cited paper)
_dates_ts  = None   # np.float64[N]  — UNIX timestamps
_abstracts = None   # list[str]      — one abstract per paper index


# ---------------------------------------------------------------------------
# Worker: BFS for a chunk of root indices, returns raw lists (no tensors)
# ---------------------------------------------------------------------------

def _bfs_worker(root_idxs):
    results = []
    for root_idx in root_idxs:
        root_ts = _dates_ts[root_idx]

        visited   = {root_idx}
        parent    = {}
        depth_map = {root_idx: 0}
        queue     = deque([root_idx])
        events    = []

        while queue and len(events) < MAX_EVENTS:
            cur       = queue.popleft()
            cur_depth = depth_map[cur]
            if cur_depth >= MAX_DEPTH:
                continue
            # iterate CSR slice
            for pos in range(_adj_ptr[cur], _adj_ptr[cur + 1]):
                child = int(_adj_data[pos])
                if child not in visited and _dates_ts[child] > root_ts:
                    visited.add(child)
                    parent[child]    = cur
                    depth_map[child] = cur_depth + 1
                    queue.append(child)
                    events.append(child)
                    if len(events) >= MAX_EVENTS:
                        break

        if len(events) < MIN_CASCADE:
            continue

        events.sort(key=lambda i: _dates_ts[i])
        last_sec = float(_dates_ts[events[-1]] - root_ts)
        t_max    = last_sec if last_sec > 0 else 1.0

        times, depths, parent_times = [], [], []
        for idx in events:
            t = float((_dates_ts[idx] - root_ts) / t_max)
            times.append(t)
            depths.append(int(depth_map[idx]))
            par = parent[idx]
            parent_times.append(
                0.0 if par == root_idx
                else float((_dates_ts[par] - root_ts) / t_max)
            )

        results.append({
            'times':        times,
            'depths':       depths,
            'parent_times': parent_times,
            'text':         _abstracts[root_idx],
            't_max':        t_max,
        })
    return results


# ---------------------------------------------------------------------------
# Load abs papers sequentially (avoid GPFS multiprocess metadata storm)
# ---------------------------------------------------------------------------

def load_papers(abs_dir):
    fnames = [f for f in os.listdir(abs_dir) if f.endswith('.json')]
    papers = {}
    for fname in tqdm(fnames, desc='abs JSON'):
        path = os.path.join(abs_dir, fname)
        try:
            with open(path) as f:
                d = json.load(f)
            doi      = d['identifiers']['doi']
            date_str = d.get('date', '')
            abstract = d.get('abstract', '')
            if date_str and abstract:
                papers[doi] = {
                    'ts':       datetime.strptime(date_str, '%Y-%m-%d').timestamp(),
                    'abstract': abstract,
                }
        except Exception:
            pass
    print(f'  Loaded {len(papers):,} papers')
    return papers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _adj_ptr, _adj_data, _dates_ts, _abstracts

    # ---- Step 1: load papers ----
    papers     = load_papers(ABS_DIR)
    doi_list   = list(papers.keys())
    doi_to_idx = {doi: i for i, doi in enumerate(doi_list)}
    N          = len(doi_list)

    _dates_ts  = np.array([papers[doi]['ts'] for doi in doi_list], dtype=np.float64)
    _abstracts = [papers[doi]['abstract'] for doi in doi_list]
    del papers

    # ---- Step 2: build CSR adjacency (cited_idx -> [citing_idx]) ----
    print('Reading citation CSV ...')
    df   = pd.read_csv(CIT_CSV, dtype=str)
    mask = df['citing_doi'].isin(doi_to_idx) & df['cited_doi'].isin(doi_to_idx)
    df   = df[mask].reset_index(drop=True)
    print(f'  Kept {len(df):,} edges (both endpoints in abs)')

    cited_idxs  = np.array([doi_to_idx[d] for d in df['cited_doi']],  dtype=np.int32)
    citing_idxs = np.array([doi_to_idx[d] for d in df['citing_doi']], dtype=np.int32)
    del df, doi_to_idx

    # sort by cited to build CSR
    order      = np.argsort(cited_idxs, kind='stable')
    cited_s    = cited_idxs[order]
    citing_s   = citing_idxs[order]

    _adj_ptr  = np.zeros(N + 1, dtype=np.int32)
    np.add.at(_adj_ptr[1:], cited_s, 1)
    np.cumsum(_adj_ptr, out=_adj_ptr)
    _adj_data = citing_s.astype(np.int32)
    del cited_idxs, citing_idxs, cited_s, citing_s, order

    root_idxs = list(np.where(_adj_ptr[1:] > _adj_ptr[:N])[0])
    print(f'  Candidate roots: {len(root_idxs):,}')

    # ---- Step 3: parallel BFS via fork ----
    chunks = [root_idxs[i::N_WORKERS] for i in range(N_WORKERS)]
    print(f'Starting BFS with {N_WORKERS} workers '
          f'(MAX_DEPTH={MAX_DEPTH}, MAX_EVENTS={MAX_EVENTS}) ...')

    ctx = mp.get_context('fork')
    with ctx.Pool(N_WORKERS) as pool:
        all_results = pool.map(_bfs_worker, chunks)

    cascades_raw = [c for batch in all_results for c in batch]
    print(f'  Cascades collected: {len(cascades_raw):,}')
    del all_results

    # ---- Step 4: convert to tensors ----
    print('Converting to tensors ...')
    cascades = [
        {
            'times':        torch.tensor(c['times'],        dtype=torch.float32),
            'depths':       torch.tensor(c['depths'],        dtype=torch.long),
            'parent_times': torch.tensor(c['parent_times'], dtype=torch.float32),
            'text':         c['text'],
            't_max':        c['t_max'],
        }
        for c in cascades_raw
    ]
    del cascades_raw

    # ---- Step 5: vocab ----
    print('Building vocabulary ...')
    counter = Counter()
    for c in cascades:
        counter.update(c['text'])
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for ch, freq in counter.most_common():
        if freq >= MIN_CHAR_FREQ:
            vocab[ch] = len(vocab)
    print(f'  Vocab size: {len(vocab):,}')

    # ---- Step 6: stats ----
    all_depths = torch.cat([c['depths'] for c in cascades])
    all_lens   = [len(c['times']) for c in cascades]
    stats = {
        'D_max':      int(all_depths.max().item()),
        'n_max':      max(all_lens),
        'n_cascades': len(cascades),
        'avg_len':    sum(all_lens) / len(all_lens),
    }
    print(f'  Stats: {stats}')

    # ---- Step 7: save ----
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save({'cascades': cascades, 'vocab': vocab, 'stats': stats}, OUT_PATH)
    print(f'Saved -> {OUT_PATH}')


if __name__ == '__main__':
    main()
