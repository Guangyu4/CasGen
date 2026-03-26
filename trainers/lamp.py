"""LAMP-CasGen: IFLTPP + EBM reranking + Qwen3-8B abductive reasoning.

Three-phase pipeline (per the LAMP NeurIPS-2023 paper, adapted for cascade
generation):

  Phase 1 — Base TPP:
    Train IFLTPP (BERT-conditioned cascade generator) on the training split.
    Auto-skipped if {outdir}/ifltpp_best.pt already exists.

  Phase 2 — EBM training (no LLM, training set only):
    For each training cascade, treat (BERT_emb, real_cascade_features) as
    the positive sample and (BERT_emb, features_from_random_other_cascade)
    as K-1 negative samples.  Train a bilinear MLP energy model with BCE loss.
    Auto-skipped if {outdir}/ebm_best.pt already exists.

  Phase 3 — Test-time LLM reranking:
    (a) Generate K candidate time-sequences per test cascade with IFLTPP.
    (b) Call Qwen3-8B (single GPU, via scripts/lamp_llm_infer.py) on the
        root-post texts to predict 4 cascade-dynamic features.
        Cached to {outdir}/llm_out.jsonl; reused if the file already exists.
    (c) Score each candidate:
            score = EBM_energy(bert, cand_feat) + λ * L2(cand_4d, llm_4d)
        Select the candidate with the lowest score.
    (d) Evaluate MMD / W1(l) / W1(t).
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from scipy.stats import wasserstein_distance

from .casflow import BertSeqDataset, collate_bert_seq
from .ifltpp import (
    IFLTPPSeqModel,
    _compute_inter_time_stats,
)
from .metrics import eval_metrics, print_and_save, to_numpy_seq

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent
_CASGEN    = _HERE.parent
_BATCHINF  = _CASGEN.parent / 'BatchInf'
_VLLM_PY   = _BATCHINF / 'vllm_env' / 'bin' / 'python'
_LLM_SCRIPT = _CASGEN / 'scripts' / 'lamp_llm_infer.py'

# ── LLM feature mappings ───────────────────────────────────────────────────────
_SPEED_MAP    = {'fast': 0.0, 'medium': 0.5, 'slow': 1.0}
_PATTERN_MAP  = {'burst': 0.0, 'sustained': 0.5, 'decaying': 1.0}
_DURATION_MAP = {'short': 0.0, 'medium': 0.5, 'long': 1.0}

_SYSTEM_PROMPT = (
    'You are a cascade dynamics analyst. '
    'Given a post, predict its information cascade pattern as JSON only. '
    'No explanation, no markdown, only the JSON object.'
)

_USER_TEMPLATE = (
    'Post: {text}\n\n'
    'Output a JSON object with exactly these keys:\n'
    '{{"expected_length": <integer, estimated number of replies>, '
    '"first_reply_time": "fast" | "medium" | "slow", '
    '"timing_pattern": "burst" | "sustained" | "decaying", '
    '"total_duration": "short" | "medium" | "long"}}'
)


# ── Cascade feature helpers ────────────────────────────────────────────────────

def _cascade_feat5(times, max_events: int = 500) -> np.ndarray:
    """5-dim feature vector for EBM: [norm_len, mean_gap, std_gap, t_first, t_last]."""
    times = np.asarray(times, dtype=np.float64)
    times = np.sort(times[times <= 1.0])
    n     = len(times)
    if n == 0:
        return np.zeros(5, dtype=np.float32)
    gaps = np.diff(times) if n > 1 else np.array([0.0])
    return np.array([
        n / max(max_events, 1),
        float(gaps.mean()),
        float(gaps.std())  if len(gaps) > 1 else 0.0,
        float(times[0]),
        float(times[-1]),
    ], dtype=np.float32)


def _cascade_feat4(feat5: np.ndarray) -> np.ndarray:
    """Map 5-dim EBM features to 4-dim LLM-comparable space.
    Dimensions: [norm_len, t_first(speed), std_gap(pattern), t_last(duration)]
    """
    return np.array([feat5[0], feat5[3], feat5[2], feat5[4]], dtype=np.float32)


def _parse_llm_feat(response: str, max_len: int = 500) -> np.ndarray | None:
    """Parse Qwen3-8B response → 4-dim feature vector. Returns None on failure."""
    try:
        m = re.search(r'\{[\s\S]*?\}', response)
        if not m:
            return None
        d = json.loads(m.group())
        el = float(d.get('expected_length', 10))
        return np.array([
            min(el, max_len) / max(max_len, 1),
            _SPEED_MAP.get(d.get('first_reply_time', 'medium'), 0.5),
            _PATTERN_MAP.get(d.get('timing_pattern', 'sustained'), 0.5),
            _DURATION_MAP.get(d.get('total_duration', 'medium'), 0.5),
        ], dtype=np.float32)
    except Exception:
        return None


# ── EBM model ──────────────────────────────────────────────────────────────────

class CascadeEBM(nn.Module):
    """Bilinear energy model: E(bert, feat) = -<MLP(bert), MLP(feat)>."""
    def __init__(self, bert_dim: int = 768, cascade_dim: int = 5, hidden: int = 128):
        super().__init__()
        self.bert_proj = nn.Sequential(
            nn.Linear(bert_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(cascade_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def energy(self, bert_emb: torch.Tensor, cascade_feat: torch.Tensor) -> torch.Tensor:
        """Lower energy = better match.  (B,) → scalar per sample."""
        h_b = self.bert_proj(bert_emb)
        h_f = self.feat_proj(cascade_feat)
        return -(h_b * h_f).sum(-1)   # (B,)

    def forward(self, bert_emb, cascade_feat):
        return self.energy(bert_emb, cascade_feat)


# ── LLM inference ──────────────────────────────────────────────────────────────

def _write_llm_inputs(texts, indices, inp_path: Path, max_chars: int = 4000):
    """Write JSONL prompts for Qwen3-8B."""
    with open(inp_path, 'w', encoding='utf-8') as f:
        for idx, text in zip(indices, texts):
            prompt = _USER_TEMPLATE.format(text=text[:max_chars].replace('\n', ' '))
            f.write(json.dumps({
                'id':     str(idx),
                'system': _SYSTEM_PROMPT,
                'prompt': prompt,
            }, ensure_ascii=False) + '\n')
    print(f'LLM input written: {inp_path}  ({len(texts)} items)')


def _run_llm(inp_path: Path, out_path: Path) -> bool:
    """Run single-GPU LLM inference via lamp_llm_infer.py subprocess."""
    if not _VLLM_PY.exists():
        print(f'[WARN] vllm_env not found at {_VLLM_PY}; skipping LLM step.')
        return False
    if not _LLM_SCRIPT.exists():
        print(f'[WARN] LLM script not found at {_LLM_SCRIPT}; skipping LLM step.')
        return False

    print(f'Running LLM inference (single GPU): {_LLM_SCRIPT.name}')
    print(f'  input:  {inp_path}')
    print(f'  output: {out_path}')

    ret = subprocess.run(
        [str(_VLLM_PY), str(_LLM_SCRIPT), str(inp_path), str(out_path)],
        check=False,
    )
    if ret.returncode != 0:
        print(f'[WARN] LLM inference exited with code {ret.returncode}')
        return False
    return True


def _load_llm_feats(out_path: Path, max_len: int = 500) -> dict:
    """Read LLM JSONL output → {id_str: 4-dim numpy array}."""
    feats = {}
    if not out_path.exists():
        return feats
    for line in out_path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            obj  = json.loads(line)
            feat = _parse_llm_feat(obj.get('response', ''), max_len)
            if feat is not None:
                feats[str(obj['id'])] = feat
        except Exception:
            pass
    return feats


# ── Phase 1: IFLTPP training ───────────────────────────────────────────────────

def _train_ifltpp(args, dataset, train_set, val_set, ckpt_path: str) -> IFLTPPSeqModel:
    """Train IFLTPP and save to ckpt_path.  Returns loaded best model."""
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_bert_seq, drop_last=True,
    )
    mean_log_tau, std_log_tau = _compute_inter_time_stats(dataset, train_set.indices)
    print(f'Inter-time stats: mean={mean_log_tau:.4f}  std={std_log_tau:.4f}')

    n_eval   = min(args.n_val_samples, len(val_set))
    val_bert = torch.stack([dataset.bert_embs[val_set.indices[i]] for i in range(n_eval)])
    ref_lens = [len(dataset.time_seqs[val_set.indices[i]]) for i in range(n_eval)]

    model = IFLTPPSeqModel(
        bert_dim=768,
        context_size=args.context_size,
        mark_emb_size=args.mark_emb_size,
        n_mix=args.n_mix,
        rnn_type=args.rnn_type,
        dropout=args.dropout,
    ).to(args.device)
    model.set_inter_time_stats(mean_log_tau, std_log_tau)
    print(f'IFLTPP params: {sum(p.numel() for p in model.parameters()):,}')

    optim     = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.max_steps, eta_min=args.lr * 0.05)
    best_wd   = float('inf')
    step      = 0
    train_it  = iter(train_loader)

    while step < args.max_steps:
        try:
            batch = next(train_it)
        except StopIteration:
            train_it = iter(train_loader)
            batch    = next(train_it)

        bert_emb, times, seq_lens = batch
        bert_emb  = bert_emb.to(args.device)
        times     = times.to(args.device)
        seq_lens  = seq_lens.to(args.device)

        model.train()
        loss = model.forward_train(bert_emb, times, seq_lens).mean()
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        sched.step()
        step += 1

        if step % 100 == 0:
            print(f'  [IFLTPP step {step:5d}] NLL={loss.item():.4f}')

        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                gen = model.generate(val_bert.to(args.device),
                                     max_len=args.max_events,
                                     stop_thresh=args.stop_thresh)
            gl  = [len(s) for s in gen]
            wd  = float(wasserstein_distance(
                np.array(gl, float) / max(np.mean(ref_lens), 1),
                np.array(ref_lens[:len(gl)], float) / max(np.mean(ref_lens), 1),
            ))
            print(f'  [IFLTPP eval] gen_mean={np.mean(gl):.1f}  W1={wd:.4f}')
            if wd < best_wd:
                best_wd = wd
                torch.save({
                    'step': step, 'model': model.state_dict(),
                    'args': vars(args), 'best_wd': best_wd,
                    'mean_log_tau': mean_log_tau, 'std_log_tau': std_log_tau,
                }, ckpt_path)
                print(f'  → ifltpp checkpoint saved  wd={best_wd:.4f}')

    print(f'IFLTPP training done. Best W1={best_wd:.4f}')
    return _load_ifltpp(ckpt_path, args)


def _load_ifltpp(ckpt_path: str, args) -> IFLTPPSeqModel:
    ckpt   = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    saved  = ckpt.get('args', {})
    model  = IFLTPPSeqModel(
        bert_dim=768,
        context_size=saved.get('context_size', args.context_size),
        mark_emb_size=saved.get('mark_emb_size', args.mark_emb_size),
        n_mix=saved.get('n_mix', args.n_mix),
        rnn_type=saved.get('rnn_type', args.rnn_type),
        dropout=saved.get('dropout', args.dropout),
    )
    model.load_state_dict(ckpt['model'])
    model.set_inter_time_stats(
        ckpt.get('mean_log_tau', 0.0),
        ckpt.get('std_log_tau',  1.0),
    )
    return model


# ── Phase 2: EBM training ──────────────────────────────────────────────────────

def _extract_all_feats(dataset, indices, max_events: int) -> torch.Tensor:
    """Extract 5-dim features for all cascades at given indices. Returns (N, 5)."""
    feats = []
    for i in indices:
        t = to_numpy_seq(dataset.time_seqs[i])
        feats.append(_cascade_feat5(t, max_events))
    return torch.tensor(np.stack(feats), dtype=torch.float32)


def _train_ebm(args, dataset, train_set, ckpt_path: str) -> CascadeEBM:
    """Train EBM on training set cascade features (no LLM, no generation)."""
    print('Extracting training cascade features...')
    train_idx   = list(train_set.indices)
    all_bert    = torch.stack([dataset.bert_embs[i] for i in train_idx])  # (N, 768)
    all_feats   = _extract_all_feats(dataset, train_idx, args.max_events)  # (N, 5)
    N           = len(train_idx)
    print(f'EBM training: {N} samples')

    ebm   = CascadeEBM(bert_dim=768, cascade_dim=5, hidden=args.ebm_hidden).to(args.device)
    optim = torch.optim.Adam(ebm.parameters(), lr=1e-3, weight_decay=1e-5)
    K     = args.n_candidates

    ebm.train()
    best_loss = float('inf')
    for ep in range(args.ebm_epochs):
        perm      = torch.randperm(N)
        ep_loss   = 0.0
        n_batches = 0
        for start in range(0, N, args.batch_size):
            idx = perm[start: start + args.batch_size]
            B   = len(idx)
            if B < 2:
                continue

            bert = all_bert[idx].to(args.device)   # (B, 768)
            pos  = all_feats[idx].to(args.device)  # (B, 5)  – real cascades

            # Build K-1 negatives: randomly permuted other cascades in batch
            neg_list = []
            for _ in range(K - 1):
                neg_idx  = torch.randperm(B)
                neg_list.append(all_feats[idx[neg_idx]].to(args.device))
            # shape: (B, K-1, 5)
            negs = torch.stack(neg_list, dim=1)

            # Positive energy (B,)
            e_pos = ebm(bert, pos)
            # Negative energies (B, K-1)
            e_neg = torch.stack([ebm(bert, negs[:, k, :]) for k in range(K - 1)], dim=1)

            # BCE: positive should have lower energy than each negative
            # Equivalent to: log σ(e_neg - e_pos)  → maximise, i.e. minimise -log σ(e_neg - e_pos)
            diff  = e_neg - e_pos.unsqueeze(1)                 # (B, K-1)
            loss  = F.binary_cross_entropy_with_logits(
                diff, torch.ones_like(diff)
            )

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ebm.parameters(), 1.0)
            optim.step()

            ep_loss   += loss.item()
            n_batches += 1

        avg = ep_loss / max(n_batches, 1)
        if (ep + 1) % 5 == 0:
            print(f'  [EBM epoch {ep+1:3d}/{args.ebm_epochs}] loss={avg:.4f}')
        if avg < best_loss:
            best_loss = avg
            torch.save({'model': ebm.state_dict(), 'args': vars(args)}, ckpt_path)

    print(f'EBM training done. Best loss={best_loss:.4f}')
    return _load_ebm(ckpt_path, args)


def _load_ebm(ckpt_path: str, args) -> CascadeEBM:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    saved = ckpt.get('args', {})
    ebm   = CascadeEBM(bert_dim=768, cascade_dim=5,
                        hidden=saved.get('ebm_hidden', args.ebm_hidden))
    ebm.load_state_dict(ckpt['model'])
    return ebm


# ── Phase 3 helpers ────────────────────────────────────────────────────────────

@torch.no_grad()
def _generate_k_candidates(model: IFLTPPSeqModel, bert_embs: torch.Tensor,
                            K: int, max_events: int, stop_thresh: float,
                            device: str) -> list[list[np.ndarray]]:
    """
    For each of N cascades, generate K candidate time-sequences.
    Returns list of N lists, each containing K numpy arrays.
    """
    model.eval()
    model.to(device)
    N = bert_embs.size(0)
    # candidates[n] = list of K arrays
    candidates = [[] for _ in range(N)]
    for k in range(K):
        torch.manual_seed(k * 1000 + 42)
        seqs = model.generate(bert_embs.to(device),
                               max_len=max_events, stop_thresh=stop_thresh)
        for n, s in enumerate(seqs):
            candidates[n].append(to_numpy_seq(s))
    return candidates


# ── Public interface ───────────────────────────────────────────────────────────

def add_args(parser):
    # Data / BERT (same as ifltpp)
    parser.add_argument('--data',         required=True)
    parser.add_argument('--bert_model',   default='pretrained/bert-base-chinese')
    parser.add_argument('--bert_device',  default='cpu')
    parser.add_argument('--bert_cache',   default='')
    parser.add_argument('--max_events',   type=int, default=500)
    parser.add_argument('--outdir',       default='runs_lamp/run')

    # IFLTPP hyper-params (mirrors ifltpp.add_args)
    parser.add_argument('--context_size',  type=int,   default=64)
    parser.add_argument('--mark_emb_size', type=int,   default=32)
    parser.add_argument('--n_mix',         type=int,   default=16)
    parser.add_argument('--rnn_type',      default='GRU', choices=['GRU', 'LSTM', 'RNN'])
    parser.add_argument('--dropout',       type=float, default=0.1)
    parser.add_argument('--stop_thresh',   type=float, default=0.5)
    parser.add_argument('--max_steps',     type=int,   default=30000)
    parser.add_argument('--batch_size',    type=int,   default=128)
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--weight_decay',  type=float, default=1e-5)
    parser.add_argument('--eval_every',    type=int,   default=1000)
    parser.add_argument('--n_val_samples', type=int,   default=500)

    # EBM hyper-params
    parser.add_argument('--ebm_hidden',  type=int,   default=128)
    parser.add_argument('--ebm_epochs',  type=int,   default=30)
    parser.add_argument('--n_candidates', type=int,  default=16)

    # LLM reranking
    parser.add_argument('--llm_weight',  type=float, default=0.3,
                        help='λ: weight of LLM alignment term in final score')
    parser.add_argument('--llm_inp',     default='',
                        help='Override path for LLM input JSONL')
    parser.add_argument('--llm_out',     default='',
                        help='Override path for LLM output JSONL (reuse if exists)')
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    ifltpp_ckpt = os.path.join(args.outdir, 'ifltpp_best.pt')
    ebm_ckpt    = os.path.join(args.outdir, 'ebm_best.pt')
    # best.pt = symlink-style sentinel pointing to ebm_ckpt (for test.py compat)
    best_pt     = os.path.join(args.outdir, 'best.pt')

    print(f'Loading dataset: {args.data}')
    dataset = BertSeqDataset(
        args.data, bert_name=args.bert_model,
        bert_device=args.bert_device, bert_cache=args.bert_cache,
        max_events=args.max_events,
    )
    n       = len(dataset)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    n_test  = n - n_train - n_val
    g       = torch.Generator().manual_seed(args.seed)
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=g)
    print(f'Split: train={n_train}  val={n_val}  test={n_test}')

    # ── Phase 1: IFLTPP ──────────────────────────────────────────────────────
    if os.path.exists(ifltpp_ckpt):
        print(f'[Phase 1] IFLTPP checkpoint found — skipping training.')
    else:
        print('[Phase 1] Training IFLTPP...')
        _train_ifltpp(args, dataset, train_set, val_set, ifltpp_ckpt)

    # ── Phase 2: EBM ─────────────────────────────────────────────────────────
    if os.path.exists(ebm_ckpt):
        print(f'[Phase 2] EBM checkpoint found — skipping training.')
    else:
        print('[Phase 2] Training EBM...')
        _train_ebm(args, dataset, train_set, ebm_ckpt)

    # Write best.pt sentinel (stores paths to both sub-checkpoints)
    torch.save({
        'ifltpp_ckpt': ifltpp_ckpt,
        'ebm_ckpt':    ebm_ckpt,
        'args':        vars(args),
    }, best_pt)
    print(f'Saved best.pt sentinel → {best_pt}')

    # ── Phase 3: test ─────────────────────────────────────────────────────────
    test(args)


def test(args):
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = getattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Resolve checkpoint paths
    best_pt  = os.path.join(args.outdir, 'best.pt')
    if os.path.exists(best_pt):
        sentinel    = torch.load(best_pt, map_location='cpu', weights_only=False)
        ifltpp_ckpt = sentinel.get('ifltpp_ckpt',
                                    os.path.join(args.outdir, 'ifltpp_best.pt'))
        ebm_ckpt    = sentinel.get('ebm_ckpt',
                                    os.path.join(args.outdir, 'ebm_best.pt'))
    else:
        ifltpp_ckpt = os.path.join(args.outdir, 'ifltpp_best.pt')
        ebm_ckpt    = os.path.join(args.outdir, 'ebm_best.pt')

    if not os.path.exists(ifltpp_ckpt):
        print(f'No IFLTPP checkpoint at {ifltpp_ckpt}; run train first.')
        return
    if not os.path.exists(ebm_ckpt):
        print(f'No EBM checkpoint at {ebm_ckpt}; run train first.')
        return

    # Load dataset (need raw texts for LLM prompts)
    print('Loading dataset for test...')
    max_events = getattr(args, 'max_events', 500)
    dataset = BertSeqDataset(
        args.data, bert_name=args.bert_model,
        bert_device=getattr(args, 'bert_device', 'cpu'),
        bert_cache=getattr(args, 'bert_cache', ''),
        max_events=max_events,
    )

    # Reconstruct test split
    n       = len(dataset)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    n_test  = n - n_train - n_val
    g       = torch.Generator().manual_seed(getattr(args, 'seed', 42))
    _, _, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=g)
    test_idx = list(test_set.indices)

    # Load raw texts for LLM prompts (re-read the .pt file)
    raw_data    = torch.load(args.data, map_location='cpu', weights_only=False)
    raw_cascades = raw_data['cascades']
    # Build text list aligned with dataset indices (same filtering as BertSeqDataset)
    _valid_texts = []
    for c in raw_cascades:
        t = c['times'][:max_events]
        if len(t) == 0:
            continue
        _valid_texts.append(c.get('text', ''))
    test_texts = [_valid_texts[i] for i in test_idx]
    ref_seqs   = [to_numpy_seq(dataset.time_seqs[i]) for i in test_idx]

    K             = getattr(args, 'n_candidates', 16)
    stop_thresh   = getattr(args, 'stop_thresh', 0.5)
    llm_weight    = getattr(args, 'llm_weight', 0.3)
    batch_size    = getattr(args, 'batch_size', 128)

    # ── (a) Generate K candidates per test cascade ────────────────────────────
    print(f'[Phase 3a] Generating {K} candidates for {len(test_idx)} test cascades...')
    tpp_model = _load_ifltpp(ifltpp_ckpt, args).to(device)
    tpp_model.eval()

    all_candidates = []   # list of N lists (each K numpy arrays)
    with torch.no_grad():
        for start in range(0, len(test_idx), batch_size):
            chunk_idx  = test_idx[start: start + batch_size]
            bert_chunk = torch.stack([dataset.bert_embs[i] for i in chunk_idx])
            chunk_cands = _generate_k_candidates(
                tpp_model, bert_chunk, K, max_events, stop_thresh, device)
            all_candidates.extend(chunk_cands)

    # Free GPU memory before launching vLLM
    del tpp_model
    torch.cuda.empty_cache()

    # ── (b) EBM scoring ───────────────────────────────────────────────────────
    print('[Phase 3b] Computing EBM scores...')
    ebm = _load_ebm(ebm_ckpt, args).to(device)
    ebm.eval()

    ebm_scores = []   # list of N arrays of shape (K,)
    with torch.no_grad():
        for n_idx, cands in enumerate(all_candidates):
            bert = dataset.bert_embs[test_idx[n_idx]].unsqueeze(0).expand(K, -1).to(device)
            feat = torch.tensor(
                np.stack([_cascade_feat5(c, max_events) for c in cands]),
                dtype=torch.float32,
            ).to(device)
            scores = ebm.energy(bert, feat).cpu().numpy()  # (K,)
            ebm_scores.append(scores)

    del ebm
    torch.cuda.empty_cache()

    # ── (c) LLM inference ─────────────────────────────────────────────────────
    llm_inp = Path(getattr(args, 'llm_inp', '') or
                   os.path.join(args.outdir, 'llm_inp.jsonl'))
    llm_out = Path(getattr(args, 'llm_out', '') or
                   os.path.join(args.outdir, 'llm_out.jsonl'))

    if llm_out.exists():
        print(f'[Phase 3c] LLM output found ({llm_out}), reusing.')
    else:
        print('[Phase 3c] Running Qwen3-8B LLM inference...')
        _write_llm_inputs(test_texts, test_idx, llm_inp)
        ok = _run_llm(llm_inp, llm_out)
        if not ok:
            print('[WARN] LLM inference failed; using EBM-only selection.')

    llm_feat_map = _load_llm_feats(llm_out, max_events)
    n_llm_parsed = sum(1 for i in test_idx if str(i) in llm_feat_map)
    print(f'LLM features parsed: {n_llm_parsed}/{len(test_idx)}')

    # ── (d) Combined scoring & selection ──────────────────────────────────────
    print('[Phase 3d] Selecting best candidates...')
    gen_seqs = []
    for n_idx, (cands, e_scores) in enumerate(zip(all_candidates, ebm_scores)):
        llm_feat = llm_feat_map.get(str(test_idx[n_idx]))

        if llm_feat is not None and llm_weight > 0:
            # LLM alignment: L2 distance between candidate 4d feat and LLM prediction
            cand_4d   = np.stack([_cascade_feat4(_cascade_feat5(c, max_events))
                                  for c in cands])            # (K, 4)
            llm_dist  = np.sum((cand_4d - llm_feat) ** 2, axis=1)  # (K,)
            final     = e_scores + llm_weight * llm_dist
        else:
            final = e_scores

        best_k   = int(np.argmin(final))
        gen_seqs.append(cands[best_k])

    m = eval_metrics(gen_seqs, ref_seqs)
    print_and_save(m, args.outdir)
    return m
