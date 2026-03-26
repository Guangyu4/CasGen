"""CTCP (adapted) trainer: sequence generation from BERT embeddings.

Adapted from: CTCP (AAAI 2023) - Continual Transfer Learning for Information
Cascade Popularity Prediction.

Original CTCP: heterogeneous temporal graph (user ↔ cascade interactions) →
GRU-based dynamic state updating + TimeDifferenceEncoder (cosine Fourier time
encoding) + temporal/structural LSTM/GNN aggregation → scalar count.
Requires DGL, user/cascade ID mappings, HGraph bookkeeping.

This adaptation preserves CTCP's two core contributions:
  1. TimeDifferenceEncoder: cos(W · Δt), W initialized as fixed log-scale
     frequencies (1/10^linspace(0,9,d)) — the key Fourier time encoding
  2. GRU-based sequence dynamics (same cell type as CTCP's state updater)

And replaces graph-dependent parts:
  - HGraph + DGL + dynamic user/cascade states → BERT projection
  - Temporal/structural aggregation module → autoregressive GRU decoder
  - Scalar count head → (delta_t, stop) head for sequence generation
  - Heterogeneous interactions → single BERT CLS embedding per cascade

Compared to CasFlow (VAE+NF+GRU) and CasCN (LSTM+interval decay):
  CTCP uses GRU + continuous cosine Fourier time encoding at each decode step.

Architecture:
  BERT emb (B,768) → Linear → GRU init hidden h_0
  Decoder GRU: input at step i = [last_delta_t | time_enc(t_accumulated)]
  time_enc: TimeDifferenceEncoder(Δt) = cos(W · Δt), W fixed log-scale
  head: Linear(h) → [delta_t_raw, stop_logit]
"""
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from scipy.stats import wasserstein_distance

from .casflow import BertSeqDataset, collate_bert_seq


# ---- CTCP core: TimeDifferenceEncoder (kept faithful to original) ----

class TimeDifferenceEncoder(nn.Module):
    """
    Maps a scalar time difference Δt → R^d via cos(W · Δt).
    W is initialized as fixed log-scale frequencies: 1/10^linspace(0,9,d).
    This is the key temporal encoding in CTCP; W is kept fixed (not trained)
    matching the original implementation.
    """
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self.w = nn.Linear(1, dimension)
        self.w.weight = nn.Parameter(
            (1.0 / 10 ** torch.linspace(0, 9, dimension))
            .float().reshape(dimension, 1),
            requires_grad=False,   # fixed frequencies, as in original CTCP
        )
        self.w.bias = nn.Parameter(torch.zeros(dimension), requires_grad=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) scalar time differences
        Returns:
            (B, dimension) cosine-encoded time vectors
        """
        return torch.cos(self.w(t.unsqueeze(1)))   # (B, dimension)


# ---- Model ----

class CTCPSeqModel(nn.Module):
    """
    Adapted CTCP for variable-length cascade sequence generation.

    Encoder : BERT emb (B,768) → Linear → GRU init hidden h_0
    Decoder : GRUCell; input at each step =
                [prev_delta_t (1) | time_enc(t_accumulated) (time_enc_dim)]
              → TimeDifferenceEncoder encodes accumulated time (core CTCP feature)
    Head    : Linear(h) → [delta_t_raw, stop_logit]
    """
    def __init__(self, bert_dim=768, hidden_dim=256,
                 rnn_units=128, time_enc_dim=16, dropout=0.1):
        super().__init__()
        self.rnn_units    = rnn_units
        self.time_enc_dim = time_enc_dim

        # encoder
        self.proj     = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.LayerNorm(hidden_dim),
        )
        self.init_h   = nn.Linear(hidden_dim, rnn_units)

        # core CTCP component: cosine Fourier time encoder
        self.time_enc = TimeDifferenceEncoder(time_enc_dim)

        # GRU decoder (same cell as CTCP's state updater)
        self.dec_gru  = nn.GRUCell(1 + time_enc_dim, rnn_units)
        self.dropout  = nn.Dropout(dropout)

        # output head
        self.dec_head = nn.Linear(rnn_units, 2)   # [delta_t_raw, stop_logit]

    def _step(self, inp_delta, t_acc, h):
        """
        One decoder step.
        inp_delta : (B, 1)  previous delta_t (teacher-forced or generated)
        t_acc     : (B,)    accumulated time so far
        h         : (B, rnn_units)
        Returns:
            h_new : (B, rnn_units)
            out   : (B, 2)   [delta_t_raw, stop_logit]
        """
        t_emb = self.time_enc(t_acc)              # (B, time_enc_dim)
        inp   = torch.cat([inp_delta, t_emb], dim=1)  # (B, 1+time_enc_dim)
        h_new = self.dec_gru(inp, h)
        out   = self.dec_head(self.dropout(h_new))
        return h_new, out

    # ---- training (teacher forcing) ----

    def forward_train(self, bert_emb, times, seq_lens):
        """
        Args:
            bert_emb : (B, 768)
            times    : (B, T_max) padded with zeros
            seq_lens : (B,)
        Returns:
            pred_delta  : (B, T_max+1)
            stop_logits : (B, T_max+1)
        """
        B = bert_emb.size(0)
        T = times.size(1)

        h = torch.tanh(self.init_h(self.proj(bert_emb)))  # (B, rnn_units)

        prev = torch.zeros(B, 1, device=times.device)
        gt_delta = torch.cat([prev, times], dim=1)
        gt_delta = gt_delta[:, 1:] - gt_delta[:, :-1]     # (B, T)

        inp   = torch.zeros(B, 1, device=times.device)
        t_acc = torch.zeros(B, device=times.device)
        pred_deltas, stop_logits = [], []

        for i in range(T + 1):
            h, out = self._step(inp, t_acc, h)
            pred_deltas.append(out[:, 0])
            stop_logits.append(out[:, 1])
            if i < T:
                inp   = gt_delta[:, i].unsqueeze(1)         # teacher force
                t_acc = t_acc + gt_delta[:, i].clamp(min=0)

        pred_delta  = torch.stack(pred_deltas,  dim=1)      # (B, T+1)
        stop_logits = torch.stack(stop_logits,  dim=1)      # (B, T+1)
        return pred_delta, stop_logits

    # ---- inference ----

    @torch.no_grad()
    def generate(self, bert_embs, max_len=500, stop_thresh=0.5):
        """
        Generate conditioned on bert_embs (B, 768).
        Returns list of (T_i,) tensors of absolute times.
        """
        B      = bert_embs.size(0)
        device = bert_embs.device

        h     = torch.tanh(self.init_h(self.proj(bert_embs)))
        inp   = torch.zeros(B, 1, device=device)
        t_acc = torch.zeros(B, device=device)
        alive = torch.ones(B, dtype=torch.bool, device=device)
        times_list = [[] for _ in range(B)]

        for _ in range(max_len):
            h, out = self._step(inp, t_acc, h)
            delta_t   = F.softplus(out[:, 0])
            stop_prob = torch.sigmoid(out[:, 1])

            t_acc = t_acc + delta_t
            stopped = stop_prob > stop_thresh

            for i in range(B):
                if alive[i] and not stopped[i]:
                    times_list[i].append(t_acc[i].item())

            alive = alive & ~stopped
            if not alive.any():
                break
            inp = delta_t.detach().unsqueeze(1)

        return [torch.tensor(ts, dtype=torch.float32) for ts in times_list]


# ---- Loss ----

def ctcp_loss(pred_delta, stop_logits, times, seq_lens):
    B, Tp1 = pred_delta.shape
    T      = Tp1 - 1
    device = pred_delta.device

    prev     = torch.zeros(B, 1, device=device)
    gt_delta = torch.cat([prev, times], dim=1)
    gt_delta = gt_delta[:, 1:] - gt_delta[:, :-1]

    stop_target = torch.zeros(B, Tp1, device=device)
    for i, L in enumerate(seq_lens.tolist()):
        stop_target[i, L] = 1.0

    mask = torch.zeros(B, T, device=device)
    for i, L in enumerate(seq_lens.tolist()):
        mask[i, :L] = 1.0

    delta_loss = ((F.softplus(pred_delta[:, :T]) - gt_delta) ** 2 * mask).sum() \
                 / mask.sum().clamp(min=1)
    stop_loss  = F.binary_cross_entropy_with_logits(
        stop_logits, stop_target, reduction='mean',
    )
    return delta_loss + stop_loss, delta_loss, stop_loss


def compute_w1(gen_lens, ref_lens):
    g = np.array(gen_lens, dtype=float)
    r = np.array(ref_lens, dtype=float)
    if len(g) == 0 or len(r) == 0:
        return 1.0
    mean_r = max(r.mean(), 1)
    return float(wasserstein_distance(g / mean_r, r / mean_r))


# ---- Public interface ----

def add_args(parser):
    parser.add_argument('--data', default='dataset/APS_burst.pt',
                        help='Processed *_burst.pt data path')
    parser.add_argument('--bert_model', default='pretrained/bert-base-chinese',
                        help='BERT model path or HuggingFace name')
    parser.add_argument('--bert_device', default='cpu',
                        help='Device for pre-computing BERT embeddings')
    parser.add_argument('--bert_cache', default='',
                        help='Path to cache BERT embeddings (empty = no cache)')
    parser.add_argument('--max_events', type=int, default=500)
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='BERT projection hidden dim')
    parser.add_argument('--rnn_units', type=int, default=128,
                        help='GRU hidden size')
    parser.add_argument('--time_enc_dim', type=int, default=16,
                        help='TimeDifferenceEncoder output dim (original CTCP default=8)')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--stop_thresh', type=float, default=0.5)
    parser.add_argument('--outdir', default='runs_ctcp')
    parser.add_argument('--max_steps', type=int, default=30000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Original CTCP default lr=1e-4')
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--n_val_samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    print(f'Model: CTCP  data={args.data}')

    print('Loading dataset...')
    dataset = BertSeqDataset(
        args.data,
        bert_name=args.bert_model,
        bert_device=args.bert_device,
        bert_cache=args.bert_cache,
        max_events=args.max_events,
    )

    n       = len(dataset)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    n_test  = n - n_train - n_val
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_bert_seq, drop_last=True,
    )

    ref_lens = [len(dataset.time_seqs[i]) for i in val_set.indices]
    n_eval   = min(args.n_val_samples, len(val_set))
    val_bert = torch.stack(
        [dataset.bert_embs[val_set.indices[i]] for i in range(n_eval)]
    )
    print(f'Data: {n} cascades  train={n_train} val={n_val} test={n_test}')
    print(f'Ref lengths: mean={np.mean(ref_lens):.1f}  '
          f'median={np.median(ref_lens):.0f}')

    model = CTCPSeqModel(
        bert_dim=768,
        hidden_dim=args.hidden_dim,
        rnn_units=args.rnn_units,
        time_enc_dim=args.time_enc_dim,
        dropout=args.dropout,
    ).to(args.device)
    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.lr * 0.05,
    )

    step      = 0
    best_wd   = float('inf')
    train_iter = iter(train_loader)

    print('Training...')
    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        bert_emb, times, seq_lens = batch
        bert_emb  = bert_emb.to(args.device)
        times     = times.to(args.device)
        seq_lens  = seq_lens.to(args.device)

        model.train()
        pred_delta, stop_logits = model.forward_train(bert_emb, times, seq_lens)
        loss, delta_l, stop_l   = ctcp_loss(pred_delta, stop_logits,
                                             times, seq_lens)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % 100 == 0:
            print(f'[Step {step:5d}] loss={loss.item():.4f}  '
                  f'delta={delta_l.item():.4f}  stop={stop_l.item():.4f}')

        if step % args.eval_every == 0:
            model.eval()
            gen_seqs = model.generate(
                val_bert[:n_eval].to(args.device),
                max_len=args.max_events,
                stop_thresh=args.stop_thresh,
            )
            gen_lens = [len(s) for s in gen_seqs]
            wd = compute_w1(gen_lens, ref_lens[:n_eval])
            print(f'  [Eval] gen_mean={np.mean(gen_lens):.1f}  '
                  f'gen_median={np.median(gen_lens):.0f}  '
                  f'ref_mean={np.mean(ref_lens):.1f}  W1={wd:.4f}')

            if wd < best_wd:
                best_wd = wd
                torch.save({
                    'step': step, 'model': model.state_dict(),
                    'args': vars(args), 'best_wd': best_wd,
                    'stats': dataset.stats,
                }, os.path.join(args.outdir, 'best.pt'))

    print(f'Training complete. Best W1: {best_wd:.4f}')
    test(args)


def test(args):
    from .metrics import eval_metrics, print_and_save, get_test_split, to_numpy_seq
    torch.manual_seed(args.seed)

    ckpt_path = os.path.join(args.outdir, 'best.pt')
    if not os.path.exists(ckpt_path):
        print(f'No checkpoint at {ckpt_path}, skipping test.')
        return None

    print('Loading dataset for test...')
    dataset = BertSeqDataset(
        args.data, bert_name=args.bert_model,
        bert_device=args.bert_device, bert_cache=args.bert_cache,
        max_events=args.max_events,
    )
    test_set = get_test_split(dataset, args)
    ref_seqs  = [to_numpy_seq(dataset.time_seqs[i]) for i in test_set.indices]

    print('Loading checkpoint...')
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    saved = ckpt.get('args', {})
    model = CTCPSeqModel(
        bert_dim=768,
        hidden_dim=saved.get('hidden_dim', args.hidden_dim),
        rnn_units=saved.get('rnn_units', args.rnn_units),
        time_enc_dim=saved.get('time_enc_dim', args.time_enc_dim),
        dropout=saved.get('dropout', args.dropout),
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    print(f'Generating for {len(ref_seqs)} test cascades...')
    gen_seqs = []
    with torch.no_grad():
        for start in range(0, len(test_set), args.batch_size):
            idx  = test_set.indices[start: start + args.batch_size]
            embs = torch.stack([dataset.bert_embs[i] for i in idx]).to(args.device)
            seqs = model.generate(embs, max_len=args.max_events,
                                  stop_thresh=args.stop_thresh)
            gen_seqs.extend(to_numpy_seq(s) for s in seqs)

    m = eval_metrics(gen_seqs, ref_seqs)
    print_and_save(m, args.outdir)
    return m
