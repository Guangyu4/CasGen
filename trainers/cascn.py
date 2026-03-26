"""CasCN (adapted) trainer: sequence generation from BERT embeddings.

Adapted from: CasCN (WWW 2019) - Information Cascades Prediction with Graph
Attention Network.

Original CasCN: sparse graph adjacency matrices → Graph Convolutional LSTM
(Chebyshev spectral conv in each LSTM gate) + time-interval decay weights
→ scalar count (TF1 implementation).

This adaptation keeps CasCN's distinguishing components:
  - LSTM decoder (CasFlow uses GRU; CasCN specifically uses LSTM)
  - Learned time decay applied to hidden states during decoding

And replaces graph-dependent parts:
  - Graph preprocessing + Chebyshev Laplacian → BERT projection
  - Scalar count head → autoregressive LSTM decoder for sequence generation
  - TF1 session → PyTorch

Compared to CasFlow (VAE + NF + GRU), CasCN is deterministic:
  BERT emb → projection → LSTM decoder with time decay → (delta_t, stop) per step
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


# ---- Model ----

class CasCNSeqModel(nn.Module):
    """
    Adapted CasCN for variable-length cascade sequence generation.

    Encoder:  BERT emb (B,768) → projection → LSTM init state (c_0, h_0)
    Decoder:  LSTM (teacher forcing / autoregressive) with time-decay on hidden
    Head:     Linear → (delta_t_raw, stop_logit) per step
    """
    def __init__(self, bert_dim=768, hidden_dim=256, rnn_units=128,
                 n_time_bins=6):
        super().__init__()
        self.rnn_units = rnn_units
        self.n_time_bins = n_time_bins

        # encoder: project BERT emb → LSTM init states
        self.proj     = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )
        self.init_c   = nn.Linear(hidden_dim, rnn_units)
        self.init_h   = nn.Linear(hidden_dim, rnn_units)

        # LSTM decoder (CasCN uses LSTM, not GRU)
        self.dec_lstm = nn.LSTMCell(1, rnn_units)

        # time decay: learned per-interval weights (original CasCN insight)
        # intervals are uniform bins over [0, 1]
        self.time_weights = nn.Parameter(torch.ones(n_time_bins))

        # output head
        self.dec_head = nn.Linear(rnn_units, 2)  # [delta_t_raw, stop_logit]

    def _time_decay(self, t_accumulated):
        """
        Compute per-sample scalar decay weight from accumulated time t in [0,1].
        Original CasCN: one-hot encode time interval → dot with learned weights.
        Args:
            t_accumulated: (B,) tensor, values in [0, 1]
        Returns:
            decay: (B,) tensor
        """
        # bin index in [0, n_time_bins-1]
        idx = (t_accumulated * self.n_time_bins).long().clamp(0, self.n_time_bins - 1)
        w = F.softplus(self.time_weights)     # (n_time_bins,) positive weights
        return w[idx]                          # (B,)

    def _apply_decay(self, h, t_accumulated):
        """Scale hidden state by time decay. Shape: h (B, rnn_units)."""
        decay = self._time_decay(t_accumulated).unsqueeze(1)  # (B, 1)
        return h * decay

    # ---- training forward (teacher forcing) ----

    def forward_train(self, bert_emb, times, seq_lens):
        """
        Args:
            bert_emb : (B, 768)
            times    : (B, T_max) padded with zeros
            seq_lens : (B,) actual lengths
        Returns:
            pred_delta  : (B, T_max+1)
            stop_logits : (B, T_max+1)
        """
        B = bert_emb.size(0)
        T = times.size(1)

        enc = self.proj(bert_emb)
        c = torch.tanh(self.init_c(enc))
        h = torch.tanh(self.init_h(enc))

        # ground truth deltas
        prev = torch.zeros(B, 1, device=times.device)
        gt_delta = torch.cat([prev, times], dim=1)
        gt_delta = gt_delta[:, 1:] - gt_delta[:, :-1]   # (B, T)

        inp = torch.zeros(B, 1, device=times.device)
        t_acc = torch.zeros(B, device=times.device)      # accumulated time
        pred_deltas, stop_logits = [], []

        for i in range(T + 1):
            h, c = self.dec_lstm(inp, (h, c))
            h_dec = self._apply_decay(h, t_acc)          # time decay
            out = self.dec_head(h_dec)                   # (B, 2)
            pred_deltas.append(out[:, 0])
            stop_logits.append(out[:, 1])

            if i < T:
                inp = gt_delta[:, i].unsqueeze(1)        # teacher force
                t_acc = t_acc + gt_delta[:, i].clamp(min=0)

        pred_delta  = torch.stack(pred_deltas,  dim=1)   # (B, T+1)
        stop_logits = torch.stack(stop_logits,  dim=1)   # (B, T+1)
        return pred_delta, stop_logits

    # ---- inference ----

    @torch.no_grad()
    def generate(self, bert_embs, max_len=500, stop_thresh=0.5):
        """
        Generate sequences conditioned on bert_embs (B, 768).
        Returns list of (T_i,) tensors of absolute times.
        """
        B = bert_embs.size(0)
        device = bert_embs.device

        enc = self.proj(bert_embs)
        c = torch.tanh(self.init_c(enc))
        h = torch.tanh(self.init_h(enc))

        inp   = torch.zeros(B, 1, device=device)
        t_acc = torch.zeros(B, device=device)
        alive = torch.ones(B, dtype=torch.bool, device=device)
        times_list = [[] for _ in range(B)]

        for _ in range(max_len):
            h, c = self.dec_lstm(inp, (h, c))
            h_dec = self._apply_decay(h, t_acc)
            out = self.dec_head(h_dec)
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


# ---- Loss (identical structure to casflow) ----

def cascn_loss(pred_delta, stop_logits, times, seq_lens):
    B, Tp1 = pred_delta.shape
    T = Tp1 - 1
    device = pred_delta.device

    prev = torch.zeros(B, 1, device=device)
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
        stop_logits, stop_target, reduction='mean'
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
                        help='Projection hidden dim (before LSTM init)')
    parser.add_argument('--rnn_units', type=int, default=128,
                        help='LSTM hidden size')
    parser.add_argument('--n_time_bins', type=int, default=6,
                        help='Number of time-decay interval bins (original CasCN=6)')
    parser.add_argument('--stop_thresh', type=float, default=0.5)
    parser.add_argument('--outdir', default='runs_cascn')
    parser.add_argument('--max_steps', type=int, default=30000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--n_val_samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    print(f'Model: CASCN  data={args.data}')

    print('Loading dataset...')
    dataset = BertSeqDataset(
        args.data,
        bert_name=args.bert_model,
        bert_device=args.bert_device,
        bert_cache=args.bert_cache,
        max_events=args.max_events,
    )

    n = len(dataset)
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
    # val BERT embeddings for conditional generation eval
    val_bert = torch.stack(
        [dataset.bert_embs[val_set.indices[i]]
         for i in range(min(args.n_val_samples, len(val_set)))]
    )
    print(f'Data: {n} cascades  train={n_train} val={n_val} test={n_test}')
    print(f'Ref lengths: mean={np.mean(ref_lens):.1f}  '
          f'median={np.median(ref_lens):.0f}')

    model = CasCNSeqModel(
        bert_dim=768,
        hidden_dim=args.hidden_dim,
        rnn_units=args.rnn_units,
        n_time_bins=args.n_time_bins,
    ).to(args.device)
    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.lr * 0.05,
    )

    step = 0
    best_wd = float('inf')
    train_iter = iter(train_loader)

    print('Training...')
    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        bert_emb, times, seq_lens = batch
        bert_emb = bert_emb.to(args.device)
        times    = times.to(args.device)
        seq_lens = seq_lens.to(args.device)

        model.train()
        pred_delta, stop_logits = model.forward_train(bert_emb, times, seq_lens)
        loss, delta_l, stop_l   = cascn_loss(pred_delta, stop_logits,
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
            n_s = min(args.n_val_samples, len(val_set))
            gen_seqs = model.generate(
                val_bert[:n_s].to(args.device),
                max_len=args.max_events,
                stop_thresh=args.stop_thresh,
            )
            gen_lens = [len(s) for s in gen_seqs]
            wd = compute_w1(gen_lens, ref_lens[:n_s])
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
    model = CasCNSeqModel(
        bert_dim=768,
        hidden_dim=saved.get('hidden_dim', args.hidden_dim),
        rnn_units=saved.get('rnn_units', args.rnn_units),
        n_time_bins=saved.get('n_time_bins', args.n_time_bins),
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
