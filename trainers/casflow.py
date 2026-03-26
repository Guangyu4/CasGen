"""CasFlow (adapted) trainer: sequence generation from BERT embeddings.

Adapted from: CasFlow (TKDE 2023) - Exploring Hierarchical Structures and
Propagation Uncertainty for Cascade Prediction.

Original CasFlow: graph embeddings → VAE + NF + BiGRU → scalar count.
This adaptation: BERT embedding (B,768) → cascade VAE + NF + GRU decoder
→ variable-length event time sequence.

Key changes vs. original:
  - Input replaced by BERT CLS embedding (no graph preprocessing needed)
  - Node-level VAE removed (no per-step embeddings)
  - Bidirectional GRU + MLP count head replaced by autoregressive GRU decoder
  - Output is a padded time sequence instead of a scalar count
  - Evaluation uses W1 on sequence lengths (same as OURS)
"""
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.stats import wasserstein_distance


# ---- BERT encoding helper ----

def _encode_bert(texts, bert_name, device='cpu', batch_size=64):
    from transformers import AutoTokenizer, AutoModel
    print(f'Loading BERT: {bert_name}')
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    model = AutoModel.from_pretrained(bert_name).to(device)
    model.eval()
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors='pt', padding=True,
                        truncation=True, max_length=128)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        embs.append(out.last_hidden_state[:, 0, :].cpu())
        if (i // batch_size) % 10 == 0:
            print(f'  BERT {i}/{len(texts)}...')
    del model
    if device != 'cpu':
        torch.cuda.empty_cache()
    result = torch.cat(embs, dim=0)
    print(f'BERT done: {result.shape}')
    return result  # (N, 768)


# ---- Dataset ----

class BertSeqDataset(Dataset):
    """Loads *_burst.pt and pre-computes BERT embeddings.

    Returns per sample:
        bert_emb  : (768,)  float32
        times     : (T,)    float32  normalized to [0,1], only real events
        seq_len   : int
    """
    def __init__(self, data_path, bert_name, bert_device='cpu',
                 bert_cache='', max_events=500):
        data = torch.load(data_path, map_location='cpu', weights_only=False)
        cascades = data['cascades']

        texts, time_seqs = [], []
        for c in cascades:
            t = c['times'][:max_events]
            if len(t) == 0:
                continue
            texts.append(c.get('text', ''))
            time_seqs.append(t.float())

        if bert_cache and os.path.exists(bert_cache):
            print(f'Loading BERT cache: {bert_cache}')
            bert_embs = torch.load(bert_cache, map_location='cpu',
                                   weights_only=False)
        else:
            bert_embs = _encode_bert(texts, bert_name, bert_device)
            if bert_cache:
                os.makedirs(os.path.dirname(bert_cache) or '.', exist_ok=True)
                torch.save(bert_embs, bert_cache)
                print(f'BERT cache saved: {bert_cache}')

        assert len(bert_embs) == len(time_seqs)
        self.bert_embs = bert_embs       # (N, 768)
        self.time_seqs = time_seqs       # list of (T_i,) tensors
        self.stats = data['stats']

        counts = [len(t) for t in self.time_seqs]
        self.d_max = self.stats.get('D_max', 0)
        hist = torch.zeros(max(counts) + 1)
        for c in counts:
            hist[c] += 1
        self.count_dist = hist / hist.sum()
        self.t_max = 1.0
        print(f'BertSeqDataset: {len(self.time_seqs)} cascades')

    def __len__(self):
        return len(self.time_seqs)

    def __getitem__(self, idx):
        return self.bert_embs[idx], self.time_seqs[idx]


def collate_bert_seq(batch):
    """Pad time sequences with zeros; return bert_embs, padded_times, seq_lens."""
    bert_embs = torch.stack([b[0] for b in batch])           # (B, 768)
    time_seqs = [b[1] for b in batch]
    seq_lens  = torch.tensor([len(t) for t in time_seqs], dtype=torch.long)
    max_len   = int(seq_lens.max().item())
    padded    = torch.zeros(len(batch), max_len)
    for i, t in enumerate(time_seqs):
        padded[i, :len(t)] = t
    return bert_embs, padded, seq_lens


# ---- Model components (from original CasFlow, kept faithful) ----

class PlanarFlowLayer(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, z_dim))
        self.u = nn.Parameter(torch.randn(1, z_dim))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, z, log_det=None):
        m = lambda x: -1 + torch.log(1 + torch.exp(x))
        h = torch.tanh
        h_prime = lambda x: 1 - torch.tanh(x) ** 2

        wu = (self.w * self.u).sum()
        u_hat = (m(wu) - wu) * (self.w / self.w.norm()) + self.u

        lin = (z * self.w).sum(dim=-1, keepdim=True) + self.b
        z_new = z + u_hat * h(lin)
        affine = h_prime(lin) * self.w
        ld = torch.log(1e-7 + torch.abs(1 + (affine * u_hat).sum(dim=-1)))
        log_det = ld if log_det is None else log_det + ld
        return z_new, log_det


class NormalizingFlows(nn.Module):
    def __init__(self, z_dim, n_flows):
        super().__init__()
        self.flows = nn.ModuleList(
            [PlanarFlowLayer(z_dim) for _ in range(n_flows)]
        )

    def forward(self, z):
        log_det = None
        for flow in self.flows:
            z, log_det = flow(z, log_det)
        return z, log_det  # z_k, sum_log_det


# ---- Main model ----

class CasFlowSeqModel(nn.Module):
    """
    Adapted CasFlow for variable-length cascade sequence generation.

    Encoder:  BERT emb (B,768) → proj → cascade VAE → NF → z_k
    Decoder:  z_k → init hidden → GRU → (delta_t, stop) per step
    """
    def __init__(self, bert_dim=768, emb_dim=128, z_dim=64,
                 rnn_units=128, n_flows=8):
        super().__init__()
        self.z_dim = z_dim
        self.rnn_units = rnn_units

        # encoder
        self.proj   = nn.Sequential(nn.Linear(bert_dim, emb_dim), nn.ReLU(),
                                    nn.LayerNorm(emb_dim))
        self.mean_l = nn.Linear(emb_dim, z_dim)
        self.logv_l = nn.Linear(emb_dim, z_dim)

        # normalizing flows (original CasFlow contribution)
        self.nf = NormalizingFlows(z_dim, n_flows)

        # decoder
        self.dec_init = nn.Linear(z_dim, rnn_units)
        self.dec_gru  = nn.GRUCell(1, rnn_units)
        self.dec_head = nn.Linear(rnn_units, 2)  # [delta_t_raw, stop_logit]

    # ---- encoder helpers ----

    def encode(self, bert_emb):
        h = self.proj(bert_emb)
        mean, log_var = self.mean_l(h), self.logv_l(h)
        return mean, log_var

    @staticmethod
    def reparameterize(mean, log_var):
        std = torch.exp(0.5 * log_var)
        return mean + std * torch.randn_like(std)

    def kl_loss(self, mean, log_var):
        return -0.5 * torch.mean(log_var - mean ** 2 - torch.exp(log_var) + 1)

    # ---- training forward (teacher forcing) ----

    def forward_train(self, bert_emb, times, seq_lens):
        """
        Args:
            bert_emb : (B, 768)
            times    : (B, T_max) padded with zeros
            seq_lens : (B,) actual lengths

        Returns:
            pred_delta  : (B, T_max+1)  predicted delta at each step
            stop_logits : (B, T_max+1)  stop logit at each step
            kl          : scalar
            nf_loss     : scalar  (−mean log_det, to be minimised)
        """
        B = bert_emb.size(0)
        T = times.size(1)

        mean, log_var = self.encode(bert_emb)
        z = self.reparameterize(mean, log_var)
        z_k, log_det = self.nf(z)
        kl = self.kl_loss(mean, log_var)
        nf_loss = -torch.mean(log_det)

        # compute ground truth deltas: prepend 0 (first event starts from 0)
        # shape (B, T): delta[i,0]=times[i,0], delta[i,j]=times[i,j]-times[i,j-1]
        prev = torch.zeros(B, 1, device=times.device)
        gt_delta = torch.cat([prev, times], dim=1)          # (B, T+1)
        gt_delta = gt_delta[:, 1:] - gt_delta[:, :-1]       # (B, T)

        h = torch.tanh(self.dec_init(z_k))                  # (B, rnn_units)
        pred_deltas, stop_logits = [], []

        # teacher-forced steps: T real events + 1 EOS step
        # input at step i is the previous ground-truth delta (0 for i=0)
        inp = torch.zeros(B, 1, device=times.device)
        for i in range(T + 1):
            h = self.dec_gru(inp, h)
            out = self.dec_head(h)                           # (B, 2)
            pred_deltas.append(out[:, 0])
            stop_logits.append(out[:, 1])
            if i < T:
                inp = gt_delta[:, i].unsqueeze(1)            # teacher force
            # at step T we predict EOS, no need to feed next

        pred_delta  = torch.stack(pred_deltas,  dim=1)      # (B, T+1)
        stop_logits = torch.stack(stop_logits,  dim=1)      # (B, T+1)
        return pred_delta, stop_logits, kl, nf_loss

    # ---- inference: sample from prior ----

    @torch.no_grad()
    def generate(self, n, max_len=500, stop_thresh=0.5, device='cpu'):
        """
        Sample n cascades from the prior.
        Returns list of (T_i,) tensors of absolute times in [0,1].
        """
        z = torch.randn(n, self.z_dim, device=device)
        z_k, _ = self.nf(z)
        h = torch.tanh(self.dec_init(z_k))                  # (n, rnn_units)

        inp = torch.zeros(n, 1, device=device)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        times_list = [[] for _ in range(n)]
        t_cur = torch.zeros(n, device=device)

        for _ in range(max_len):
            h = self.dec_gru(inp, h)
            out = self.dec_head(h)                           # (n, 2)
            delta_t   = F.softplus(out[:, 0])               # positive
            stop_prob = torch.sigmoid(out[:, 1])

            # advance time for still-alive cascades
            t_cur = t_cur + delta_t
            stopped = stop_prob > stop_thresh

            for i in range(n):
                if alive[i] and not stopped[i]:
                    times_list[i].append(t_cur[i].item())

            alive = alive & ~stopped
            if not alive.any():
                break
            inp = delta_t.detach().unsqueeze(1)

        return [torch.tensor(ts, dtype=torch.float32) for ts in times_list]


# ---- Loss ----

def casflow_loss(pred_delta, stop_logits, times, seq_lens,
                 kl, nf_loss, kl_weight, nf_weight):
    """
    pred_delta  : (B, T+1)
    stop_logits : (B, T+1)
    times       : (B, T)  padded
    seq_lens    : (B,)
    """
    B, Tp1 = pred_delta.shape
    T = Tp1 - 1
    device = pred_delta.device

    # ground truth delta at positions 0..T-1
    prev = torch.zeros(B, 1, device=device)
    gt_delta = torch.cat([prev, times], dim=1)
    gt_delta = gt_delta[:, 1:] - gt_delta[:, :-1]           # (B, T)

    # stop target: 1 at position seq_len (EOS step), 0 before
    stop_target = torch.zeros(B, Tp1, device=device)
    for i, L in enumerate(seq_lens.tolist()):
        stop_target[i, L] = 1.0                             # step after last event

    # mask for real event positions (0..seq_len-1)
    mask = torch.zeros(B, T, device=device)
    for i, L in enumerate(seq_lens.tolist()):
        mask[i, :L] = 1.0

    # delta MSE on real positions
    delta_loss = ((F.softplus(pred_delta[:, :T]) - gt_delta) ** 2 * mask).sum() \
                 / mask.sum().clamp(min=1)

    # stop BCE on all T+1 positions
    stop_loss = F.binary_cross_entropy_with_logits(
        stop_logits, stop_target, reduction='mean'
    )

    total = delta_loss + stop_loss + kl_weight * kl + nf_weight * nf_loss
    return total, delta_loss, stop_loss


# ---- Eval metric (same as OURS) ----

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
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--rnn_units', type=int, default=128)
    parser.add_argument('--n_flows', type=int, default=8)
    parser.add_argument('--kl_weight', type=float, default=0.1)
    parser.add_argument('--nf_weight', type=float, default=0.1)
    parser.add_argument('--stop_thresh', type=float, default=0.5,
                        help='Stop probability threshold during inference')
    parser.add_argument('--outdir', default='runs_casflow')
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
    print(f'Model: CASFLOW  data={args.data}')

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
    print(f'Data: {n} cascades  train={n_train} val={n_val} test={n_test}')
    print(f'Ref lengths: mean={np.mean(ref_lens):.1f}  '
          f'median={np.median(ref_lens):.0f}')

    model = CasFlowSeqModel(
        bert_dim=768,
        emb_dim=args.emb_dim,
        z_dim=args.z_dim,
        rnn_units=args.rnn_units,
        n_flows=args.n_flows,
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
        pred_delta, stop_logits, kl, nf_loss = model.forward_train(
            bert_emb, times, seq_lens
        )
        loss, delta_l, stop_l = casflow_loss(
            pred_delta, stop_logits, times, seq_lens,
            kl, nf_loss, args.kl_weight, args.nf_weight,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % 100 == 0:
            print(f'[Step {step:5d}] loss={loss.item():.4f}  '
                  f'delta={delta_l.item():.4f}  stop={stop_l.item():.4f}  '
                  f'kl={kl.item():.4f}')

        if step % args.eval_every == 0:
            model.eval()
            n_s = min(args.n_val_samples, len(val_set))
            gen_seqs = model.generate(
                n_s, max_len=args.max_events,
                stop_thresh=args.stop_thresh,
                device=args.device,
            )
            gen_lens = [len(s) for s in gen_seqs]
            wd = compute_w1(gen_lens, ref_lens)
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
        args.data,
        bert_name=args.bert_model,
        bert_device=args.bert_device,
        bert_cache=args.bert_cache,
        max_events=args.max_events,
    )
    test_set = get_test_split(dataset, args)
    ref_seqs = [to_numpy_seq(dataset.time_seqs[i]) for i in test_set.indices]

    print('Loading checkpoint...')
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    saved = ckpt.get('args', {})
    model = CasFlowSeqModel(
        bert_dim=768,
        emb_dim=saved.get('emb_dim', args.emb_dim),
        z_dim=saved.get('z_dim', args.z_dim),
        rnn_units=saved.get('rnn_units', args.rnn_units),
        n_flows=saved.get('n_flows', args.n_flows),
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # CasFlow samples from the VAE prior (unconditional at inference)
    n_test = len(ref_seqs)
    print(f'Generating {n_test} samples from prior...')
    with torch.no_grad():
        gen_list = model.generate(
            n_test, max_len=args.max_events,
            stop_thresh=args.stop_thresh, device=args.device,
        )
    gen_seqs = [to_numpy_seq(s) for s in gen_list]

    m = eval_metrics(gen_seqs, ref_seqs)
    print_and_save(m, args.outdir)
    return m
