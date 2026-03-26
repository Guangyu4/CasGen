"""CasFT (adapted) trainer: sequence generation from BERT embeddings.

Adapted from: CasFT (ACM MM 2024) - Diffusion-based Cascade Forecasting via
Transformer with Spatio-Temporal Point Processes.

Original CasFT pipeline:
  1. GraphWave + Transformer encoder: per-event spatial graph embeddings
     (N, T, cg_emb_dim=80) processed with multi-head attention + Neural ODE.
  2. Neural Temporal Point Process (NeuralPointProcess): ODE-based intensity
     model (torchdiffeq dopri5), produces per-event hidden states and
     cumulative intensities Λ for `interval_num=10` fixed time intervals.
  3. Conditional Gaussian DDPM (GaussianDiffusion_ST + ST_Diffusion): full
     DDPM (T_diff=1000 or 100 sampling steps, cosine beta schedule, pred_x0
     objective) over an `interval_num`-dim interval feature vector, conditioned
     on TPP hidden state + Λ vector.
  4. Predictor MLP: [tpp_state, diffused_interval_features] → scalar count.

CasFT's distinguishing contribution: the conditional DDPM learns to generate
a compact temporal distribution (histogram over intervals) from cascade
context, which is richer than a point estimate.

This adaptation preserves CasFT's two core contributions:

  1. Interval-based temporal representation: ground-truth times are bucketed
     into `interval_num` bins; the bin occupancy vector captures the temporal
     distribution of the cascade.

  2. Conditional DDPM over interval features: a multi-step denoising process
     (pred_x0 objective, cosine beta schedule, same as original CasFT)
     generates interval features conditioned on the BERT embedding. This is
     structurally identical to CasFT's GaussianDiffusion_ST, adapted to
     operate on a single BERT vector instead of the ODE-TPP hidden state.

What is removed / replaced:
  - GraphWave + NetSMF graph preprocessing + Transformer encoder →
    single BERT CLS embedding (B, 768)
  - ODE-based TPP (torchdiffeq, dopri5, per-event intensity modeling) →
    removed; BERT embedding directly serves as conditioning (the TPP provides
    cascade context, which BERT already encodes)
  - `Λ` vector (integrated intensities per interval) → replaced by predicted
    interval soft-histograms computed from GT times during training (gives
    the DDPM the same semantic target without requiring ODE integration)
  - Scalar count Predictor → GRU autoregressive decoder from interval
    features to variable-length event time sequence
  - DDP multi-GPU training, torchdiffeq, omegaconf, tensorboard →
    standard PyTorch single-GPU loop
  - Pickle data format → *_burst.pt processed files with BERT text

Architecture summary:
  BERT emb (B, 768) → Linear + LayerNorm → cond (B, cond_dim)

  Training:
    GT times → soft histogram → interval_feat (B, interval_num)   [target]
    DDPM loss: forward noise interval_feat → learn to denoise conditioned on cond

  Inference (ancestral sampling, same as CasFT's DDPM):
    x_T ~ N(0,I)  [size interval_num]
    for t = T..1: x_{t-1} = denoiser(x_t, t, cond)  [reverse diffusion]
    x_0 = sampled interval features

  Decoder (both training and inference):
    seq_feat = MLP(x_0)         [same as CasFT's seq_feature]
    h_0 = Linear([cond; seq_feat])
    GRUCell autoregressively: h_i, (delta_t_i, stop_i) = f(h_{i-1}, delta_{i-1})
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


# ---- Sinusoidal timestep embedding (same as CasFT's get_timestep_embedding) ----

def _sinusoidal_emb(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding for DDPM timestep, matching CasFT's implementation."""
    half = dim // 2
    freq = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / (half - 1)
    )
    emb = timesteps.float()[:, None] * freq[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, dim)


# ---- Conditional Denoiser (CasFT's ST_Diffusion adapted) ----

class CondDenoiser(nn.Module):
    """
    Conditional denoising network: predicts x_0 from x_t, timestep t, and
    conditioning vector cond.

    Faithfully adapts CasFT's ST_Diffusion (a 3-layer MLP conditioned on
    time embedding + cascade context) to work on a single BERT-derived
    conditioning vector instead of [tpp_hidden; Λ_intervals].
    """
    def __init__(self, cond_dim: int, interval_num: int, t_emb_dim: int = 64):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.t_proj = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 2), nn.SiLU(),
            nn.Linear(t_emb_dim * 2, t_emb_dim),
        )
        inp = interval_num + cond_dim + t_emb_dim
        self.net = nn.Sequential(
            nn.Linear(inp, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, interval_num),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t   : (B, interval_num) noisy interval features
            t     : (B,) integer diffusion timesteps
            cond  : (B, cond_dim) BERT-derived condition
        Returns:
            x_0_pred : (B, interval_num) predicted clean interval features
        """
        t_emb = _sinusoidal_emb(t, self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        inp   = torch.cat([x_t, cond, t_emb], dim=-1)
        return self.net(inp)


# ---- Simplified DDPM (pred_x0, cosine beta schedule, same as CasFT) ----

class IntervalDDPM(nn.Module):
    """
    Conditional DDPM over interval_num-dim interval features.

    Mirrors CasFT's GaussianDiffusion_ST:
      - objective : 'pred_x0'   (same as CasFT default)
      - beta_schedule : 'cosine' (same as CasFT)
      - T (n_timesteps): 200 by default (CasFT uses 1000 train / 100 sample;
        we use 200/50 to stay tractable on a single GPU with BERT preprocessing)
    """
    def __init__(self, cond_dim: int, interval_num: int,
                 T: int = 200, sample_T: int = 50):
        super().__init__()
        self.T          = T
        self.sample_T   = sample_T
        self.interval_num = interval_num
        self.denoiser   = CondDenoiser(cond_dim, interval_num)

        # cosine beta schedule (Nichol & Dhariwal 2021, same as CasFT)
        steps    = T + 1
        x        = torch.linspace(0, T, steps) / T
        alphas_b = torch.cos((x + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_b = alphas_b / alphas_b[0]
        betas    = torch.clamp(1 - alphas_b[1:] / alphas_b[:-1], max=0.9999)
        alphas   = 1.0 - betas
        alpha_bar = alphas.cumprod(dim=0)       # ᾱ_t  (T,)

        self.register_buffer('betas',     betas)
        self.register_buffer('alphas',    alphas)
        self.register_buffer('alpha_bar', alpha_bar)

    # ---- training loss ----

    def loss(self, x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Compute DDPM training loss (pred_x0 MSE).
        Args:
            x0   : (B, interval_num) clean interval features
            cond : (B, cond_dim)
        """
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=x0.device)
        ab = self.alpha_bar[t][:, None]          # (B, 1)
        eps = torch.randn_like(x0)
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        x0_pred = self.denoiser(x_t, t, cond)
        return F.mse_loss(x0_pred, x0)

    # ---- ancestral sampling (DDIM-style, strided, same objective as CasFT) ----

    @torch.no_grad()
    def sample(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Ancestral sampling conditioned on cond (B, cond_dim).
        Returns x_0 (B, interval_num).
        """
        B, device = cond.size(0), cond.device
        x = torch.randn(B, self.interval_num, device=device)

        # uniformly strided timesteps (DDIM-like subset)
        stride    = max(self.T // self.sample_T, 1)
        timesteps = list(range(self.T - 1, -1, -stride))

        for t_val in timesteps:
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            x0_pred = self.denoiser(x, t, cond)

            ab   = self.alpha_bar[t_val]
            ab_p = self.alpha_bar[t_val - 1] if t_val > 0 else torch.ones(1, device=device)
            beta = self.betas[t_val]

            # DDPM posterior mean
            coef1 = ab_p.sqrt() * beta / (1 - ab)
            coef2 = self.alphas[t_val].sqrt() * (1 - ab_p) / (1 - ab)
            mean  = coef1 * x0_pred + coef2 * x

            if t_val > 0:
                var  = (1 - ab_p) / (1 - ab) * beta
                x    = mean + var.sqrt() * torch.randn_like(mean)
            else:
                x    = mean
        return x


# ---- GRU-based sequence decoder ----

class SeqDecoder(nn.Module):
    """Autoregressive GRU decoder: interval features + cond → event time sequence."""
    def __init__(self, cond_dim: int, interval_num: int, rnn_units: int):
        super().__init__()
        # seq_feature: same structure as CasFT's seq_feature MLP
        self.seq_feature = nn.Sequential(
            nn.Linear(interval_num, interval_num * 2), nn.ReLU(),
            nn.Linear(interval_num * 2, interval_num * 2), nn.ReLU(),
            nn.Linear(interval_num * 2, interval_num),
        )
        self.h0_proj  = nn.Linear(cond_dim + interval_num, rnn_units)
        self.rnn_cell = nn.GRUCell(1, rnn_units)
        self.head     = nn.Linear(rnn_units, 2)   # [delta_t_raw, stop_logit]

    def forward_train(self, cond, interval_feat, times, seq_lens):
        """
        Args:
            cond          : (B, cond_dim)
            interval_feat : (B, interval_num)
            times         : (B, T_max) padded
            seq_lens      : (B,)
        Returns:
            pred_delta    : (B, T_max+1)
            stop_logits   : (B, T_max+1)
        """
        B, T = times.shape
        device = times.device

        seq_emb = self.seq_feature(interval_feat)        # (B, interval_num)
        h = torch.tanh(self.h0_proj(
            torch.cat([cond, seq_emb], dim=-1)           # (B, cond_dim+interval_num)
        ))                                               # (B, rnn_units)

        # build GT delta_t for teacher forcing
        prev     = torch.zeros(B, 1, device=device)
        gt_times = torch.cat([prev, times], dim=1)
        gt_delta = gt_times[:, 1:] - gt_times[:, :-1]   # (B, T_max)

        pred_deltas  = []
        stop_logits_ = []
        x_in = torch.zeros(B, 1, device=device)         # first input: 0

        for i in range(T + 1):
            h = self.rnn_cell(x_in, h)
            out = self.head(h)
            pred_deltas.append(out[:, 0])
            stop_logits_.append(out[:, 1])
            # teacher forcing
            if i < T:
                x_in = gt_delta[:, i:i+1]

        pred_delta  = torch.stack(pred_deltas,  dim=1)   # (B, T+1)
        stop_logits = torch.stack(stop_logits_, dim=1)   # (B, T+1)
        return pred_delta, stop_logits

    @torch.no_grad()
    def generate(self, cond, interval_feat, max_len, stop_thresh):
        B      = cond.size(0)
        device = cond.device

        seq_emb = self.seq_feature(interval_feat)
        h = torch.tanh(self.h0_proj(torch.cat([cond, seq_emb], dim=-1)))

        x_in = torch.zeros(B, 1, device=device)
        times_list = [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len + 1):
            h = self.rnn_cell(x_in, h)
            out = self.head(h)
            delta = F.softplus(out[:, 0])             # (B,)
            stop  = torch.sigmoid(out[:, 1])          # (B,)

            for i in range(B):
                if done[i]:
                    continue
                if stop[i].item() > stop_thresh:
                    done[i] = True
                else:
                    prev_t = times_list[i][-1] if times_list[i] else 0.0
                    times_list[i].append(prev_t + delta[i].item())

            if done.all():
                break
            x_in = delta.unsqueeze(1)

        return [torch.tensor(ts, dtype=torch.float32) for ts in times_list]


# ---- Full model ----

class CasFTSeqModel(nn.Module):
    """
    Adapted CasFT for variable-length cascade sequence generation.

    BERT (B,768) → cond  →  DDPM samples interval features
                          →  GRU decoder → event time sequence
    """
    def __init__(self, bert_dim: int = 768, cond_dim: int = 64,
                 interval_num: int = 10, rnn_units: int = 128,
                 T_diff: int = 200, sample_T: int = 50,
                 diff_weight: float = 0.5, dropout: float = 0.1,
                 max_time: float = 1.0):
        super().__init__()
        self.interval_num = interval_num
        self.diff_weight  = diff_weight
        self.max_time     = max_time

        self.bert_proj = nn.Sequential(
            nn.Linear(bert_dim, cond_dim), nn.LayerNorm(cond_dim),
            nn.Dropout(dropout),
        )
        self.ddpm    = IntervalDDPM(cond_dim, interval_num, T=T_diff, sample_T=sample_T)
        self.decoder = SeqDecoder(cond_dim, interval_num, rnn_units)

    def _times_to_interval_feat(self, times: torch.Tensor,
                                seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Compute soft histogram over interval_num bins from GT times.
        This replaces CasFT's Λ (integrated intensities per interval) which
        requires the ODE-TPP solver.

        Strategy: normalise each cascade's last event time to [0,1], then
        bin events into interval_num uniform bins and L1-normalise.  This
        gives the same semantics as Λ (distribution of activity over time)
        without needing a differential equation solver.
        """
        B, T = times.shape
        bins = torch.zeros(B, self.interval_num, device=times.device)
        for i in range(B):
            L   = seq_lens[i].item()
            ts  = times[i, :L]
            if L == 0:
                continue
            t_max = ts.max().clamp(min=1e-6)
            t_norm = ts / t_max                              # [0, 1]
            idx    = (t_norm * self.interval_num).long().clamp(0, self.interval_num - 1)
            bins[i].scatter_add_(0, idx, torch.ones_like(ts))
        # L1-normalise each row (so it's a probability vector)
        bins = bins / (bins.sum(dim=-1, keepdim=True).clamp(min=1e-6))
        return bins.detach()

    def forward_train(self, bert_emb, times, seq_lens):
        cond          = self.bert_proj(bert_emb)
        interval_feat = self._times_to_interval_feat(times, seq_lens)

        diff_loss = self.ddpm.loss(interval_feat, cond)

        # Use GT interval features for decoder (teacher forcing in diffusion)
        pred_delta, stop_logits = self.decoder.forward_train(
            cond, interval_feat, times, seq_lens,
        )
        return pred_delta, stop_logits, diff_loss

    @torch.no_grad()
    def generate(self, bert_embs, max_len=500, stop_thresh=0.5):
        cond          = self.bert_proj(bert_embs)
        interval_feat = self.ddpm.sample(cond)
        return self.decoder.generate(cond, interval_feat, max_len, stop_thresh)


# ---- Loss ----

def casft_loss(pred_delta, stop_logits, times, seq_lens, diff_loss, diff_weight):
    B, Tp1 = pred_delta.shape
    T      = Tp1 - 1
    device = pred_delta.device

    prev     = torch.zeros(B, 1, device=device)
    gt_delta = torch.cat([prev, times], dim=1)
    gt_delta = gt_delta[:, 1:] - gt_delta[:, :-1]   # (B, T)

    stop_target = torch.zeros(B, Tp1, device=device)
    for i, L in enumerate(seq_lens.tolist()):
        stop_target[i, int(L)] = 1.0

    mask = torch.zeros(B, T, device=device)
    for i, L in enumerate(seq_lens.tolist()):
        mask[i, :int(L)] = 1.0

    delta_loss = ((F.softplus(pred_delta[:, :T]) - gt_delta) ** 2 * mask).sum() \
                 / mask.sum().clamp(min=1)
    stop_loss  = F.binary_cross_entropy_with_logits(
        stop_logits, stop_target, reduction='mean',
    )
    return delta_loss + stop_loss + diff_weight * diff_loss, delta_loss, stop_loss


def compute_w1(gen_lens, ref_lens):
    g = np.array(gen_lens, dtype=float)
    r = np.array(ref_lens, dtype=float)
    if len(g) == 0 or len(r) == 0:
        return 1.0
    mean_r = max(r.mean(), 1)
    return float(wasserstein_distance(g / mean_r, r / mean_r))


# ---- Public interface ----

def add_args(parser):
    parser.add_argument('--data', default='dataset/APS_burst.pt')
    parser.add_argument('--bert_model', default='pretrained/bert-base-chinese')
    parser.add_argument('--bert_device', default='cpu')
    parser.add_argument('--bert_cache', default='')
    parser.add_argument('--max_events', type=int, default=500)
    parser.add_argument('--cond_dim', type=int, default=64,
                        help='BERT projection dim (plays role of tpp_hdims in original)')
    parser.add_argument('--interval_num', type=int, default=10,
                        help='Number of time intervals for DDPM target (same as original)')
    parser.add_argument('--rnn_units', type=int, default=128)
    parser.add_argument('--T_diff', type=int, default=200,
                        help='DDPM training timesteps (original uses 1000)')
    parser.add_argument('--sample_T', type=int, default=50,
                        help='DDPM sampling steps (original uses 100)')
    parser.add_argument('--diff_weight', type=float, default=0.5,
                        help='Weight for DDPM loss (same role as loss_scale in original)')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--stop_thresh', type=float, default=0.5)
    parser.add_argument('--outdir', default='runs_casft')
    parser.add_argument('--max_steps', type=int, default=30000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--n_val_samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    print(f'Model: CASFT  data={args.data}')

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

    model = CasFTSeqModel(
        bert_dim=768,
        cond_dim=args.cond_dim,
        interval_num=args.interval_num,
        rnn_units=args.rnn_units,
        T_diff=args.T_diff,
        sample_T=args.sample_T,
        diff_weight=args.diff_weight,
        dropout=args.dropout,
    ).to(args.device)
    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.lr * 0.05,
    )

    step       = 0
    best_wd    = float('inf')
    train_iter = iter(train_loader)

    print('Training...')
    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch      = next(train_iter)

        bert_emb, times, seq_lens = batch
        bert_emb  = bert_emb.to(args.device)
        times     = times.to(args.device)
        seq_lens  = seq_lens.to(args.device)

        model.train()
        pred_delta, stop_logits, diff_loss = model.forward_train(
            bert_emb, times, seq_lens,
        )
        loss, delta_l, stop_l = casft_loss(
            pred_delta, stop_logits, times, seq_lens,
            diff_loss, args.diff_weight,
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
                  f'diff={diff_loss.item():.4f}')

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
    model = CasFTSeqModel(
        bert_dim=768,
        cond_dim=saved.get('cond_dim', args.cond_dim),
        interval_num=saved.get('interval_num', args.interval_num),
        rnn_units=saved.get('rnn_units', args.rnn_units),
        T_diff=saved.get('T_diff', args.T_diff),
        sample_T=saved.get('sample_T', args.sample_T),
        diff_weight=saved.get('diff_weight', args.diff_weight),
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
