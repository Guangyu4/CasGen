"""Add-Thin (adapted) trainer: cascade sequence generation from BERT embeddings.

Adapted from: Add and Thin: Diffusion for Temporal Point Processes (NeurIPS 2023).
Original repo: https://github.com/niklasnikolaus/add-thin

Original Add-Thin:
  - Forward process: thin x_0 (keep each event with prob alpha_cumprod[n])
    then superpose with a HPP noise process (intensity 1 - alpha_cumprod[n])
  - Backward: reverse Markov chain via PointClassifier (BCE) + MixtureIntensity (NLL)
  - Backbone: Dilated CNN (CNNSeqEmb) for event sequence encoding
  - Forecast conditioning: history GRU embedding injected into dif_time_emb channel

Adaptations for cascade sequence generation from BERT embeddings:
  - BERT (768) → Linear + LayerNorm → bert_vec replaces history GRU encoding.
    bert_vec is injected the same way: history_mlp(cat([bert_vec, dif_time_emb])) →
    conditioned dif_time_emb.  The structural changes are minimal and all original
    schedule/loss logic is preserved.
  - Input data: *_burst.pt with BERT text, via BertSeqDataset (no .pkl format).
  - tmax = 1.0 (all times are already normalised to [0, 1]).
  - NyquistFrequencyEmbedding, CNNSeqEmb, PointClassifier, MixtureIntensity, Batch
    are inlined so no add_thin package installation is required.

What is removed / replaced:
  - `history_encoder` (GRU over event history) → `bert_proj` (Linear + LayerNorm).
  - `.pkl` data pipeline → BertSeqDataset (shared with casflow / cascn / etc.).
  - Hydra / Lightning training loop → plain PyTorch loop.
  - Forecast-window time rescaling (not needed; full sequence generation).
"""
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.data import DataLoader, random_split

from .casflow import BertSeqDataset, collate_bert_seq

# ============================================================
# Inlined utilities (identical to add_thin.backbones / diffusion)
# ============================================================

def _betas_for_alpha_bar(steps, alpha_bar_fn, max_beta=0.999):
    betas = []
    for i in range(steps):
        t1, t2 = i / steps, (i + 1) / steps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class NyquistEmb(nn.Module):
    """Sine-cosine embedding for scalar/multi-dim timestamps (from add_thin.backbones.embeddings)."""
    def __init__(self, dim: int, timesteps: float):
        super().__init__()
        assert dim % 2 == 0
        k = dim // 2
        nyq = timesteps / 2
        phi = (1 + math.sqrt(5)) / 2
        freqs = np.geomspace(1 / 8, nyq / (2 * phi), num=k)
        scale = np.repeat(2 * np.pi * freqs / timesteps, 2)
        bias  = np.tile([0, np.pi / 2], k)
        self.register_buffer('scale', torch.from_numpy(scale.astype(np.float32)))
        self.register_buffer('bias',  torch.from_numpy(bias.astype(np.float32)))

    def forward(self, t):
        return torch.addcmul(self.bias, self.scale, t[..., None]).sin()


class CNNSeqEmb(nn.Module):
    """Dilated residual CNN for sequence embedding (from add_thin.backbones.cnn)."""
    def __init__(self, n_layers, input_dim, emb_dim, kernel_size=16):
        super().__init__()
        dil = [1, 4, 8, 16, 32, 64]
        layers = []
        for i in range(n_layers):
            in_d = input_dim if i == 0 else emb_dim
            layers.append(nn.Sequential(
                nn.Conv1d(in_d, emb_dim, kernel_size, padding='same',
                          padding_mode='zeros', dilation=dil[i]),
                nn.GroupNorm(max(1, emb_dim // 16), emb_dim),
            ))
        self.layers = nn.ModuleList(layers)
        self.act    = nn.ReLU(inplace=True)
        self.proj   = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        orig_len = x.shape[1]
        if orig_len < 30:
            x = F.pad(x, (0, 0, 0, 30 - orig_len))
        x = x.transpose(1, 2)
        for l in self.layers:
            x = x + self.act(l(x))
        x = x.transpose(1, 2)
        x = x[:, :orig_len]   # trim back to original length
        return self.proj(x)


class PointClassifier(nn.Module):
    """Binary classifier predicting x_0 ∩ x_n (from add_thin.backbones.classifier)."""
    def __init__(self, hidden_dim, n_layers):
        super().__init__()
        inp = 3 * hidden_dim
        layers = [nn.Linear(inp, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, dif_time_emb, time_emb, event_emb):
        L = time_emb.shape[1]
        x = torch.cat([time_emb, event_emb,
                        dif_time_emb.unsqueeze(1).expand(-1, L, -1)], dim=-1)
        return self.model(x).squeeze(-1)  # (B, L)


class _NormalTrunc(D.Normal):
    """Normal truncated to [0, 1] for MixtureIntensity (from add_thin.distributions.densities)."""
    def __init__(self, mean, std):
        super().__init__(mean.sigmoid(), F.softplus(std).clamp(min=0.01, max=2.0))

    def cdf(self, x):
        return super().cdf(x) - super().cdf(torch.zeros_like(x))


class MixtureIntensity(nn.Module):
    """Mixture intensity model (from add_thin.distributions.intensities)."""
    def __init__(self, n_components, emb_size):
        super().__init__()
        self.n_components = n_components
        self.w_act = nn.Softplus()
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size), nn.ReLU(),
            nn.Linear(emb_size, 3 * n_components),
        )
        self._rejection_mult = 2

    def _get_dist(self, event_emb, dif_time_emb, x_n, L):
        n_events = x_n.mask.float().sum(-1)
        seq_emb  = event_emb.nan_to_num(nan=0.0).sum(1) / n_events.clamp(min=1).unsqueeze(-1)
        dif_time_emb = dif_time_emb.nan_to_num(nan=0.0)
        params   = self.mlp(torch.cat([seq_emb, dif_time_emb], dim=-1))
        params   = params.nan_to_num(nan=0.0)
        loc, scale, w_raw = params.split(self.n_components, dim=-1)
        w = self.w_act(w_raw).nan_to_num(nan=0.0).clamp(min=1e-8)
        cif = (w.sum(-1) * (n_events + 1)).clamp(min=1e-8)
        mixture = D.Categorical(probs=w.unsqueeze(1).expand(-1, L, -1))
        comp    = _NormalTrunc(loc.unsqueeze(1).expand(-1, L, -1),
                               scale.unsqueeze(1).expand(-1, L, -1))
        return D.MixtureSameFamily(mixture, comp), cif

    def log_likelihood(self, event_emb, dif_time_emb, x_0, x_n):
        dist, cif = self._get_dist(event_emb, dif_time_emb, x_n, x_0.seq_len)
        x = x_0.time / x_0.tmax
        log_int = ((dist.log_prob(x).clamp(min=-50, max=50) + cif.clamp(min=1e-8).log().unsqueeze(-1)) * x_0.mask).sum(-1)
        cdf_val = dist.cdf(torch.ones_like(x)).mean(1)
        return log_int - cif * cdf_val

    def sample(self, event_emb, dif_time_emb, x_n):
        tmax = x_n.tmax
        dist, cif = self._get_dist(event_emb, dif_time_emb, x_n, 1)
        seq_len = D.Poisson(cif * dist.cdf(torch.ones(x_n.batch_size, 1, device=tmax.device)).squeeze()).sample().long()
        max_len = seq_len.max().item() + 1
        while True:
            raw  = dist.sample((max_len * self._rejection_mult,)).squeeze(-1).T * tmax
            ok   = (raw >= 0) & (raw <= tmax)
            idx  = torch.argsort(ok.int(), stable=True, descending=True, dim=-1)
            raw  = torch.take_along_dim(raw, idx, dim=-1)[:, :max_len]
            ok   = torch.take_along_dim(ok,  idx, dim=-1)[:, :max_len]
            mask = torch.arange(max_len, device=tmax.device).unsqueeze(0) < seq_len.unsqueeze(1)
            mask = mask & ok
            if (mask.sum(-1) == seq_len).all():
                break
            self._rejection_mult += 1
        times = raw * mask
        return Batch1D.from_tensors(times, mask, tmax)


# ============================================================
# Simplified 1D Batch (temporal only, no spatial)
# ============================================================

class Batch1D:
    """Simplified 1D temporal Batch for Add-Thin inference.
    Mirrors the add_thin.data.Batch interface for the operations we use.
    """
    def __init__(self, time, mask, tmax, unpadded_length, kept=None):
        self.time   = time             # (B, L)
        self.mask   = mask             # (B, L) bool
        self.tmax   = tmax             # scalar tensor
        self.unpadded_length = unpadded_length  # (B,)
        self.kept   = kept             # (B, L) bool or None
        t_pad = torch.cat([torch.zeros(time.shape[0], 1, device=time.device), time], 1)
        self.tau = (t_pad[:, 1:] - t_pad[:, :-1]) * mask

    @property
    def batch_size(self): return self.time.shape[0]
    @property
    def seq_len(self):    return self.time.shape[1]
    def __len__(self):    return self.batch_size

    def to(self, device):
        self.time = self.time.to(device)
        self.mask = self.mask.to(device)
        self.tmax = self.tmax.to(device)
        self.unpadded_length = self.unpadded_length.to(device)
        if self.kept is not None:
            self.kept = self.kept.to(device)
        self.tau = self.tau.to(device)
        return self

    def to_time_list(self):
        return [self.time[i][self.mask[i]].detach().cpu().numpy()
                for i in range(self.batch_size)]

    @staticmethod
    def _compact(time, mask, tmax, kept=None):
        time = time.clone(); time[~mask] = 2 * tmax.item()
        idx  = torch.argsort(time, dim=-1)
        time = torch.take_along_dim(time, idx, dim=-1)
        mask = torch.take_along_dim(mask, idx, dim=-1)
        if kept is not None:
            kept = torch.take_along_dim(kept, idx, dim=-1)
        max_len = max(1, mask.sum(-1).max().item())
        time, mask = time[:, :max_len], mask[:, :max_len]
        if kept is not None: kept = kept[:, :max_len]
        time = time * mask
        ul   = mask.long().sum(-1)
        return Batch1D(time, mask, tmax, ul, kept)

    @staticmethod
    def from_tensors(time, mask, tmax):
        return Batch1D._compact(time, mask.bool(), tmax)

    def thin(self, alpha):
        """Stochastic thinning with per-sequence probability alpha (B,)."""
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(1).expand(-1, self.seq_len)
        keep     = torch.bernoulli(alpha.clamp(0, 1)).bool() & self.mask
        not_keep = (~torch.bernoulli(alpha.clamp(0, 1)).bool()) & self.mask
        # recompute properly
        keep     = torch.bernoulli(alpha.clamp(0, 1)).bool()
        keep_m   = self.mask & keep
        rem_m    = self.mask & ~keep
        k_kept   = self.kept * keep_m.float() if self.kept is not None else None
        k_rem    = self.kept * rem_m.float()  if self.kept is not None else None
        return (Batch1D._compact(self.time * keep_m, keep_m, self.tmax, k_kept),
                Batch1D._compact(self.time * rem_m,  rem_m,  self.tmax, k_rem))

    def add_events(self, other):
        """Merge two Batch1D objects, re-sorting by time."""
        other = other.to(self.time.device)
        if self.kept is not None:
            kept = torch.cat([self.kept.float(),
                              torch.zeros_like(other.mask.float())], dim=-1)
        else:
            kept = torch.cat([torch.ones_like(self.mask.float()),
                              torch.zeros_like(other.mask.float())], dim=-1)
        kept = kept.bool()
        return Batch1D._compact(
            torch.cat([self.time, other.time], dim=-1),
            torch.cat([self.mask, other.mask], dim=-1),
            self.tmax,
            kept,
        )


def _generate_hpp(tmax, n_sequences, intensity=None):
    """1D HPP: n ~ Poisson(tmax * intensity), times ~ Uniform[0, tmax]."""
    device = tmax.device
    if intensity is None:
        intensity = torch.ones(n_sequences, device=device)
    n_pts    = torch.poisson(tmax * intensity).long()
    max_pts  = max(1, int(n_pts.max().item()) + 1)
    times    = torch.rand(n_sequences, max_pts, device=device) * tmax
    mask     = torch.arange(max_pts, device=device).unsqueeze(0) < n_pts.unsqueeze(1)
    times    = times * mask
    return Batch1D.from_tensors(times, mask, tmax)


# ============================================================
# Main model: Add-Thin with BERT conditioning
# ============================================================

class AddThinBertModel(nn.Module):
    """Add-Thin diffusion model conditioned on BERT embeddings.

    BERT embedding (B, 768) → bert_proj → bert_vec (B, hidden_dim)
    history_mlp(cat[bert_vec, dif_time_emb]) → conditioned_dif_time_emb (B, hidden_dim)
    This is identical to the original forecast conditioning path.
    """
    def __init__(self, bert_dim=768, hidden_dim=64, n_mix=10, n_cnn_layers=4,
                 kernel_size=16, steps=100, dropout=0.1):
        super().__init__()
        self.steps = steps
        self.n_max = 500

        # cosine beta schedule (identical to add_thin.diffusion.model)
        beta = _betas_for_alpha_bar(
            steps, lambda n: math.cos((n + 0.008) / 1.008 * math.pi / 2) ** 2)
        alpha = 1 - beta
        alpha_cumprod = alpha.cumprod(0)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        add_remove  = (1 - alpha_cumprod)[:-1] * beta[1:]
        alpha_x0_k  = (alpha_cumprod[:-1] - alpha_cumprod[1:]) / (1 - alpha_cumprod[1:])
        alpha_xn_k  = ((alpha - alpha_cumprod) / (1 - alpha_cumprod))[1:]
        self.register_buffer('alpha_x0_kept', alpha_x0_k)
        self.register_buffer('alpha_xn_kept', alpha_xn_k)
        self.register_buffer('add_remove',    add_remove)

        # BERT projection (replaces history GRU)
        self.bert_proj   = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.history_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU())

        # Event time encoder: NyquistEmb on [time, tau] → (B, L, hidden_dim)
        self.time_encoder = NyquistEmb(dim=hidden_dim // 2, timesteps=1.0)

        # Diffusion step encoder
        self.diffusion_time_encoder = nn.Sequential(
            NyquistEmb(dim=hidden_dim, timesteps=steps),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Sequence CNN encoder
        self.sequence_encoder = CNNSeqEmb(
            n_layers=n_cnn_layers, input_dim=hidden_dim, emb_dim=hidden_dim,
            kernel_size=kernel_size)

        # Classifier: predicts x_0 ∩ x_n (BCE loss)
        self.classifier = PointClassifier(hidden_dim=hidden_dim, n_layers=3)

        # Intensity model: models x_0 \ x_n (NLL loss)
        self.intensity = MixtureIntensity(n_components=n_mix, emb_size=2 * hidden_dim)

        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def _compute_emb(self, n, x_n, bert_vec):
        """Compute embeddings conditioning on bert_vec."""
        B, L = x_n.batch_size, x_n.seq_len
        dif_time_emb = self.diffusion_time_encoder(n)              # (B, hidden)
        dif_time_emb = self.history_mlp(
            torch.cat([bert_vec, dif_time_emb], dim=-1))           # inject BERT

        inp = torch.stack([x_n.time, x_n.tau], dim=-1)            # (B, L, 2)
        time_emb  = self.time_encoder(inp).reshape(B, L, -1)       # (B, L, hidden)
        event_emb = self.sequence_encoder(time_emb) * x_n.mask.unsqueeze(-1)
        return dif_time_emb, time_emb, event_emb

    def forward(self, bert_emb, x_0):
        """Training forward: add noise, classify, compute intensity NLL."""
        B = bert_emb.shape[0]
        bert_vec = self.bert_proj(bert_emb)                        # (B, hidden)
        n = torch.randint(0, self.steps, (B,), device=bert_emb.device, dtype=torch.long)

        # Forward process: thin x_0, superpose with HPP
        x_n, x_0_thin = x_0.thin(self.alpha_cumprod[n])
        hpp = _generate_hpp(x_0.tmax, B, 1 - self.alpha_cumprod[n])
        x_n = x_n.add_events(hpp)

        dif_time_emb, time_emb, event_emb = self._compute_emb(n, x_n, bert_vec)

        # Classifier
        logits = self.classifier(dif_time_emb, time_emb, event_emb)

        # BCE loss on x_n ∩ x_0 labels
        logits_flat = logits.flatten()[x_n.mask.flatten()]
        target_flat = x_n.kept.float().flatten()[x_n.mask.flatten()]
        clf_loss = self.bce(logits_flat, target_flat).sum() / B / self.n_max

        # Intensity NLL
        log_ll   = self.intensity.log_likelihood(event_emb, dif_time_emb, x_0_thin, x_n)
        int_loss = -log_ll.mean() / self.n_max

        return clf_loss + int_loss

    @torch.no_grad()
    def generate(self, bert_embs, tmax=None):
        """Reverse diffusion: generate event sequences conditioned on bert_embs."""
        B      = bert_embs.shape[0]
        device = bert_embs.device
        bert_vec = self.bert_proj(bert_embs)
        if tmax is None:
            tmax = torch.ones(1, device=device)

        x_n = _generate_hpp(tmax, B)

        for n_int in range(self.steps - 1, 0, -1):
            n = torch.full((B,), n_int, device=device, dtype=torch.long)
            x_n = self._sample_posterior(n, x_n, bert_vec)

        # final step
        n   = torch.zeros(B, device=device, dtype=torch.long)
        x_0 = self._sample_x0(n, x_n, bert_vec)
        return x_0

    def _sample_x0(self, n, x_n, bert_vec):
        dif_time_emb, time_emb, event_emb = self._compute_emb(n, x_n, bert_vec)
        sampled = self.intensity.sample(event_emb, dif_time_emb, x_n)
        logits  = self.classifier(dif_time_emb, time_emb, event_emb)
        clf_x0, not_x0 = x_n.thin(logits.sigmoid())
        return clf_x0.add_events(sampled)

    def _sample_posterior(self, n, x_n, bert_vec):
        _, clf_x0, sampled, not_x0 = self._sample_x0_full(n, x_n, bert_vec)
        x0_kept, _ = sampled.thin(self.alpha_x0_kept[n - 1])
        hpp         = _generate_hpp(x_n.tmax, x_n.batch_size, self.add_remove[n - 1])
        xn_kept, _  = not_x0.thin(self.alpha_xn_kept[n - 1])
        return clf_x0.add_events(hpp).add_events(xn_kept).add_events(x0_kept)

    def _sample_x0_full(self, n, x_n, bert_vec):
        dif_time_emb, time_emb, event_emb = self._compute_emb(n, x_n, bert_vec)
        sampled = self.intensity.sample(event_emb, dif_time_emb, x_n)
        logits  = self.classifier(dif_time_emb, time_emb, event_emb)
        clf_x0, not_x0 = x_n.thin(logits.sigmoid())
        return clf_x0.add_events(sampled), clf_x0, sampled, not_x0


# ============================================================
# Dataset helpers: convert BertSeqDataset outputs to Batch1D
# ============================================================

def _times_to_batch(times_tensor, seq_lens, device):
    """Convert (B, T_max) padded times tensor → Batch1D."""
    B, T = times_tensor.shape
    tmax = torch.ones(1, device=device)
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    for i, l in enumerate(seq_lens.tolist()):
        mask[i, :int(l)] = True
    return Batch1D.from_tensors(times_tensor * mask, mask, tmax)


# ============================================================
# Public interface
# ============================================================

def add_args(parser):
    parser.add_argument('--data', default='dataset/APS_burst.pt')
    parser.add_argument('--bert_model', default='pretrained/bert-base-chinese')
    parser.add_argument('--bert_device', default='cpu')
    parser.add_argument('--bert_cache', default='')
    parser.add_argument('--max_events', type=int, default=500)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--n_mix', type=int, default=10)
    parser.add_argument('--n_cnn_layers', type=int, default=4)
    parser.add_argument('--kernel_size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=100,
                        help='Number of diffusion steps')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--outdir', default='runs_addthin')
    parser.add_argument('--max_steps', type=int, default=30000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--n_val_samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    from scipy.stats import wasserstein_distance
    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    print(f'Model: ADDTHIN  data={args.data}')

    dataset = BertSeqDataset(
        args.data, bert_name=args.bert_model,
        bert_device=args.bert_device, bert_cache=args.bert_cache,
        max_events=args.max_events,
    )
    n       = len(dataset)
    n_train = int(n * 0.8); n_val = int(n * 0.1); n_test = n - n_train - n_val
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed))

    loader  = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                         collate_fn=collate_bert_seq, drop_last=True)
    n_eval  = min(args.n_val_samples, len(val_set))
    ref_lens = [len(dataset.time_seqs[val_set.indices[i]]) for i in range(n_eval)]
    val_bert = torch.stack([dataset.bert_embs[val_set.indices[i]] for i in range(n_eval)])

    model = AddThinBertModel(
        hidden_dim=args.hidden_dim, n_mix=args.n_mix,
        n_cnn_layers=args.n_cnn_layers, kernel_size=args.kernel_size,
        steps=args.steps, dropout=args.dropout,
    ).to(args.device)
    model.n_max = args.max_events
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_steps, eta_min=args.lr*0.05)

    step  = 0; best_wd = float('inf'); it = iter(loader)
    print('Training...')
    while step < args.max_steps:
        try: batch = next(it)
        except StopIteration: it = iter(loader); batch = next(it)
        bert_emb, times, seq_lens = batch
        bert_emb  = bert_emb.to(args.device)
        times     = times.to(args.device)
        seq_lens  = seq_lens.to(args.device)
        x_0 = _times_to_batch(times, seq_lens, args.device)
        model.train()
        loss = model(bert_emb, x_0)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        if step % 100 == 0:
            print(f'[{step:5d}] loss={loss.item():.4f}')
        if step % args.eval_every == 0:
            model.eval()
            gen = model.generate(val_bert[:n_eval].to(args.device))
            gen_lens = [len(g) for g in gen.to_time_list()]
            w1 = wasserstein_distance(
                np.array(gen_lens)/max(np.mean(ref_lens),1),
                np.array(ref_lens)/max(np.mean(ref_lens),1))
            print(f'  [Eval] gen_mean={np.mean(gen_lens):.1f}  ref_mean={np.mean(ref_lens):.1f}  W1={w1:.4f}')
            if w1 < best_wd:
                best_wd = w1
                torch.save({'step': step, 'model': model.state_dict(),
                            'args': vars(args), 'best_wd': best_wd},
                           os.path.join(args.outdir, 'best.pt'))
    print(f'Training done. Best W1: {best_wd:.4f}')
    test(args)


def test(args):
    from .metrics import eval_metrics, print_and_save, get_test_split, to_numpy_seq
    torch.manual_seed(args.seed)

    ckpt_path = os.path.join(args.outdir, 'best.pt')
    if not os.path.exists(ckpt_path):
        print(f'No checkpoint at {ckpt_path}, skipping test.'); return None

    dataset  = BertSeqDataset(
        args.data, bert_name=args.bert_model,
        bert_device=args.bert_device, bert_cache=args.bert_cache,
        max_events=args.max_events,
    )
    test_set = get_test_split(dataset, args)
    ref_seqs = [to_numpy_seq(dataset.time_seqs[i]) for i in test_set.indices]

    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    saved = ckpt.get('args', {})
    model = AddThinBertModel(
        hidden_dim=saved.get('hidden_dim', args.hidden_dim),
        n_mix=saved.get('n_mix', args.n_mix),
        n_cnn_layers=saved.get('n_cnn_layers', args.n_cnn_layers),
        kernel_size=saved.get('kernel_size', args.kernel_size),
        steps=saved.get('steps', args.steps),
        dropout=saved.get('dropout', args.dropout),
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.n_max = saved.get('max_events', args.max_events)
    model.eval()

    print(f'Generating for {len(ref_seqs)} test cascades...')
    gen_seqs = []
    with torch.no_grad():
        for start in range(0, len(test_set), args.batch_size):
            idx  = test_set.indices[start: start + args.batch_size]
            embs = torch.stack([dataset.bert_embs[i] for i in idx]).to(args.device)
            batch = model.generate(embs)
            for t in batch.to_time_list():
                gen_seqs.append(to_numpy_seq(torch.from_numpy(t)))

    m = eval_metrics(gen_seqs, ref_seqs)
    print_and_save(m, args.outdir)
    return m
