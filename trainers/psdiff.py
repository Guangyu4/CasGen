"""PSDiff (adapted) trainer: cascade sequence generation from BERT embeddings.

Adapted from: Point Set Diffusion (a.k.a. ps_diff baseline in the editpp repo).
Original code: /scratch/gw2556/ODE/CodeRef/editpp/editflows/baselines/ps_diff/

Original PSDiff:
  - Spatial point process extension of Add-Thin diffusion.
  - Forward: thin x_0 (retain prob from cosine schedule) + HPP superposition.
  - Backward: PointClassifier (attention backbone) + MixtureIntensity (NLL).
  - Backbone: TransformerEncoder (AttentionPointEmb) instead of CNN.
  - No built-in conditioning; works on 2D/3D spatial points.

Differences from the original:
  - Spatial (d-dim) points → 1D temporal events (time in [0, 1]).
  - No conditioning → BERT (768) conditioning injected via bert_proj (same
    history_mlp mechanism as Add-Thin adaptation).
  - 1D Batch (Batch1D, shared with addthin.py concept but re-implemented here
    with points tensor of shape (B, L, 1) to match AttentionPointEmb interface).
  - Different diffusion schedule coefficients (thin_and_add_probs, cosine alpha_bar
    identical to ps_diff, vs add_thin's slightly different alpha/beta formulation).
  - MixtureIntensity uses 1D Normal (same truncated normal as add-thin 1D variant).
  - NyquistEmb on sequence-length as extra conditioning (faithful to ps_diff).
  - Data: *_burst.pt via BertSeqDataset; no .pkl format.
  - Training loop: plain PyTorch (no Hydra/Lightning).
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
# Inlined utilities
# ============================================================

def _cosine_alpha_bar(n, steps):
    return math.cos((n / steps) / 1.0 * math.pi / 2) ** 2


def _thin_and_add_probs(steps):
    """Compute PSDiff schedule coefficients (from ps_diff/diffusion/utils.py)."""
    alpha_bar = torch.tensor(
        [_cosine_alpha_bar(n, steps) for n in range(steps)], dtype=torch.float64)
    retain   = F.pad(alpha_bar, (0, 1), value=0.0)
    add_prob = 1 - retain
    add_pre  = add_prob[:-1]; add_prob  = add_prob[1:]
    ret_pre  = retain[:-1];   retain    = retain[1:]
    retain_x0_kept = (ret_pre - retain) / (1 - retain).clamp(min=1e-8)
    add_keep       = add_pre / add_prob.clamp(min=1e-8)
    return retain.float(), add_prob.float(), add_keep.float(), retain_x0_kept.float()


class NyquistEmb(nn.Module):
    """Sine-cosine embedding (from ps_diff/backbones/embeddings.py)."""
    def __init__(self, dim: int, timesteps: float):
        super().__init__()
        assert dim % 2 == 0
        k = dim // 2
        phi = (1 + math.sqrt(5)) / 2
        freqs = np.geomspace(1 / 8, timesteps / 2 / (2 * phi), num=k)
        scale = np.repeat(2 * np.pi * freqs / timesteps, 2)
        bias  = np.tile([0.0, np.pi / 2], k)
        self.register_buffer('scale', torch.from_numpy(scale.astype(np.float32)))
        self.register_buffer('bias',  torch.from_numpy(bias.astype(np.float32)))

    def forward(self, t):
        return torch.addcmul(self.bias, self.scale, t[..., None]).sin()


class AttentionPointEmb(nn.Module):
    """Transformer-based point embedding (from ps_diff/backbones/attention.py).
    Adapted to accept (B, L, 1) 1D time points directly.
    """
    def __init__(self, n_blocks, input_dim, embs_dim):
        super().__init__()
        if input_dim != embs_dim:
            self.proj = nn.Sequential(
                nn.Linear(input_dim, embs_dim * 2), nn.ReLU(),
                nn.Linear(embs_dim * 2, embs_dim),
            )
        else:
            self.proj = nn.Identity()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embs_dim, nhead=max(1, embs_dim // 16),
            dim_feedforward=4 * embs_dim, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_blocks)

    def forward(self, points, mask):
        """points: (B, L, 1), mask: (B, L) bool → (B, L, embs_dim)."""
        x = self.proj(points) * mask.unsqueeze(-1)
        valid = mask.sum(-1) > 0
        out   = torch.zeros(x.shape[0], x.shape[1], x.shape[2], device=x.device)
        if not valid.any():
            return out
        out[valid] = self.encoder(
            x[valid], src_key_padding_mask=~mask[valid])
        return out.nan_to_num(nan=0.0)


class PointClassifier(nn.Module):
    """Binary classifier predicting x_0 ∩ x_n (from ps_diff/backbones/classifier.py).
    Input: [event_emb, dif_time_emb_broadcast] → 2 * hidden_dim.
    """
    def __init__(self, hidden_dim, n_layers):
        super().__init__()
        layers = [nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*layers)
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, dif_time_emb, event_emb):
        L = event_emb.shape[1]
        x = torch.cat([event_emb, dif_time_emb.unsqueeze(1).expand(-1, L, -1)], dim=-1)
        return self.model(x).squeeze(-1)


class _NormalTrunc(D.Normal):
    def __init__(self, mean, std):
        super().__init__(mean.sigmoid(), F.softplus(std).clamp(min=0.01, max=2.0))
    def cdf(self, x):
        return super().cdf(x) - super().cdf(torch.zeros_like(x))


class MixtureIntensity(nn.Module):
    """Mixture intensity for 1D temporal events (adapted from ps_diff/distributions/intensities.py)."""
    def __init__(self, n_components, emb_size):
        super().__init__()
        self.n_components = n_components
        self.w_act = nn.Softplus()
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size), nn.ReLU(),
            nn.Linear(emb_size, 3 * n_components),
        )
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
        self._rej_mult = 2

    def _get_dist(self, event_emb, dif_time_emb, mask, L):
        n_ev   = mask.float().sum(-1)
        seq_emb = event_emb.nan_to_num(nan=0.0).sum(1) / n_ev.clamp(min=1).unsqueeze(-1)
        dif_time_emb = dif_time_emb.nan_to_num(nan=0.0)
        params  = self.mlp(torch.cat([seq_emb, dif_time_emb], dim=-1))
        params  = params.nan_to_num(nan=0.0)
        loc, sc, w_raw = params.split(self.n_components, dim=-1)
        w   = self.w_act(w_raw).nan_to_num(nan=0.0).clamp(min=1e-8)
        cif = (w.sum(-1) * (n_ev + 1)).clamp(min=1e-8)
        mix = D.Categorical(probs=w.unsqueeze(1).expand(-1, L, -1))
        comp = _NormalTrunc(loc.unsqueeze(1).expand(-1, L, -1),
                            sc.unsqueeze(1).expand(-1, L, -1))
        return D.MixtureSameFamily(mix, comp), cif

    def log_likelihood(self, event_emb, dif_time_emb, x_0_time, x_0_mask, x_n_mask):
        dist, cif = self._get_dist(event_emb, dif_time_emb, x_n_mask, x_0_time.shape[1])
        x = x_0_time.clamp(0.0, 1.0)
        log_int = ((dist.log_prob(x).clamp(min=-50, max=50) + cif.clamp(min=1e-8).log().unsqueeze(-1)) * x_0_mask).sum(-1)
        cdf_val = dist.cdf(torch.ones_like(x)).mean(1)
        return log_int - cif * cdf_val

    def sample(self, event_emb, dif_time_emb, x_n_mask, tmax):
        B = event_emb.shape[0]
        dist, cif = self._get_dist(event_emb, dif_time_emb, x_n_mask, 1)
        seq_len = D.Poisson(cif * dist.cdf(torch.ones(B, 1, device=tmax.device)).squeeze()).sample().long()
        max_len = max(1, seq_len.max().item() + 1)
        while True:
            raw   = dist.sample((max_len * self._rej_mult,)).squeeze(-1).T * tmax
            ok    = (raw >= 0) & (raw <= tmax)
            idx   = torch.argsort(ok.int(), stable=True, descending=True, dim=-1)
            raw   = torch.take_along_dim(raw, idx, dim=-1)[:, :max_len]
            ok    = torch.take_along_dim(ok,  idx, dim=-1)[:, :max_len]
            mask  = torch.arange(max_len, device=tmax.device).unsqueeze(0) < seq_len.unsqueeze(1)
            mask  = mask & ok
            if (mask.sum(-1) == seq_len).all(): break
            self._rej_mult += 1
        times = raw * mask
        return times, mask


# ============================================================
# Simplified Batch (with 1D points tensor for AttentionPointEmb)
# ============================================================

class Batch1D:
    """1D temporal batch compatible with PSDiff's Batch interface."""
    def __init__(self, time, mask, tmax, unpadded_length, kept=None):
        self.time   = time            # (B, L)
        self.mask   = mask            # (B, L) bool
        self.tmax   = tmax
        self.unpadded_length = unpadded_length
        self.kept   = kept

    @property
    def batch_size(self): return self.time.shape[0]
    @property
    def seq_len(self):    return self.time.shape[1]
    def __len__(self):    return self.batch_size

    @property
    def points(self):
        """(B, L, 1) for AttentionPointEmb compatibility."""
        return self.time.unsqueeze(-1)

    def to(self, device):
        self.time = self.time.to(device)
        self.mask = self.mask.to(device)
        self.tmax = self.tmax.to(device)
        self.unpadded_length = self.unpadded_length.to(device)
        if self.kept is not None: self.kept = self.kept.to(device)
        return self

    def to_time_list(self):
        return [self.time[i][self.mask[i]].detach().cpu().numpy()
                for i in range(self.batch_size)]

    @staticmethod
    def _compact(time, mask, tmax, kept=None):
        time = time.clone(); time[~mask] = 2.0
        idx  = torch.argsort(time, dim=-1)
        time = torch.take_along_dim(time, idx, dim=-1)
        mask = torch.take_along_dim(mask, idx, dim=-1)
        if kept is not None: kept = torch.take_along_dim(kept, idx, dim=-1)
        ml   = max(1, mask.sum(-1).max().item())
        time, mask = time[:, :ml], mask[:, :ml]
        if kept is not None: kept = kept[:, :ml]
        time = time * mask
        ul   = mask.long().sum(-1)
        return Batch1D(time, mask, tmax, ul, kept)

    @staticmethod
    def from_tensors(time, mask, tmax):
        return Batch1D._compact(time, mask.bool(), tmax)

    def thin(self, alpha):
        if alpha.dim() == 1: alpha = alpha.unsqueeze(1).expand(-1, self.seq_len)
        keep  = torch.bernoulli(alpha.clamp(0, 1)).bool()
        km    = self.mask & keep
        rm    = self.mask & ~keep
        kk    = self.kept * km.float() if self.kept is not None else None
        kr    = self.kept * rm.float() if self.kept is not None else None
        return (Batch1D._compact(self.time * km, km, self.tmax, kk),
                Batch1D._compact(self.time * rm, rm, self.tmax, kr))

    def add_events(self, other):
        other = other.to(self.time.device)
        if self.kept is not None:
            kept = torch.cat([self.kept.float(),
                              torch.zeros_like(other.mask.float())], dim=-1).bool()
        else:
            kept = torch.cat([torch.ones_like(self.mask.float()),
                              torch.zeros_like(other.mask.float())], dim=-1).bool()
        return Batch1D._compact(
            torch.cat([self.time, other.time], dim=-1),
            torch.cat([self.mask, other.mask], dim=-1),
            self.tmax, kept)


def _generate_hpp(tmax, B, intensity=None):
    device = tmax.device
    if intensity is None:
        intensity = torch.ones(B, device=device)
    n_pts   = torch.poisson(tmax * intensity).long()
    max_pts = max(1, int(n_pts.max().item()) + 1)
    times   = torch.rand(B, max_pts, device=device) * tmax
    mask    = torch.arange(max_pts, device=device).unsqueeze(0) < n_pts.unsqueeze(1)
    return Batch1D.from_tensors(times * mask, mask, tmax)


# ============================================================
# Main model: PSDiff with BERT conditioning
# ============================================================

class PSDiffBertModel(nn.Module):
    """Point Set Diffusion adapted for 1D temporal sequence generation with BERT conditioning."""

    def __init__(self, bert_dim=768, hidden_dim=64, n_mix=10, n_blocks=4,
                 steps=100, dropout=0.1):
        super().__init__()
        self.steps = steps
        self.n_max = 500

        retain, add_prob, add_keep, retain_x0_kept = _thin_and_add_probs(steps)
        self.register_buffer('retain',        retain)
        self.register_buffer('add_prob',      add_prob)
        self.register_buffer('add_keep',      add_keep)
        self.register_buffer('retain_x0_kept', retain_x0_kept)

        # BERT → conditioning vector (replaces no-conditioning in original PSDiff)
        self.bert_proj   = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.history_mlp = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU())

        # Diffusion time encoder (identical to PSDiff)
        self.diffusion_time_encoder = nn.Sequential(
            NyquistEmb(dim=hidden_dim, timesteps=steps),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        for m in self.diffusion_time_encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)

        # Attention encoder: 1D points (dim=1) → embs_dim = hidden_dim // 2
        self.sequence_encoder = AttentionPointEmb(
            n_blocks=n_blocks, input_dim=1, embs_dim=hidden_dim // 2)

        # Sequence-length encoder (NyquistEmb on count, same as PSDiff)
        self.seq_len_encoder = nn.Sequential(
            NyquistEmb(dim=hidden_dim // 2, timesteps=500),
            nn.Linear(hidden_dim // 2, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )

        # Classifier (PSDiff variant: input = 2 * hidden_dim)
        self.classifier = PointClassifier(hidden_dim=hidden_dim, n_layers=3)

        # Intensity model
        self.intensity = MixtureIntensity(n_components=n_mix, emb_size=2 * hidden_dim)

        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def _compute_emb(self, n, x_n, bert_vec):
        B = x_n.batch_size
        dif_time_emb = self.diffusion_time_encoder(n)         # (B, hidden)
        dif_time_emb = self.history_mlp(
            torch.cat([bert_vec, dif_time_emb], dim=-1))      # inject BERT

        # Attention over 1D points (B, L, 1)
        point_emb = self.sequence_encoder(x_n.points, x_n.mask)   # (B, L, h//2)
        len_emb   = self.seq_len_encoder(
            x_n.unpadded_length.float())                           # (B, h//2)
        event_emb = torch.cat(
            [point_emb, len_emb.unsqueeze(1).expand(-1, x_n.seq_len, -1)],
            dim=-1) * x_n.mask.unsqueeze(-1)                       # (B, L, hidden)

        return dif_time_emb, event_emb

    def forward(self, bert_emb, x_0):
        B = bert_emb.shape[0]
        bert_vec = self.bert_proj(bert_emb)
        n = torch.randint(0, self.steps, (B,), device=bert_emb.device, dtype=torch.long)

        x_n, x_0_thin = x_0.thin(self.retain[n])
        hpp = _generate_hpp(x_0.tmax, B, self.add_prob[n])
        x_n = x_n.add_events(hpp)

        dif_time_emb, event_emb = self._compute_emb(n, x_n, bert_vec)

        logits = self.classifier(dif_time_emb, event_emb)  # (B, L)

        logits_flat = logits.flatten()[x_n.mask.flatten()]
        target_flat = x_n.kept.float().flatten()[x_n.mask.flatten()]
        clf_loss = self.bce(logits_flat, target_flat).sum() / B / self.n_max

        log_ll   = self.intensity.log_likelihood(
            event_emb, dif_time_emb,
            x_0_thin.time, x_0_thin.mask, x_n.mask)
        int_loss = -log_ll.mean() / self.n_max

        return clf_loss + int_loss

    @torch.no_grad()
    def generate(self, bert_embs):
        B      = bert_embs.shape[0]
        device = bert_embs.device
        bert_vec = self.bert_proj(bert_embs)
        tmax     = torch.ones(1, device=device)

        x_n = _generate_hpp(tmax, B)

        for n_int in range(self.steps - 1, 0, -1):
            n = torch.full((B,), n_int, device=device, dtype=torch.long)
            x_n = self._sample_posterior(n, x_n, bert_vec)

        n   = torch.zeros(B, device=device, dtype=torch.long)
        x_0 = self._sample_x0(n, x_n, bert_vec)
        return x_0

    def _sample_x0(self, n, x_n, bert_vec):
        dif_time_emb, event_emb = self._compute_emb(n, x_n, bert_vec)
        times, mask = self.intensity.sample(event_emb, dif_time_emb,
                                            x_n.mask, x_n.tmax)
        sampled = Batch1D.from_tensors(times, mask, x_n.tmax)
        logits  = self.classifier(dif_time_emb, event_emb)
        clf_x0, not_x0 = x_n.thin(logits.sigmoid())
        return clf_x0.add_events(sampled)

    def _sample_posterior(self, n, x_n, bert_vec):
        dif_time_emb, event_emb = self._compute_emb(n, x_n, bert_vec)
        times_s, mask_s = self.intensity.sample(event_emb, dif_time_emb,
                                                x_n.mask, x_n.tmax)
        sampled = Batch1D.from_tensors(times_s, mask_s, x_n.tmax)
        logits  = self.classifier(dif_time_emb, event_emb)
        clf_x0, not_x0 = x_n.thin(logits.sigmoid())

        x0_kept, _ = sampled.thin(self.retain_x0_kept[n - 1])
        xn_kept, _ = not_x0.thin(self.add_keep[n - 1])
        return clf_x0.add_events(xn_kept).add_events(x0_kept)


def _times_to_batch(times_tensor, seq_lens, device):
    B, T = times_tensor.shape
    tmax  = torch.ones(1, device=device)
    mask  = torch.zeros(B, T, dtype=torch.bool, device=device)
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
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--outdir', default='runs_psdiff')
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
    print(f'Model: PSDIFF  data={args.data}')

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

    loader   = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_bert_seq, drop_last=True)
    n_eval   = min(args.n_val_samples, len(val_set))
    ref_lens = [len(dataset.time_seqs[val_set.indices[i]]) for i in range(n_eval)]
    val_bert = torch.stack([dataset.bert_embs[val_set.indices[i]] for i in range(n_eval)])

    model = PSDiffBertModel(
        hidden_dim=args.hidden_dim, n_mix=args.n_mix,
        n_blocks=args.n_blocks, steps=args.steps, dropout=args.dropout,
    ).to(args.device)
    model.n_max = args.max_events
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_steps, eta_min=args.lr*0.05)

    step = 0; best_wd = float('inf'); it = iter(loader)
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
            gen     = model.generate(val_bert[:n_eval].to(args.device))
            gen_lens = [len(t) for t in gen.to_time_list()]
            ref_m   = max(np.mean(ref_lens), 1)
            w1 = wasserstein_distance(
                np.array(gen_lens)/ref_m, np.array(ref_lens)/ref_m)
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
    model = PSDiffBertModel(
        hidden_dim=saved.get('hidden_dim', args.hidden_dim),
        n_mix=saved.get('n_mix', args.n_mix),
        n_blocks=saved.get('n_blocks', args.n_blocks),
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
