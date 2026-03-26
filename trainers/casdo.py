"""CasDO (adapted) trainer: sequence generation from BERT embeddings.

Adapted from: CasDO (AAAI 2024) - Cascade Popularity Prediction with a
Diffusion-ODE Model.

Original CasDO: graph embeddings (GraphWave+NetSMF) → ODE-RNN encoder
(runs ODE backwards through event sequence, GRU update at each event)
+ separate DDPM diffusion model (AttnBlock with Conv1d + self-attention)
that augments embeddings → bidirectional GRU decoder → scalar count.
Requires pickle data, graphwave preprocessing, torchdiffeq.

This adaptation preserves CasDO's two core contributions:

  1. Neural ODE hidden-state dynamics (ODEFunc: 2-layer MLP defining dz/dt;
     Euler integration over the generation trajectory).  Unlike RNN-based
     models the latent state evolves continuously between steps.

  2. Diffusion augmentation of the input embedding: learned noise-level σ,
     single forward-noising step (x_t = x_0 + σ·ε), MLP denoiser that
     reconstructs x_0.  This is the conceptual core of CasDO's diffusion
     component reduced to a tractable single-step form — full DDPM would
     require O(T_diff) passes and Conv1d/attention blocks that assume a
     fixed spatial layout incompatible with a single BERT vector.

What is removed / replaced:
  - GraphWave / NetSMF graph embeddings → BERT CLS embedding (B, 768)
  - ODE-RNN encoder (backward ODE over observed events) → simple
    linear projection, because we have one vector per cascade, not a
    per-event sequence to run backwards
  - Full DDPM (AttnBlock, beta schedule, noise-estimation loss, separate
    diff_optimizer) → single-step noising + MLP denoising (same principle)
  - Bidirectional GRU + MLP count head → autoregressive ODE trajectory
    readout with (delta_t, stop) head
  - pickle data format → *_burst.pt processed files with BERT text

Architecture summary:
  BERT emb (B,768)
    → Linear + LayerNorm → z_clean (B, z_dim)      [encoder]
    → z_noisy = z_clean + σ·ε, z_aug = denoiser(z_noisy)  [diffusion aug]
  ODE trajectory (Euler):
    z_0 = z_aug
    dz/dt = ode_func(z)   [ODEFunc: Linear→Tanh→Linear→Tanh→Linear, same as original]
    z_traj: (B, T+1, z_dim)  computed in one forward pass
  Head: Linear(z_dim → 2) applied at every step  → delta_t, stop_logit
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


# ---- Neural ODE components (kept faithful to original CasDO) ----

class ODEFunc(nn.Module):
    """
    Defines dz/dt = f_θ(z).
    Architecture matches original CasDO's ODEFunc exactly:
      Linear → Tanh → Linear → Tanh → Linear
    """
    def __init__(self, z_dim: int, ode_units: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, ode_units), nn.Tanh(),
            nn.Linear(ode_units, ode_units), nn.Tanh(),
            nn.Linear(ode_units, z_dim),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class EulerODE(nn.Module):
    """
    Euler integrator: z(t + dt) = z(t) + dt * f(z(t)).
    Original CasDO uses 'euler' as the ODE method.
    """
    def __init__(self, ode_func: ODEFunc):
        super().__init__()
        self.func = ode_func

    def integrate(self, z0: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Args:
            z0      : (B, z_dim) initial state
            n_steps : number of Euler steps
        Returns:
            traj    : (B, n_steps+1, z_dim)  z0 followed by n_steps updates
        """
        dt = 1.0 / max(n_steps, 1)
        z = z0
        traj = [z]
        for _ in range(n_steps):
            z = z + dt * self.func(z)
            traj.append(z)
        return torch.stack(traj, dim=1)   # (B, n_steps+1, z_dim)


# ---- Diffusion augmentation (single-step, faithful to CasDO concept) ----

class DiffusionAug(nn.Module):
    """
    Single-step diffusion augmentation.

    Forward: x_t = x_0 + σ · ε,  ε ~ N(0, I)
    Denoiser: x̂_0 = denoiser(x_t)
    Loss: MSE(x̂_0, x_0.detach())

    This captures CasDO's diffusion-augmentation concept (adding structured
    noise then learning to denoise) in a form compatible with a single
    z_dim-dimensional vector.  The original uses a full DDPM with T_diff=1000
    steps, beta schedule, and Conv1d AttnBlocks — those require fixed spatial
    size and multiple forward passes, making them incompatible here.
    """
    def __init__(self, z_dim: int):
        super().__init__()
        self.log_noise_std = nn.Parameter(torch.tensor(-2.0))   # learnable σ
        self.denoiser = nn.Sequential(
            nn.Linear(z_dim, z_dim * 2), nn.SiLU(),
            nn.Linear(z_dim * 2, z_dim),
        )

    def forward(self, z_clean: torch.Tensor):
        """
        Args:
            z_clean : (B, z_dim) clean latent
        Returns:
            z_aug       : (B, z_dim) denoised (augmented) latent
            diff_loss   : scalar reconstruction MSE
        """
        sigma = F.softplus(self.log_noise_std)
        eps   = torch.randn_like(z_clean)
        z_noisy = z_clean + sigma * eps
        z_aug   = self.denoiser(z_noisy)
        diff_loss = F.mse_loss(z_aug, z_clean.detach())
        return z_aug, diff_loss


# ---- Full model ----

class CasDOSeqModel(nn.Module):
    """
    Adapted CasDO for variable-length cascade sequence generation.

    Encoder:  BERT emb (B,768) → Linear → z_clean → DiffusionAug → z_aug
    ODE:      Euler integration of z_aug under ODEFunc → trajectory (B, T+1, z_dim)
    Head:     Linear(z_dim → 2) at each step → [delta_t_raw, stop_logit]
    """
    def __init__(self, bert_dim: int = 768, z_dim: int = 64,
                 ode_units: int = 128, diff_weight: float = 0.1,
                 dropout: float = 0.1):
        super().__init__()
        self.z_dim       = z_dim
        self.diff_weight = diff_weight

        # encoder
        self.bert_proj = nn.Sequential(
            nn.Linear(bert_dim, z_dim), nn.LayerNorm(z_dim),
            nn.Dropout(dropout),
        )

        # diffusion augmentation
        self.diff_aug = DiffusionAug(z_dim)

        # Neural ODE (Euler)
        self.ode_func   = ODEFunc(z_dim, ode_units)
        self.euler_ode  = EulerODE(self.ode_func)

        # output head
        self.dec_head = nn.Linear(z_dim, 2)   # [delta_t_raw, stop_logit]

    # ---- training (parallel over trajectory) ----

    def forward_train(self, bert_emb, times, seq_lens):
        """
        Args:
            bert_emb : (B, 768)
            times    : (B, T_max) padded with zeros
            seq_lens : (B,) actual lengths
        Returns:
            pred_delta  : (B, T_max+1)
            stop_logits : (B, T_max+1)
            diff_loss   : scalar
        """
        T = times.size(1)

        z_clean         = self.bert_proj(bert_emb)          # (B, z_dim)
        z_aug, diff_loss = self.diff_aug(z_clean)           # (B, z_dim), scalar

        traj = self.euler_ode.integrate(z_aug, n_steps=T)  # (B, T+1, z_dim)

        out         = self.dec_head(traj)                   # (B, T+1, 2)
        pred_delta  = out[:, :, 0]                          # (B, T+1)
        stop_logits = out[:, :, 1]                          # (B, T+1)

        return pred_delta, stop_logits, diff_loss

    # ---- inference ----

    @torch.no_grad()
    def generate(self, bert_embs, max_len=500, stop_thresh=0.5):
        """
        Generate conditioned on bert_embs (B, 768).
        Returns list of (T_i,) tensors of absolute times.
        """
        B      = bert_embs.size(0)
        z_clean = self.bert_proj(bert_embs)
        # no diffusion noise at inference — use clean encoding
        traj   = self.euler_ode.integrate(z_clean, n_steps=max_len)
        # (B, max_len+1, z_dim)

        out         = self.dec_head(traj)               # (B, max_len+1, 2)
        delta_raw   = out[:, :, 0]                      # (B, max_len+1)
        stop_logits = out[:, :, 1]                      # (B, max_len+1)

        delta_t   = F.softplus(delta_raw)               # (B, max_len+1) positive
        stop_prob = torch.sigmoid(stop_logits)          # (B, max_len+1)

        times_list = []
        for i in range(B):
            t_acc = 0.0
            ts = []
            for step in range(max_len + 1):
                if stop_prob[i, step].item() > stop_thresh:
                    break
                t_acc += delta_t[i, step].item()
                ts.append(t_acc)
            times_list.append(torch.tensor(ts, dtype=torch.float32))
        return times_list


# ---- Loss ----

def casdo_loss(pred_delta, stop_logits, times, seq_lens,
               diff_loss, diff_weight):
    B, Tp1 = pred_delta.shape
    T      = Tp1 - 1
    device = pred_delta.device

    prev     = torch.zeros(B, 1, device=device)
    gt_delta = torch.cat([prev, times], dim=1)
    gt_delta = gt_delta[:, 1:] - gt_delta[:, :-1]      # (B, T)

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
    total = delta_loss + stop_loss + diff_weight * diff_loss
    return total, delta_loss, stop_loss


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
    parser.add_argument('--z_dim', type=int, default=64,
                        help='Latent ODE state dimension')
    parser.add_argument('--ode_units', type=int, default=128,
                        help='ODEFunc hidden units (original CasDO default=128)')
    parser.add_argument('--diff_weight', type=float, default=0.1,
                        help='Weight for diffusion reconstruction loss')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--stop_thresh', type=float, default=0.5)
    parser.add_argument('--outdir', default='runs_casdo')
    parser.add_argument('--max_steps', type=int, default=30000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--n_val_samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    print(f'Model: CASDO  data={args.data}')

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

    model = CasDOSeqModel(
        bert_dim=768,
        z_dim=args.z_dim,
        ode_units=args.ode_units,
        diff_weight=args.diff_weight,
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
        pred_delta, stop_logits, diff_loss = model.forward_train(
            bert_emb, times, seq_lens,
        )
        loss, delta_l, stop_l = casdo_loss(
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
    model = CasDOSeqModel(
        bert_dim=768,
        z_dim=saved.get('z_dim', args.z_dim),
        ode_units=saved.get('ode_units', args.ode_units),
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
