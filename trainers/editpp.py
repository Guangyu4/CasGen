"""EditFlows (adapted) trainer: cascade sequence generation from BERT embeddings.

Adapted from: Edit Flows: Flow Matching with Edit Operations (2024).
Original code: /scratch/gw2556/ODE/CodeRef/editpp/

Original EditFlows:
  - Continuous-time flow matching over event sequences via CTMC edit operations
    (insert, delete, optionally substitute).
  - Forward/Backward path: ElementWiseMixturePath — each position independently
    comes from x_1 (data) or x_0 (Poisson noise) with probability kappa(t).
  - Network: LlamaModel (causal Transformer) predicts log-rates lambda_ins / lambda_del
    at each token position, given current x_t and flow time t.
  - Training loss: Bregman divergence = rate_penalty - sum_log_rates (per token).
  - Inference: Euler steps — at each sub-step, stochastically insert/delete tokens.
  - Conditioning: none in the original (density estimation / forecasting via partial
    sequence conditioning).

Adaptations for BERT-conditioned cascade sequence generation:
  - BERT (768) → Linear + LayerNorm → bert_tok (B, hidden_dim): a single "context
    token" prepended to each sequence before feeding to the Transformer.
    All other tokens attend to it via standard self-attention.
  - LlamaModel (requires HuggingFace transformers + nested attention tricks) →
    PyTorch nn.TransformerEncoder (standard, no extra dependencies).
    The backbone change is the only architectural departure; the Edit Flow loss
    (Bregman divergence) and Euler sampling logic are preserved faithfully.
  - Sequence representation: padded tensor with EPSILON = -1 (instead of jagged
    DataBatch + numba JIT). Functionally equivalent for the operations we need.
  - ElementWiseMixturePath (LinearKappaSchedule: kappa(t) = t) is preserved exactly.
  - Data: *_burst.pt via BertSeqDataset; no .pkl format.
  - Training loop: plain PyTorch (no Hydra/Lightning).

What is removed / replaced:
  - LlamaModel → PyTorch TransformerEncoder (same role, no special installation).
  - DataBatch (jagged) → padded tensor with EPSILON mask.
  - numba JIT functions (rm_blanks, compute_interpolation_bins, etc.) → pure PyTorch.
  - Hydra config / Lightning → plain argparse + training loop.
  - Substitution operations (not needed for 1D temporal sequences).
"""
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from .casflow import BertSeqDataset, collate_bert_seq

# ============================================================
# Core constants & schedule
# ============================================================

EPSILON = -1.0   # sentinel token for "empty slot" (equivalent to EPSILON_TOKEN)


def _linear_kappa(t):
    """kappa(t) = t  (LinearKappaSchedule)."""
    return t


def _linear_loss_coeff(t):
    """d/dt kappa / (1 - kappa) = 1 / (1 - t)  (loss weight)."""
    return 1.0 / (1.0 - t).clamp(min=1e-3)


# ============================================================
# NyquistEmb for event times and flow time
# ============================================================

class NyquistEmb(nn.Module):
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


# ============================================================
# EditFlows backbone with TransformerEncoder
# ============================================================

class EditFlowModel(nn.Module):
    """
    Predicts log-rates (lambda_ins, lambda_del) at each token position given:
      - Current sequence x_t (padded with EPSILON tokens)
      - Flow time t (scalar per sequence)
      - BERT conditioning (per sequence)

    Architecture:
      BERT context token (1 token per sequence) + event tokens →
      TransformerEncoder → MLP head → (log_lambda_ins, log_lambda_del) per token.

    The BERT token serves as a global conditioning signal attending to all event
    tokens. Event tokens attend back to the BERT token (standard self-attention).
    """

    def __init__(self, bert_dim=768, hidden_dim=128, n_heads=4, n_layers=4,
                 n_ins_bins=32, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_ins_bins = n_ins_bins

        # BERT → single context token
        self.bert_proj = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Event time token embedding: NyquistEmb on raw time + projection
        self.time_emb  = NyquistEmb(dim=hidden_dim, timesteps=1.0)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)

        # Flow time embedding (scalar per sequence, broadcast to all tokens)
        self.flow_emb = nn.Sequential(
            NyquistEmb(dim=hidden_dim, timesteps=1.0),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformer encoder (replaces LlamaModel)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=4 * hidden_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Rate heads: predict log-rate for each real/epsilon token
        self.head_del = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Insertion rate + discrete bin logits Q_ins (where to insert)
        self.head_ins = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1 + n_ins_bins),
        )

    def forward(self, x_t, t, bert_emb, x_t_mask):
        """
        x_t       : (B, L) padded sequence (real times or EPSILON = -1)
        t         : (B,) flow time in [0, 1]
        bert_emb  : (B, 768)
        x_t_mask  : (B, L) bool — True for real tokens, False for EPSILON or padding

        Returns:
          log_lambda_ins : (B, L)
          log_lambda_del : (B, L)
          Q_ins_logits   : (B, L, n_ins_bins)
        """
        B, L = x_t.shape
        device = x_t.device

        # BERT context token: (B, 1, hidden)
        bert_tok = self.bert_proj(bert_emb).unsqueeze(1)   # (B, 1, hidden)

        # Event token embeddings
        real_times = x_t.clamp(0.0, 1.0)
        tok_emb    = self.time_proj(self.time_emb(real_times))  # (B, L, hidden)
        # Zero out EPSILON tokens (they still attend via self-attention)
        tok_emb    = tok_emb * x_t_mask.unsqueeze(-1)

        # Flow time embedding broadcast to all tokens
        flow_vec   = self.flow_emb(t)                           # (B, hidden)
        flow_tok   = flow_vec.unsqueeze(1).expand(-1, L, -1)    # (B, L, hidden)
        tok_emb    = tok_emb + flow_tok

        # Concat BERT token in front: (B, 1+L, hidden)
        src = torch.cat([bert_tok, tok_emb], dim=1)

        # Attention mask: all tokens attend to each other (BERT token + event tokens)
        # Padding positions (neither real nor epsilon but just extra padding) are masked
        pad_mask = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=device), ~x_t_mask],
            dim=1,
        )  # (B, 1+L): False = attend

        out = self.transformer(src, src_key_padding_mask=pad_mask)   # (B, 1+L, h)
        out = out[:, 1:, :]                                           # (B, L, hidden), drop BERT tok

        raw_ins = self.head_ins(out)                        # (B, L, 1+n_ins_bins)
        log_lambda_ins = raw_ins[:, :, 0]                   # (B, L)
        Q_ins_logits   = raw_ins[:, :, 1:]                  # (B, L, n_ins_bins)
        log_lambda_del = self.head_del(out).squeeze(-1)     # (B, L)

        return log_lambda_ins, log_lambda_del, Q_ins_logits


# ============================================================
# EditFlows training loss (Bregman divergence)
# ============================================================

def _editflow_loss(model, bert_emb, x_1_times, x_1_lens, max_len, device):
    """
    Compute the EditFlows Bregman divergence loss.

    Steps:
      1. Sample noise x_0 ~ Poisson(lambda * T), truncated to max_len.
      2. Pad x_0 and x_1 to the same augmented length with EPSILON.
      3. Sample flow time t ~ Uniform[0, 1].
      4. For each position, independently: x_t[i,j] = x_1_aug[i,j] if Bernoulli(kappa(t))
         else x_0_aug[i,j].
      5. Classify per position: "comes from x_1" or "comes from x_0".
         - If from x_0 and x_1[j] != EPSILON → should insert (delete this eps, insert x_1[j])
         - If from x_1 and x_1[j] == EPSILON → should delete
         - If both same source → no operation
      6. Loss = kappa_loss_coeff(t) * mean BCE on insert/delete decisions.
         This is a simplified but faithful implementation of the sum_log_rates term
         in the Bregman divergence (Eq. 28 in the EditFlows paper).
    """
    B = bert_emb.shape[0]

    # ---- Sample noise x_0 ----
    # Per the original PoissonPointProcess, sample ~Poisson(mean_n * T) events
    mean_n   = float(x_1_lens.float().mean().item())
    n_noise  = torch.poisson(torch.full((B,), mean_n, device=device)).long().clamp(1, max_len)
    x_0_list = [torch.sort(torch.rand(n_noise[i].item(), device=device)).values
                for i in range(B)]

    # Pad x_0 and x_1 to max_len with EPSILON
    x_0_pad = torch.full((B, max_len), EPSILON, device=device)
    x_1_pad = torch.full((B, max_len), EPSILON, device=device)
    for i in range(B):
        l0 = len(x_0_list[i])
        x_0_pad[i, :l0] = x_0_list[i]
        l1 = int(x_1_lens[i].item())
        x_1_pad[i, :l1] = x_1_times[i, :l1]

    # ---- Sample flow time t (clamp both sides to avoid 1/t and 1/(1-t) explosion) ----
    t = torch.rand(B, device=device).clamp(min=0.05, max=0.95)  # (B,)
    kappa = _linear_kappa(t)                                     # (B,)

    # ---- Sample x_t via ElementWiseMixturePath ----
    from_x1 = torch.rand(B, max_len, device=device) < kappa.unsqueeze(1)  # (B, L)
    x_t     = torch.where(from_x1, x_1_pad, x_0_pad)            # (B, L)

    # x_t_mask: True if token is real (not EPSILON)
    x_t_mask = x_t != EPSILON                                    # (B, L)

    # ---- Run model ----
    log_lambda_ins, log_lambda_del, Q_ins_logits = model(
        x_t, t, bert_emb, x_t_mask)

    # ---- Compute Bregman Poisson regression loss ----
    # should_del: x_t has ANY x_0 event (whether x_1 is empty or has a different event)
    # Covers both "extra noise" and "wrong-position noise" (substitution) cases.
    should_del = ~from_x1 & x_t_mask                            # (B, L)

    # should_ins: x_1 has a real event at this position and x_t came from x_0.
    # Covers both "empty slot needs insert" and "after-deletion slot needs insert" cases.
    should_ins = ~from_x1 & (x_1_pad != EPSILON)                # (B, L)

    # Separate Bregman coefficients for del (1/(1-t)) and ins (1/t)
    del_coeff = (1.0 / (1.0 - t).clamp(min=0.05)).unsqueeze(1)  # (B, 1)
    ins_coeff = (1.0 / t.clamp(min=0.05)).unsqueeze(1)           # (B, 1)

    R_del = del_coeff * should_del.float()                       # (B, L)
    R_ins = ins_coeff * should_ins.float()                       # (B, L)
    loss_del = log_lambda_del.exp() - R_del * log_lambda_del     # (B, L)
    loss_ins = log_lambda_ins.exp() - R_ins * log_lambda_ins     # (B, L)

    # Cross-entropy for insertion time bin (all should_ins positions, including substitution)
    bins     = Q_ins_logits.shape[-1]
    q_target = (x_1_pad.clamp(0, 1 - 1e-6) * bins).long().clamp(0, bins - 1)
    q_loss = F.cross_entropy(
        Q_ins_logits[should_ins], q_target[should_ins], reduction='mean'
    ) if should_ins.any() else 0.0

    loss = (loss_del + loss_ins).mean() + q_loss
    return loss


# ============================================================
# Euler sampling
# ============================================================

@torch.no_grad()
def _euler_sample(model, bert_embs, n_steps, max_len, device):
    """
    Euler-step sampling for EditFlows.
    Start from Poisson noise, iteratively insert/delete tokens.
    Returns list of (L_i,) tensors of event times.
    """
    B  = bert_embs.shape[0]
    h  = 1.0 / n_steps

    # Initial x_0 ~ Poisson(data_mean_n) — match training noise distribution
    mean_n = float(getattr(model, 'data_mean_n', max_len // 4))
    n_init = torch.poisson(torch.full((B,), mean_n, device=device)).long().clamp(1, max_len)
    x_t    = torch.full((B, max_len), EPSILON, device=device)
    for i in range(B):
        ni  = n_init[i].item()
        tms = torch.sort(torch.rand(ni, device=device)).values
        x_t[i, :ni] = tms

    for step_i in range(n_steps):
        t_cur = torch.full((B,), step_i / n_steps, device=device)
        x_t_mask = x_t != EPSILON                                    # (B, L)

        log_lambda_ins, log_lambda_del, Q_ins_logits = model(
            x_t, t_cur, bert_embs, x_t_mask)

        # Deletion: p = h * lambda_del for each real token
        p_del   = (h * log_lambda_del.exp()).clamp(0, 1)             # (B, L)
        del_ops = torch.rand(B, max_len, device=device) < p_del      # (B, L) bool
        del_ops = del_ops & x_t_mask

        # Insertion: p = h * lambda_ins for each epsilon slot
        p_ins   = (h * log_lambda_ins.exp()).clamp(0, 1)             # (B, L)
        ins_ops = torch.rand(B, max_len, device=device) < p_ins      # (B, L) bool
        ins_ops = ins_ops & ~x_t_mask

        # Sample insertion values
        bins    = Q_ins_logits.shape[-1]
        p_bins  = torch.softmax(Q_ins_logits, dim=-1)                # (B, L, bins)
        sampled_bins = torch.multinomial(
            p_bins.reshape(B * max_len, bins), 1
        ).reshape(B, max_len)
        bin_width = 1.0 / bins
        ins_vals  = (sampled_bins.float() + torch.rand(B, max_len, device=device)) * bin_width

        # Apply operations
        x_t_new = x_t.clone()
        x_t_new[del_ops] = EPSILON
        # Only insert if the slot is epsilon AND ins_ops is True
        x_t_new = torch.where(ins_ops, ins_vals, x_t_new)
        x_t     = x_t_new

    # Collect final event times for each sequence
    results = []
    for i in range(B):
        mask = x_t[i] != EPSILON
        times = x_t[i][mask].sort().values
        results.append(times)
    return results


# ============================================================
# Main model wrapper
# ============================================================

class EditFlowBertModel(nn.Module):
    """Thin wrapper that holds the EditFlowModel and sampling hyper-params."""
    def __init__(self, bert_dim=768, hidden_dim=128, n_heads=4, n_layers=4,
                 n_ins_bins=32, dropout=0.1, n_euler_steps=50, max_len=500):
        super().__init__()
        self.net = EditFlowModel(
            bert_dim=bert_dim, hidden_dim=hidden_dim, n_heads=n_heads,
            n_layers=n_layers, n_ins_bins=n_ins_bins, dropout=dropout,
        )
        self.n_euler_steps = n_euler_steps
        self.max_len       = max_len

    def forward(self, bert_emb, times, seq_lens):
        return _editflow_loss(self.net, bert_emb, times, seq_lens,
                              self.max_len, bert_emb.device)

    @torch.no_grad()
    def generate(self, bert_embs):
        return _euler_sample(self.net, bert_embs, self.n_euler_steps,
                             self.max_len, bert_embs.device)


# ============================================================
# Public interface
# ============================================================

def add_args(parser):
    parser.add_argument('--data', default='dataset/APS_burst.pt')
    parser.add_argument('--bert_model', default='pretrained/bert-base-chinese')
    parser.add_argument('--bert_device', default='cpu')
    parser.add_argument('--bert_cache', default='')
    parser.add_argument('--max_events', type=int, default=500)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_ins_bins', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_euler_steps', type=int, default=50)
    parser.add_argument('--outdir', default='runs_editpp')
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
    print(f'Model: EDITPP  data={args.data}')

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

    model = EditFlowBertModel(
        hidden_dim=args.hidden_dim, n_heads=args.n_heads, n_layers=args.n_layers,
        n_ins_bins=args.n_ins_bins, dropout=args.dropout,
        n_euler_steps=args.n_euler_steps, max_len=args.max_events,
    ).to(args.device)
    # Store data mean for consistent Euler sampler starting density
    all_train_lens = [len(dataset.time_seqs[i]) for i in train_set.indices]
    model.data_mean_n = float(np.mean(all_train_lens))
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
    print(f'data_mean_n = {model.data_mean_n:.1f}')

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
        model.train()
        loss = model(bert_emb, times, seq_lens)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        if step % 100 == 0:
            print(f'[{step:5d}] loss={loss.item():.4f}')
        if step % args.eval_every == 0:
            model.eval()
            gen_seqs = model.generate(val_bert[:n_eval].to(args.device))
            gen_lens = [len(s) for s in gen_seqs]
            ref_m    = max(np.mean(ref_lens), 1)
            w1 = wasserstein_distance(
                np.array(gen_lens)/ref_m, np.array(ref_lens)/ref_m)
            print(f'  [Eval] gen_mean={np.mean(gen_lens):.1f}  ref_mean={np.mean(ref_lens):.1f}  W1={w1:.4f}')
            if w1 < best_wd:
                best_wd = w1
                torch.save({'step': step, 'model': model.state_dict(),
                            'args': vars(args), 'best_wd': best_wd,
                            'data_mean_n': model.data_mean_n},
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
    model = EditFlowBertModel(
        hidden_dim=saved.get('hidden_dim', args.hidden_dim),
        n_heads=saved.get('n_heads', args.n_heads),
        n_layers=saved.get('n_layers', args.n_layers),
        n_ins_bins=saved.get('n_ins_bins', args.n_ins_bins),
        dropout=saved.get('dropout', args.dropout),
        n_euler_steps=saved.get('n_euler_steps', args.n_euler_steps),
        max_len=saved.get('max_events', args.max_events),
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.data_mean_n = ckpt.get('data_mean_n', saved.get('max_events', args.max_events) // 4)
    model.eval()

    print(f'Generating for {len(ref_seqs)} test cascades...')
    gen_seqs_all = []
    with torch.no_grad():
        for start in range(0, len(test_set), args.batch_size):
            idx  = test_set.indices[start: start + args.batch_size]
            embs = torch.stack([dataset.bert_embs[i] for i in idx]).to(args.device)
            seqs = model.generate(embs)
            gen_seqs_all.extend(to_numpy_seq(s) for s in seqs)

    m = eval_metrics(gen_seqs_all, ref_seqs)
    print_and_save(m, args.outdir)
    return m
