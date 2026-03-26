"""IFL-TPP (adapted) trainer: cascade sequence generation from BERT embeddings.

Adapted from: Intensity-Free Learning of Temporal Point Processes (ICLR 2020).
Original repo: https://github.com/shchur/ifl-tpp

Original ifl-tpp:
  - RNN (GRU/LSTM) encodes event history into context vectors
  - LogNormalMixtureDistribution models inter-event time distribution
  - Marks (event types) predicted as Categorical from context
  - Training via NLL (maximum likelihood) of the full TPP sequence
  - Generation: autoregressive sampling until cumulative time >= t_end
  - Data: .pkl files with arrival_times, marks, t_start, t_end

This adaptation preserves ifl-tpp's core contributions:

  1. LogNormalMixtureDistribution: GMM in log-space → affine normalisation →
     ExpTransform → positive inter-event times.  The linear layer, parameterisation
     (locs, log_scales, log_weights), and clamping are unchanged.

  2. RNN context with time-shift trick (RecurrentTPP.get_context): context after
     event i predicts event i+1; context_init prepended so step 0 is predicted
     from the initial state.  Fully preserved except context_init source (see below).

  3. Survival function in the likelihood: log_survival_function of the inter-time
     distribution at the last observed event — the "no event in [t_N, t_end]" term.
     Here t_end is replaced by the EOS mark at position T, and the survival term is
     computed at the context of the last real event (position T-1 in the context).

  4. NLL training objective: the only model in this baseline suite trained with
     proper maximum likelihood rather than MSE + BCE.

What is removed / replaced:

  - context_init as nn.Parameter → BERT (768) → Linear + LayerNorm → GRU hidden
    state.  The original context_init is a single learned vector shared across all
    sequences; replacing it with a per-sequence BERT projection provides text
    conditioning while keeping the RNN architecture identical.

  - t_end stopping → 2-class mark system (real=1, EOS=0).  Original ifl-tpp
    generates until cumulative time >= t_end.  We add a Categorical mark head
    (mark_linear: context_size → 2) and stop generation when mark=EOS is sampled.
    The EOS mark log-probability at position T replaces the survival term role:
    both capture "the sequence ends here".  We also compute the inter-time
    survival term at the last real event position for a more faithful likelihood.

  - .pkl data format → *_burst.pt with BERT text, reusing BertSeqDataset from
    casflow.py.  Mean/std of log inter-event times are computed from the training
    split (as in dpp.data.dataset.get_inter_time_statistics).

  - dpp package dependency → all distribution classes inlined below so no
    external package installation is required.  The code is identical to the
    original dpp.distributions.{normal,mixture_same_family,transformed_distribution}.
"""
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.distributions import Categorical
from torch.utils.data import DataLoader, random_split
from scipy.stats import wasserstein_distance

from .casflow import BertSeqDataset, collate_bert_seq


# ============================================================
# Inlined distribution helpers (identical to dpp.distributions)
# ============================================================

def _clamp_pg(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Clamp preserving gradients in clamped region (from dpp.utils)."""
    return x + (x.clamp(lo, hi) - x).detach()


class _Normal(D.Normal):
    """Normal with log_cdf / log_survival_function (from dpp.distributions.normal)."""
    def log_cdf(self, x):
        cdf = _clamp_pg(self.cdf(x), 1e-7, 1 - 1e-7)
        return cdf.log()

    def log_survival_function(self, x):
        cdf = _clamp_pg(self.cdf(x), 1e-7, 1 - 1e-7)
        return torch.log(1.0 - cdf)


class _MixtureSameFamily(D.MixtureSameFamily):
    """MixtureSameFamily with log_survival_function (from dpp.distributions.mixture_same_family)."""
    def log_cdf(self, x):
        x = self._pad(x)
        return torch.logsumexp(
            self.component_distribution.log_cdf(x) + self.mixture_distribution.logits,
            dim=-1,
        )

    def log_survival_function(self, x):
        x = self._pad(x)
        return torch.logsumexp(
            self.component_distribution.log_survival_function(x) + self.mixture_distribution.logits,
            dim=-1,
        )


class _TransformedDistribution(D.TransformedDistribution):
    """TransformedDistribution with log_survival_function (from dpp.distributions.transformed_distribution)."""
    def __init__(self, base_distribution, transforms, validate_args=None):
        super().__init__(base_distribution, transforms, validate_args=validate_args)
        sign = 1
        for t in self.transforms:
            sign = sign * t.sign
        self.sign = int(sign)

    def log_cdf(self, x):
        for t in self.transforms[::-1]:
            x = t.inv(x)
        return self.base_dist.log_cdf(x) if self.sign == 1 \
            else self.base_dist.log_survival_function(x)

    def log_survival_function(self, x):
        for t in self.transforms[::-1]:
            x = t.inv(x)
        return self.base_dist.log_survival_function(x) if self.sign == 1 \
            else self.base_dist.log_cdf(x)


# ============================================================
# LogNormalMixtureDistribution (identical to dpp/models/log_norm_mix.py)
# ============================================================

class LogNormalMixtureDistribution(_TransformedDistribution):
    """
    Mixture of log-normal distributions.

    x ~ GaussianMixtureModel(locs, log_scales, log_weights)
    y = std_log_inter_time * x + mean_log_inter_time
    z = exp(y)

    Exactly matches LogNormalMixtureDistribution in the original ifl-tpp repo.
    """
    def __init__(self, locs, log_scales, log_weights,
                 mean_log_inter_time=0.0, std_log_inter_time=1.0):
        mixture  = D.Categorical(logits=log_weights)
        comp     = _Normal(loc=locs, scale=log_scales.exp())
        gmm      = _MixtureSameFamily(mixture, comp)
        if mean_log_inter_time == 0.0 and std_log_inter_time == 1.0:
            transforms = []
        else:
            transforms = [D.AffineTransform(loc=mean_log_inter_time,
                                            scale=std_log_inter_time)]
        transforms.append(D.ExpTransform())
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time  = std_log_inter_time
        super().__init__(gmm, transforms)

    @property
    def mean(self):
        a      = self.std_log_inter_time
        b      = self.mean_log_inter_time
        loc    = self.base_dist._component_distribution.loc
        var    = self.base_dist._component_distribution.variance
        logw   = self.base_dist._mixture_distribution.logits
        return (logw + a * loc + b + 0.5 * a**2 * var).logsumexp(-1).exp()


# ============================================================
# Main model
# ============================================================

class IFLTPPSeqModel(nn.Module):
    """
    Adapted ifl-tpp for variable-length cascade sequence generation.

    Encoder:  BERT emb (B, 768) → Linear + LayerNorm → h_0  [context_init per cascade]
    RNN:      GRU step-by-step, input = [log_inter_time_norm, mark_emb]
    Dist:     LogNormalMixtureDistribution from context (same as original LogNormMix)
    Mark:     Categorical(2): real=1, EOS=0 — replaces t_end stopping criterion
    Training: NLL = -[inter_time log_prob + survival at last event + mark log_prob]
    """
    MARK_EOS  = 0
    MARK_REAL = 1

    def __init__(self, bert_dim=768, context_size=64, mark_emb_size=32,
                 n_mix=16, rnn_type='GRU', dropout=0.1):
        super().__init__()
        self.context_size  = context_size
        self.mark_emb_size = mark_emb_size
        self.n_mix         = n_mix

        # BERT → context_init (replaces original nn.Parameter context_init)
        self.bert_proj = nn.Sequential(
            nn.Linear(bert_dim, context_size), nn.LayerNorm(context_size),
            nn.Dropout(dropout),
        )

        # Mark embedding: 2 marks (EOS=0, real=1) — same as original mark_embedding
        self.mark_embedding = nn.Embedding(2, mark_emb_size)

        # RNN: input = [log_inter_time_norm (1), mark_emb (mark_emb_size)]
        # Using GRUCell for step-by-step generation; matches original GRU module logic
        rnn_cls = {'GRU': nn.GRUCell, 'LSTM': nn.LSTMCell, 'RNN': nn.RNNCell}[rnn_type]
        self.rnn      = rnn_cls(1 + mark_emb_size, context_size)
        self.rnn_type = rnn_type

        # Inter-time distribution head: same as original LogNormMix.linear
        self.linear = nn.Linear(context_size, 3 * n_mix)

        # Mark head: same as original mark_linear
        self.mark_linear = nn.Linear(context_size, 2)

        # Log inter-time normalisation statistics (set from data)
        self.register_buffer('mean_log_tau', torch.tensor(0.0))
        self.register_buffer('std_log_tau',  torch.tensor(1.0))

    def set_inter_time_stats(self, mean_log_tau: float, std_log_tau: float):
        self.mean_log_tau.fill_(mean_log_tau)
        self.std_log_tau.fill_(max(std_log_tau, 1e-6))

    def _get_inter_time_dist(self, context: torch.Tensor) -> LogNormalMixtureDistribution:
        """context: (B, S, context_size) → LogNormMix over (B, S)."""
        raw    = self.linear(context)              # (B, S, 3*n_mix)
        locs   = raw[..., :self.n_mix]
        lscale = _clamp_pg(raw[..., self.n_mix: 2*self.n_mix], -5.0, 3.0)
        lw     = torch.log_softmax(raw[..., 2*self.n_mix:], dim=-1)
        return LogNormalMixtureDistribution(
            locs, lscale, lw,
            float(self.mean_log_tau), float(self.std_log_tau),
        )

    def _input_feature(self, inter_time: torch.Tensor, mark: torch.Tensor) -> torch.Tensor:
        """Build RNN input feature from one inter-time step and one mark.

        inter_time : (B,)  — raw (not log-normalised here; we normalise inside)
        mark       : (B,)  long
        returns    : (B, 1 + mark_emb_size)
        """
        log_tau = (torch.log(inter_time.clamp(1e-8)) - self.mean_log_tau) \
                  / self.std_log_tau                         # (B,)
        mark_e  = self.mark_embedding(mark)                  # (B, mark_emb_size)
        return torch.cat([log_tau.unsqueeze(-1), mark_e], dim=-1)  # (B, 1+E)

    # ---- training (parallel over sequence) ----

    def forward_train(self, bert_emb, times, seq_lens):
        """
        Args:
            bert_emb : (B, 768)
            times    : (B, T_max) padded with zeros, absolute times in [0,1]
            seq_lens : (B,) actual sequence lengths
        Returns:
            nll_per_seq : (B,) per-sequence NLL (already negated sign, so minimise)
        """
        B      = bert_emb.size(0)
        T      = times.size(1)          # T_max (padded)
        device = bert_emb.device

        # context_init from BERT
        h = self.bert_proj(bert_emb)    # (B, context_size)

        # Build inter-event gaps: prepend 0, diff → (B, T+1)
        #   position 0 … T-1 are real event gaps; position T is the "EOS gap"
        #   We use a small positive value for the EOS gap since we don't observe it;
        #   the EOS position is handled by the mark loss (EOS mark), not time loss.
        prev    = torch.zeros(B, 1, device=device)
        abs_t   = torch.cat([prev, times], dim=1)   # (B, T+1), abs_t[:,0]=0
        deltas  = abs_t[:, 1:] - abs_t[:, :-1]     # (B, T), deltas for real events

        # Build mark sequence: T real marks + 1 EOS mark = T+1 marks
        marks = torch.ones(B, T + 1, dtype=torch.long, device=device)  # all real
        for i, L in enumerate(seq_lens.tolist()):
            marks[i, int(L)] = self.MARK_EOS   # EOS at step L

        # Run RNN step by step to collect contexts.
        # The "time-shifted context" trick: context at step i predicts event i.
        # We feed [inter_time_{i-1}, mark_{i-1}] to get context for step i.
        # Step 0 is predicted from h_0 = BERT projection (no RNN step needed).
        contexts = [h.unsqueeze(1)]                 # [(B, 1, context_size)]
        h_cur    = h
        for i in range(T + 1):                      # steps 0…T (T+1 steps)
            # input for step i uses (delta_{i-1}, mark_{i-1})
            if i == 0:
                tau_in  = torch.ones(B, device=device)  # dummy first input
                mark_in = torch.full((B,), self.MARK_REAL, dtype=torch.long, device=device)
            else:
                # delta for step i-1 (already computed for i-1 < T; for i-1 == T use 0)
                if i - 1 < T:
                    tau_in = deltas[:, i - 1].clamp(1e-8)
                else:
                    tau_in = torch.ones(B, device=device)
                mark_in = marks[:, i - 1]
            feat = self._input_feature(tau_in, mark_in)  # (B, 1+E)
            if self.rnn_type == 'LSTM':
                h_cur, _ = self.rnn(feat, (h_cur, torch.zeros_like(h_cur)))
            else:
                h_cur = self.rnn(feat, h_cur)            # (B, context_size)
            contexts.append(h_cur.unsqueeze(1))          # (B, 1, ctx)

        context = torch.cat(contexts, dim=1)             # (B, T+2, context_size)
        # context[:, i] predicts event at step i  (step 0…T+1)
        # For real events (steps 0…T-1): use context[:, 0:T]
        # For EOS (step T): use context[:, T]

        # ---- inter-event time NLL (real events only) ----
        ctx_real = context[:, :T]                        # (B, T, ctx)
        dist     = self._get_inter_time_dist(ctx_real)   # LogNormMix over (B, T)
        log_p_t  = dist.log_prob(deltas.clamp(1e-8))     # (B, T)

        mask = torch.zeros(B, T, device=device)
        for i, L in enumerate(seq_lens.tolist()):
            mask[i, :int(L)] = 1.0
        inter_nll = -(log_p_t * mask).sum(dim=-1)        # (B,)

        # ---- survival term at last real event (same as RecurrentTPP.log_prob) ----
        # last_event_idx points to the context position of the last real event
        last_idx = (seq_lens - 1).clamp(min=0).long()    # (B,)
        ctx_last = context[torch.arange(B), last_idx]    # (B, ctx)
        dist_last = self._get_inter_time_dist(ctx_last.unsqueeze(1))
        # evaluate survival at a large time (practically infinite) relative to [0,1] scale
        t_inf    = torch.ones(B, 1, device=device) * 1.0
        log_surv = dist_last.log_survival_function(t_inf).squeeze(1)  # (B,)
        surv_nll = -log_surv                                            # (B,)

        # ---- mark NLL (all T+1 steps) ----
        ctx_marks   = context[:, :T + 1]                 # (B, T+1, ctx)
        mark_logits = torch.log_softmax(
            self.mark_linear(ctx_marks), dim=-1
        )                                                # (B, T+1, 2)
        mark_dist   = Categorical(logits=mark_logits)
        log_p_m     = mark_dist.log_prob(marks)          # (B, T+1)

        mark_mask = torch.zeros(B, T + 1, device=device)
        for i, L in enumerate(seq_lens.tolist()):
            mark_mask[i, :int(L) + 1] = 1.0             # real events + EOS
        mark_nll = -(log_p_m * mark_mask).sum(dim=-1)   # (B,)

        return inter_nll + surv_nll + mark_nll           # (B,)

    # ---- inference ----

    @torch.no_grad()
    def generate(self, bert_embs, max_len=500, stop_thresh=0.5):
        """
        Autoregressive generation conditioned on bert_embs (B, 768).
        Stops when mark=EOS is sampled (or max_len reached).
        Returns list of (T_i,) tensors of absolute times in [0,1].
        """
        B      = bert_embs.size(0)
        device = bert_embs.device

        h       = self.bert_proj(bert_embs)                 # (B, context_size)
        t_cur   = torch.zeros(B, device=device)             # accumulated time
        tau_in  = torch.ones(B, device=device)              # initial dummy inter-time
        mark_in = torch.full((B,), self.MARK_REAL,
                             dtype=torch.long, device=device)
        alive   = torch.ones(B, dtype=torch.bool, device=device)
        times_list = [[] for _ in range(B)]

        for _ in range(max_len + 1):
            feat   = self._input_feature(tau_in, mark_in)   # (B, 1+E)
            if self.rnn_type == 'LSTM':
                h, _ = self.rnn(feat, (h, torch.zeros_like(h)))
            else:
                h = self.rnn(feat, h)                        # (B, context_size)

            # sample inter-time
            dist    = self._get_inter_time_dist(h.unsqueeze(1))  # over (B, 1)
            delta   = dist.sample().squeeze(1).clamp(1e-8)       # (B,)

            # sample mark
            mlogits = torch.log_softmax(
                self.mark_linear(h), dim=-1
            )                                                # (B, 2)
            next_mark = Categorical(logits=mlogits).sample()# (B,)

            t_cur = t_cur + delta

            for i in range(B):
                if not alive[i]:
                    continue
                if next_mark[i].item() == self.MARK_EOS:
                    alive[i] = False
                else:
                    times_list[i].append(t_cur[i].item())

            if not alive.any():
                break

            tau_in  = delta.detach()
            mark_in = next_mark

        return [torch.tensor(ts, dtype=torch.float32) for ts in times_list]


# ============================================================
# Public interface
# ============================================================

def add_args(parser):
    parser.add_argument('--data', default='dataset/APS_burst.pt')
    parser.add_argument('--bert_model', default='pretrained/bert-base-chinese')
    parser.add_argument('--bert_device', default='cpu')
    parser.add_argument('--bert_cache', default='')
    parser.add_argument('--max_events', type=int, default=500)
    parser.add_argument('--context_size', type=int, default=64,
                        help='RNN hidden / context size (original default 32)')
    parser.add_argument('--mark_emb_size', type=int, default=32,
                        help='Mark embedding size (same as original)')
    parser.add_argument('--n_mix', type=int, default=16,
                        help='Number of LogNormMix components (original default 16)')
    parser.add_argument('--rnn_type', default='GRU', choices=['GRU', 'LSTM', 'RNN'])
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--stop_thresh', type=float, default=0.5)
    parser.add_argument('--outdir', default='runs_ifltpp')
    parser.add_argument('--max_steps', type=int, default=30000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--n_val_samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')


def _compute_inter_time_stats(dataset, train_indices):
    """Compute mean/std of log inter-event times from training split (as in
    dpp.data.dataset.get_inter_time_statistics)."""
    all_log_taus = []
    for i in train_indices:
        t = dataset.time_seqs[i]
        if len(t) < 2:
            continue
        deltas = t[1:] - t[:-1]
        log_d  = torch.log(deltas.clamp(1e-8))
        all_log_taus.append(log_d)
    if not all_log_taus:
        return 0.0, 1.0
    cat = torch.cat(all_log_taus)
    return float(cat.mean()), float(cat.std().clamp(min=1e-6))


def _w1(gen_lens, ref_lens):
    g = np.array(gen_lens, dtype=float)
    r = np.array(ref_lens, dtype=float)
    if len(g) == 0 or len(r) == 0:
        return 1.0
    mean_r = max(r.mean(), 1)
    return float(wasserstein_distance(g / mean_r, r / mean_r))


def train(args):
    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    print(f'Model: IFLTPP  data={args.data}')

    print('Loading dataset...')
    dataset = BertSeqDataset(
        args.data, bert_name=args.bert_model,
        bert_device=args.bert_device, bert_cache=args.bert_cache,
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

    # Compute inter-time statistics from training split (same as original dpp)
    mean_log_tau, std_log_tau = _compute_inter_time_stats(dataset, train_set.indices)
    print(f'Inter-time stats: mean_log_tau={mean_log_tau:.4f}  std={std_log_tau:.4f}')

    ref_lens  = [len(dataset.time_seqs[i]) for i in val_set.indices]
    n_eval    = min(args.n_val_samples, len(val_set))
    val_bert  = torch.stack(
        [dataset.bert_embs[val_set.indices[i]] for i in range(n_eval)]
    )
    print(f'Data: {n} cascades  train={n_train} val={n_val} test={n_test}')
    print(f'Ref lengths: mean={np.mean(ref_lens):.1f}  median={np.median(ref_lens):.0f}')

    model = IFLTPPSeqModel(
        bert_dim=768,
        context_size=args.context_size,
        mark_emb_size=args.mark_emb_size,
        n_mix=args.n_mix,
        rnn_type=args.rnn_type,
        dropout=args.dropout,
    ).to(args.device)
    model.set_inter_time_stats(mean_log_tau, std_log_tau)
    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = torch.optim.Adam(
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
        nll = model.forward_train(bert_emb, times, seq_lens)
        loss = nll.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % 100 == 0:
            print(f'[Step {step:5d}] NLL={loss.item():.4f}')

        if step % args.eval_every == 0:
            model.eval()
            gen_seqs = model.generate(
                val_bert[:n_eval].to(args.device),
                max_len=args.max_events,
                stop_thresh=args.stop_thresh,
            )
            gen_lens = [len(s) for s in gen_seqs]
            wd       = _w1(gen_lens, ref_lens[:n_eval])
            print(f'  [Eval] gen_mean={np.mean(gen_lens):.1f}  '
                  f'gen_median={np.median(gen_lens):.0f}  '
                  f'ref_mean={np.mean(ref_lens):.1f}  W1={wd:.4f}')

            if wd < best_wd:
                best_wd = wd
                torch.save({
                    'step': step, 'model': model.state_dict(),
                    'args': vars(args), 'best_wd': best_wd,
                    'stats': dataset.stats,
                    'mean_log_tau': mean_log_tau, 'std_log_tau': std_log_tau,
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
    model = IFLTPPSeqModel(
        bert_dim=768,
        context_size=saved.get('context_size', args.context_size),
        mark_emb_size=saved.get('mark_emb_size', args.mark_emb_size),
        n_mix=saved.get('n_mix', args.n_mix),
        rnn_type=saved.get('rnn_type', args.rnn_type),
        dropout=saved.get('dropout', args.dropout),
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.set_inter_time_stats(
        ckpt.get('mean_log_tau', 0.0),
        ckpt.get('std_log_tau',  1.0),
    )
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
