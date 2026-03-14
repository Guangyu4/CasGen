"""Depth-aware Edit Flow for cascade tree generation.
Events are (time, depth, parent_time) triples. Alignment and edit ops work on time;
depth and parent_time follow as attached attributes.
"""
import numba
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from torch import Tensor

EPSILON = -float('inf')
D_MAX = 2
GAMMA = 0.3


# ---- DataBatch ----

@dataclass
class DataBatch:
    seq_lens: tuple
    x: Tensor           # (n,) event times
    depth: Tensor        # (n,) event depth (long)
    parent_time: Tensor  # (n,) parent event time (float)

    @property
    def device(self):
        return self.x.device

    @property
    def batch_size(self):
        return len(self.seq_lens)

    @property
    def seq_lens_tensor(self):
        return torch.tensor(self.seq_lens, dtype=torch.long, device=self.device)

    @property
    def seq_idx(self):
        return torch.repeat_interleave(
            torch.arange(self.batch_size, device=self.device),
            self.seq_lens_tensor, output_size=sum(self.seq_lens),
        )

    @property
    def token_pos(self):
        shifts = F.pad(self.seq_lens_tensor[:-1].cumsum(0), (1, 0))
        return torch.arange(len(self.x), device=self.device) - shifts[self.seq_idx]

    def split_sequences(self):
        return torch.split(self.x, list(self.seq_lens), dim=0)

    def split_depth(self):
        return torch.split(self.depth, list(self.seq_lens), dim=0)

    def split_parent_time(self):
        return torch.split(self.parent_time, list(self.seq_lens), dim=0)

    @staticmethod
    def from_sequences(time_seqs, depth_seqs=None, pt_seqs=None):
        time_seqs = [torch.as_tensor(s, dtype=torch.float32) for s in time_seqs]
        x = torch.cat(time_seqs) if time_seqs else torch.tensor([], dtype=torch.float32)
        n = len(x)
        if depth_seqs is not None:
            depth = torch.cat([torch.as_tensor(d, dtype=torch.long) for d in depth_seqs])
        else:
            depth = torch.zeros(n, dtype=torch.long)
        if pt_seqs is not None:
            pt = torch.cat([torch.as_tensor(p, dtype=torch.float32) for p in pt_seqs])
        else:
            pt = torch.zeros(n, dtype=torch.float32)
        return DataBatch(
            seq_lens=tuple(len(s) for s in time_seqs), x=x, depth=depth, parent_time=pt,
        )

    def wrap(self, prefix=0.0, suffix=1.0):
        dev, dt = self.device, self.x.dtype
        pv = torch.tensor([prefix], device=dev, dtype=dt)
        sv = torch.tensor([suffix], device=dev, dtype=dt)
        d0 = torch.tensor([0], device=dev, dtype=torch.long)
        p0 = torch.tensor([0.0], device=dev, dtype=dt)
        ts, ds, ps = [], [], []
        for t, d, p in zip(self.split_sequences(), self.split_depth(), self.split_parent_time()):
            ts.append(torch.cat([pv, t, sv]))
            ds.append(torch.cat([d0, d, d0]))
            ps.append(torch.cat([p0, p, p0]))
        return DataBatch.from_sequences(ts, ds, ps)

    def unwrap(self):
        ts, ds, ps = [], [], []
        for t, d, p in zip(self.split_sequences(), self.split_depth(), self.split_parent_time()):
            ts.append(t[1:-1])
            ds.append(d[1:-1])
            ps.append(p[1:-1])
        return DataBatch.from_sequences(ts, ds, ps)

    def to(self, device):
        return DataBatch(
            seq_lens=self.seq_lens, x=self.x.to(device),
            depth=self.depth.to(device), parent_time=self.parent_time.to(device),
        )


# ---- Alignment (Needleman-Wunsch) ----
# Alignment is purely on time values; depth/parent_time follow the same alignment indices.

@numba.jit(nopython=True)
def _align_pair(x0, x1, eps, delta):
    n, m = len(x0), len(x1)
    c_insdel = (delta ** 2) / 2
    D = np.zeros((n + 1, m + 1), dtype=np.float64)
    bp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        D[i, 0] = D[i - 1, 0] + c_insdel
    for j in range(1, m + 1):
        D[0, j] = D[0, j - 1] + c_insdel
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(x0[i - 1] - x1[j - 1])
            if d < delta and (i == 1 or x1[j - 1] > x0[i - 2]) and (i == n or x1[j - 1] < x0[i]):
                c_match = D[i - 1, j - 1] + d * d
            else:
                c_match = np.inf
            c_del = D[i - 1, j] + c_insdel
            c_ins = D[i, j - 1] + c_insdel
            best = c_match
            b = 0
            if c_del < best:
                best = c_del
                b = 1
            if c_ins < best:
                best = c_ins
                b = 2
            D[i, j] = best
            bp[i, j] = b
    z0 = np.full(n + m, eps)
    z1 = np.full(n + m, eps)
    i, j, k = n, m, n + m - 1
    while i > 0 and j > 0:
        if bp[i, j] == 0:
            z0[k] = x0[i - 1]; z1[k] = x1[j - 1]; i -= 1; j -= 1
        elif bp[i, j] == 1:
            z0[k] = x0[i - 1]; i -= 1
        else:
            z1[k] = x1[j - 1]; j -= 1
        k -= 1
    while i > 0:
        z0[k] = x0[i - 1]; i -= 1; k -= 1
    while j > 0:
        z1[k] = x1[j - 1]; j -= 1; k -= 1
    return z0[k + 1:], z1[k + 1:]


def _align_attrs(z_time, original_time, original_attr, eps_val):
    """Map aligned time back to original attributes. Non-eps positions get their original attr."""
    out = np.full(len(z_time), eps_val, dtype=np.float64)
    orig_idx = 0
    for i in range(len(z_time)):
        if z_time[i] != EPSILON:
            if orig_idx < len(original_attr):
                out[i] = original_attr[orig_idx]
            orig_idx += 1
    return out


def align_batch(x0_batch, x1_batch, delta=0.01):
    """Align time sequences; also produce aligned depth and parent_time."""
    B = x0_batch.batch_size
    z0_ts, z1_ts = [], []
    z0_ds, z1_ds = [], []
    z0_ps, z1_ps = [], []

    for i, (s0, s1) in enumerate(zip(x0_batch.split_sequences(), x1_batch.split_sequences())):
        s0_np, s1_np = s0.cpu().numpy(), s1.cpu().numpy()
        z0_t, z1_t = _align_pair(s0_np, s1_np, EPSILON, delta)
        z0_ts.append(z0_t)
        z1_ts.append(z1_t)

        d0 = x0_batch.split_depth()[i].cpu().numpy().astype(np.float64)
        d1 = x1_batch.split_depth()[i].cpu().numpy().astype(np.float64)
        z0_ds.append(_align_attrs(z0_t, s0_np, d0, EPSILON))
        z1_ds.append(_align_attrs(z1_t, s1_np, d1, EPSILON))

        p0 = x0_batch.split_parent_time()[i].cpu().numpy()
        p1 = x1_batch.split_parent_time()[i].cpu().numpy()
        z0_ps.append(_align_attrs(z0_t, s0_np, p0, EPSILON))
        z1_ps.append(_align_attrs(z1_t, s1_np, p1, EPSILON))

    aug_len = max(len(z) for z in z0_ts + z1_ts)

    def pad_stack(arrs):
        return np.stack([np.pad(a, (0, aug_len - len(a)), constant_values=EPSILON) for a in arrs])

    dev = x0_batch.device
    to_t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
    return (
        to_t(pad_stack(z0_ts)), to_t(pad_stack(z1_ts)),
        to_t(pad_stack(z0_ds)), to_t(pad_stack(z1_ds)),
        to_t(pad_stack(z0_ps)), to_t(pad_stack(z1_ps)),
    )


# ---- Depth-aware probability path ----

def sample_zt(t, z0, z1, z1_depth=None, gamma=GAMMA, d_max=D_MAX, generator=None):
    """z_t[i] = z_1[i] with prob kappa(t, depth), else z_0[i].
    Root nodes (depth=0) converge faster; leaves converge slower.
    """
    if z1_depth is not None and gamma > 0:
        depth_safe = z1_depth.clone()
        depth_safe[z1_depth == EPSILON] = 0
        exponent = 1.0 + gamma * depth_safe.clamp(min=0) / max(d_max, 1)
        kappa = t[:, None] ** exponent
    else:
        kappa = t[:, None]
    mask = torch.rand(z1.shape, device=z1.device, generator=generator) <= kappa
    return torch.where(mask, z1, z0)


def rm_blanks(z_t, z_depth=None, z_pt=None):
    exists = z_t != EPSILON
    seq_lens = exists.sum(dim=1).tolist()
    x = z_t[exists]
    d = z_depth[exists].long() if z_depth is not None else torch.zeros(len(x), dtype=torch.long, device=x.device)
    p = z_pt[exists] if z_pt is not None else torch.zeros(len(x), device=x.device)
    return DataBatch(seq_lens=tuple(seq_lens), x=x, depth=d, parent_time=p)


def rm_blanks_idx(z_t):
    non_blank = z_t != EPSILON
    within = non_blank.cumsum(dim=-1) - 1
    counts = non_blank.sum(dim=-1)
    offsets = counts.roll(1).cumsum(dim=-1) - counts[-1]
    global_idx = within + offsets[:, None]
    return torch.where(within == -1, -1, global_idx)


# ---- LogRate and Operations ----

@dataclass
class LogRate:
    log_lambda_ins: Tensor
    log_lambda_del: Tensor
    Q_ins_logits: Tensor
    depth_logits: Tensor     # (n, D_max+1) current token's depth, for context understanding
    parent_time_pred: Tensor # (n,) current token's parent_time, for context understanding
    ins_type_logit: Tensor   # (n,) scalar: >0 → child insertion, <0 → sibling insertion
    seq_idx: Tensor

    @property
    def n_ins_bins(self):
        return self.Q_ins_logits.shape[-1]

    def sample_euler_ops(self, h, generator=None):
        dev = self.log_lambda_ins.device
        n = len(self.log_lambda_ins)
        p_ins = (h * self.log_lambda_ins.exp()).clamp(0, 1)
        p_del = (h * self.log_lambda_del.exp()).clamp(0, 1)
        op_ins = torch.rand(n, device=dev, generator=generator) < p_ins
        op_del = torch.rand(n, device=dev, generator=generator) < p_del
        tok_ins = torch.multinomial(
            F.softmax(self.Q_ins_logits, dim=-1), 1, generator=generator
        )[:, 0]
        # Structured insertion type: 0=sibling (same parent as left), 1=child (left is parent)
        ins_type = (self.ins_type_logit > 0).long()
        return op_ins, op_del, tok_ins, ins_type


def apply_ops(x_t, op_ins, op_del, tok_ins, ins_type, n_ins_bins, d_max,
              max_seq_len=None, generator=None):
    """Apply insert and delete operations with structured insertion.

    ins_type: 0 = sibling (same depth/parent as left neighbor)
              1 = child   (left neighbor becomes parent, depth += 1)
    """
    x = x_t.x
    dep = x_t.depth
    pt = x_t.parent_time
    n = len(x)
    slt = x_t.seq_lens_tensor
    last = slt.cumsum(0) - 1
    first = slt.cumsum(0) - slt
    seq_idx = x_t.seq_idx

    op_ins = op_ins.clone()
    op_del = op_del.clone()
    op_ins[last] = False
    op_del[first] = False
    op_del[last] = False

    # Cap insertions for sequences already at max length
    if max_seq_len is not None:
        cur_lens = torch.zeros(x_t.batch_size, dtype=torch.long, device=x.device)
        cur_lens.scatter_add_(0, seq_idx, torch.ones(n, dtype=torch.long, device=x.device))
        op_ins = op_ins & ~(cur_lens >= max_seq_len)[seq_idx]

    interior = torch.ones(n, dtype=torch.bool, device=x.device)
    interior[first] = False
    interior[last] = False

    # Vectorized: prevent all interior tokens of any sequence from being deleted.
    interior_count = torch.zeros(x_t.batch_size, dtype=torch.long, device=x.device)
    interior_count.scatter_add_(0, seq_idx, interior.long())
    del_interior_count = torch.zeros(x_t.batch_size, dtype=torch.long, device=x.device)
    del_interior_count.scatter_add_(0, seq_idx, (op_del & interior).long())
    all_del_seqs = (interior_count > 0) & (del_interior_count >= interior_count)
    if all_del_seqs.any():
        cumsum_interior = torch.cumsum(interior.long(), dim=0)
        seq_start_cumsum = cumsum_interior[first[seq_idx]] - interior[first[seq_idx]].long()
        within_rank = cumsum_interior - seq_start_cumsum
        is_first_interior = interior & (within_rank == 1)
        op_del = op_del & ~(is_first_interior & all_del_seqs[seq_idx])

    n_ins = int(op_ins.sum())

    ins_base = op_ins.nonzero(as_tuple=True)[0]
    ins_idx = ins_base + torch.arange(1, n_ins + 1, device=x.device)
    ins_mask = torch.zeros(n + n_ins, dtype=torch.bool, device=x.device)
    ins_mask[ins_idx] = True
    keep_mask = ~ins_mask

    # Time: interpolate between left and right neighbor
    xt_new = x.new_empty(n + n_ins)
    xt_new[keep_mask] = x
    left_x = x[ins_base]
    right_x = x[ins_base + 1]
    u = torch.rand(n_ins, device=x.device, generator=generator)
    xt_new[ins_mask] = torch.lerp(left_x, right_x, (tok_ins[op_ins] + u) / n_ins_bins)

    # Structured insertion: depth and parent_time derived from tree structure
    #   type=0 sibling: new node shares the same parent as the left neighbor
    #   type=1 child:   new node is a child of the left neighbor
    t_for_ins = ins_type[op_ins]                        # 0 or 1 per insertion
    is_child = (t_for_ins == 1)

    dep_sib = dep[ins_base]                             # same depth as left neighbor
    dep_child = (dep[ins_base] + 1).clamp(max=d_max)   # one level deeper
    pt_sib = pt[ins_base]                               # inherit left neighbor's parent
    pt_child = x[ins_base]                              # left neighbor becomes parent

    dep_new = dep.new_zeros(n + n_ins)
    dep_new[keep_mask] = dep
    dep_new[ins_mask] = torch.where(is_child, dep_child, dep_sib)

    pt_new = pt.new_zeros(n + n_ins)
    pt_new[keep_mask] = pt
    pt_new[ins_mask] = torch.where(is_child, pt_child, pt_sib)

    # Seq idx
    seq_new = seq_idx.new_empty(n + n_ins)
    seq_new[keep_mask] = seq_idx
    seq_new[ins_mask] = seq_idx[op_ins]

    # Delete
    del_new = torch.zeros(n + n_ins, dtype=torch.bool, device=x.device)
    del_new[keep_mask] = op_del
    del_new[ins_mask] = False

    alive = ~del_new
    x_out = xt_new[alive]
    dep_out = dep_new[alive]
    pt_out = pt_new[alive]
    seq_out = seq_new[alive]

    new_lens = torch.zeros(x_t.batch_size, dtype=torch.long, device=x.device)
    new_lens.scatter_add_(0, seq_out, torch.ones_like(seq_out))

    n_del = int(op_del.sum())
    return DataBatch(seq_lens=tuple(new_lens.tolist()), x=x_out, depth=dep_out, parent_time=pt_out), n_ins, n_del


# ---- Interpolation bins ----

def compute_ins_bins(z_t, z_1, n_bins):
    dev = z_t.device
    B, L = z_t.shape
    valid = z_t != EPSILON
    indices = torch.arange(L, device=dev).expand(B, -1)
    left_idx = torch.cummax(torch.where(valid, indices, -1), dim=1)[0]
    right_idx = torch.cummin(torch.where(valid, indices, L).flip(1), dim=1)[0].flip(1)
    gap = ~valid & (left_idx >= 0) & (right_idx < L)
    l_vals = torch.gather(z_t, 1, left_idx.clamp(min=0))
    r_vals = torch.gather(z_t, 1, right_idx.clamp(max=L - 1))
    width = (r_vals - l_vals).clamp(min=1e-6)
    norm = ((z_1 - l_vals) / width).clamp(0, 1 - 1e-6)
    bins = (norm * n_bins).floor().long()
    return torch.where(gap, bins, torch.zeros_like(bins)), gap


# ---- EditFlow core ----

class EditFlow:
    def __init__(self, n_ins_bins=64, delta=0.05, rate_penalty=0.5, gamma=GAMMA, d_max=D_MAX):
        self.n_ins_bins = n_ins_bins
        self.delta = delta
        self.rate_penalty = rate_penalty
        self.gamma = gamma
        self.d_max = d_max

    def compute_loss(self, model, x0_batch, x1_batch, generator=None):
        dev = x1_batch.device
        B = x1_batch.batch_size

        z0_t, z1_t, z0_d, z1_d, z0_p, z1_p = align_batch(x0_batch, x1_batch, delta=self.delta)
        t = torch.rand(B, device=dev, generator=generator)
        z_t = sample_zt(t, z0_t, z1_t, z1_depth=z1_d, gamma=self.gamma, d_max=self.d_max, generator=generator)

        # Depth and parent_time follow the same mixing as time
        z_t_is_z1 = (z_t == z1_t)
        z_d = torch.where(z_t_is_z1, z1_d, z0_d)
        z_p = torch.where(z_t_is_z1, z1_p, z0_p)

        x_t = rm_blanks(z_t, z_d, z_p)
        x_t = x_t.to(dev)
        log_rate = model(x_t, t)

        # Rate penalty
        slt = x_t.seq_lens_tensor
        last_tok = (x_t.token_pos == (slt - 1)[x_t.seq_idx])
        first_tok = (x_t.token_pos == 0)
        not_last = ~last_tok
        not_pad = ~first_tok & ~last_tok

        penalty = torch.zeros(B, device=dev)
        penalty.scatter_add_(0, x_t.seq_idx, log_rate.log_lambda_ins.exp() * not_last)
        penalty.scatter_add_(0, x_t.seq_idx, log_rate.log_lambda_del.exp() * not_pad)

        # Sum of log-rates
        z1_is_eps = z1_t == EPSILON
        zt_is_eps = z_t == EPSILON
        zt_idx = rm_blanks_idx(z_t)

        per_tok_del = z1_is_eps * log_rate.log_lambda_del[zt_idx.clamp(min=0)]

        ins_bins, gap_mask = compute_ins_bins(z_t, z1_t, self.n_ins_bins)
        log_q_ins = F.log_softmax(log_rate.Q_ins_logits, dim=-1)
        per_tok_ins = (zt_is_eps & gap_mask) * (
            log_rate.log_lambda_ins[zt_idx.clamp(min=0)]
            + log_q_ins[zt_idx.clamp(min=0), ins_bins]
        )

        per_tok = ((zt_idx != -1) & (z1_t != z_t)) * (per_tok_del + per_tok_ins)

        # Per-token depth-aware kappa coefficient: d_kappa/dt / (1 - kappa)
        z1_depth_safe = z1_d.clone()
        z1_depth_safe[z1_d == EPSILON] = 0
        a = 1.0 + self.gamma * z1_depth_safe.clamp(min=0) / max(self.d_max, 1)
        kappa = t[:, None] ** a
        kappa_coeff = (a * t[:, None] ** (a - 1)) / (1 - kappa).clamp(min=1e-3)
        sum_log_rates = (kappa_coeff * per_tok).sum(dim=-1)

        loss = self.rate_penalty * penalty - sum_log_rates

        # Auxiliary loss 1: current token attributes (context understanding)
        real_mask = not_pad.float()
        depth_loss = F.cross_entropy(
            log_rate.depth_logits, x_t.depth.clamp(max=self.d_max), reduction='none'
        ) * real_mask
        pt_loss = F.mse_loss(
            log_rate.parent_time_pred, x_t.parent_time, reduction='none'
        ) * real_mask
        ctx_aux = (depth_loss.sum() + pt_loss.sum()) / real_mask.sum().clamp(min=1)

        # Auxiliary loss 2: structured insertion type (sibling=0 vs child=1)
        # For each gap in z_t, compare target depth with left anchor's depth:
        #   target == anchor_dep     → sibling (type=0)
        #   target == anchor_dep + 1 → child   (type=1)
        valid_gaps = (z_t == EPSILON) & (z1_t != EPSILON) & (zt_idx != -1)
        ins_type_aux = torch.tensor(0.0, device=dev)
        if valid_gaps.any():
            anchor_idx = zt_idx[valid_gaps]
            anchor_dep = x_t.depth[anchor_idx].float()
            target_dep = z1_d[valid_gaps].clamp(min=0, max=self.d_max)
            is_sibling = (target_dep == anchor_dep)
            is_child = (target_dep == anchor_dep + 1)
            trainable = is_sibling | is_child
            if trainable.any():
                target_type = is_child[trainable].float()
                ins_type_aux = F.binary_cross_entropy_with_logits(
                    log_rate.ins_type_logit[anchor_idx[trainable]], target_type
                )

        aux = 0.05 * ctx_aux + 0.2 * ins_type_aux
        return loss.mean() + aux

    @torch.no_grad()
    def sample(self, model, noise_batch, n_steps=100, max_seq_len=600, generator=None):
        x_t = noise_batch
        h = 1.0 / n_steps
        dev = x_t.device
        B = x_t.batch_size
        total_ins, total_del = 0, 0

        for k in range(n_steps):
            t_val = k / n_steps
            t = torch.full((B,), t_val, device=dev)
            log_rate = model(x_t, t)
            op_ins, op_del, tok_ins, ins_type = log_rate.sample_euler_ops(h, generator=generator)
            x_t, ni, nd = apply_ops(x_t, op_ins, op_del, tok_ins, ins_type,
                                     self.n_ins_bins, self.d_max,
                                     max_seq_len=max_seq_len, generator=generator)
            total_ins += ni
            total_del += nd

        return x_t.unwrap(), total_ins, total_del

    @staticmethod
    def reconstruct_tree(times, parent_times):
        n = len(times)
        if n == 0:
            return []
        parents = [-1] * n
        for i in range(1, n):
            target = parent_times[i].item() if i < len(parent_times) else 0.0
            best_j, best_dist = 0, float('inf')
            for j in range(i):
                dist = abs(times[j].item() - target)
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
            parents[i] = best_j
        return parents


# ---- DDPMFlow core ----

class DDPMFlow:
    """DDPM-style denoising for event sequences using HPP noise.

    Forward process: at time t in [0,1], corrupt x1 by keeping each real event
    with probability t and adding HPP events at rate lambda_hpp*(1-t).
    At t=0: pure HPP noise (many events). At t=1: clean data.

    Training loss: supervised denoising — predict which events to delete (HPP events)
    and where to insert missing real events, without NW alignment.
    Sampling: same Euler loop as EditFlow, starting from HPP noise (many deletions).
    """

    def __init__(self, n_ins_bins=64, lambda_hpp=50.0, t_max=1.0,
                 rate_penalty=0.5, d_max=D_MAX):
        self.n_ins_bins = n_ins_bins
        self.lambda_hpp = lambda_hpp
        self.t_max = t_max
        self.rate_penalty = rate_penalty
        self.d_max = d_max

    def corrupt(self, x1_batch, t_vals, rng):
        """Forward DDPM process on CPU using numpy rng.

        Returns:
            x_t_seqs: list of np arrays (event times in x_t)
            del_targets: list of bool arrays, True = HPP event (should delete)
            ins_targets: list of bool arrays per gap (True = dropped real event in gap)
        """
        x_t_seqs, del_targets, ins_targets = [], [], []
        for i, seq in enumerate(x1_batch.split_sequences()):
            events = seq.cpu().numpy()
            n = len(events)
            t = float(t_vals[i].item())

            # Keep each real event with prob t
            keep = rng.random(n) < t
            survived = events[keep]
            dropped = events[~keep]

            # Add HPP events: Poisson(lambda_hpp * t_max * (1-t))
            rate = self.lambda_hpp * self.t_max * max(1.0 - t, 0.0)
            n_hpp = rng.poisson(rate)
            hpp = rng.uniform(0.0, self.t_max, n_hpp)

            combined = np.concatenate([survived, hpp])
            is_hpp = np.concatenate([
                np.zeros(len(survived), dtype=bool),
                np.ones(n_hpp, dtype=bool),
            ])
            order = np.argsort(combined, kind='stable')
            combined = combined[order]
            is_hpp = is_hpp[order]

            # Insertion targets: for each event position i in x_t,
            # should a real event be inserted in the gap AFTER position i?
            # (gap between combined[i] and combined[i+1], or after last event)
            n_xt = len(combined)
            ins_tgt = np.zeros(n_xt, dtype=bool)
            for d_t in dropped:
                # Find gap index: the position just to the left of where d_t would go
                idx = int(np.searchsorted(combined, d_t, side='right')) - 1
                idx = max(0, min(idx, n_xt - 1))
                ins_tgt[idx] = True

            x_t_seqs.append(combined)
            del_targets.append(is_hpp)
            ins_targets.append(ins_tgt)
        return x_t_seqs, del_targets, ins_targets

    def compute_loss(self, model, x1_batch, generator=None):
        dev = x1_batch.device
        B = x1_batch.batch_size

        t = torch.rand(B, device='cpu')  # keep on CPU for numpy ops

        seed = int(torch.randint(0, 2**31, (1,)).item())
        rng = np.random.default_rng(seed)
        x_t_seqs, del_targets_np, ins_targets_np = self.corrupt(x1_batch, t, rng)

        # Build x_t DataBatch (flat: depth=0, parent_time=0)
        x_t = DataBatch.from_sequences(x_t_seqs).to(dev)
        x_t = x_t.wrap(prefix=0.0, suffix=self.t_max)

        t_dev = t.to(dev)
        log_rate = model(x_t, t_dev)

        slt = x_t.seq_lens_tensor
        first_tok = (x_t.token_pos == 0)
        last_tok = (x_t.token_pos == (slt - 1)[x_t.seq_idx])
        not_last = ~last_tok
        not_pad = ~first_tok & ~last_tok

        # Rate penalty (same as EditFlow)
        penalty = torch.zeros(B, device=dev)
        penalty.scatter_add_(0, x_t.seq_idx, log_rate.log_lambda_ins.exp() * not_last)
        penalty.scatter_add_(0, x_t.seq_idx, log_rate.log_lambda_del.exp() * not_pad)

        # Deletion supervision: HPP events (not_pad positions only) should be deleted
        del_tgt_list, ins_tgt_list = [], []
        for del_np, ins_np in zip(del_targets_np, ins_targets_np):
            # wrap adds prefix+suffix tokens → pad with False on both ends
            del_tgt_list.append(np.concatenate([[False], del_np, [False]]))
            ins_tgt_list.append(np.concatenate([[False], ins_np, [False]]))

        del_tgt = torch.tensor(
            np.concatenate(del_tgt_list), dtype=torch.float32, device=dev
        )
        ins_tgt = torch.tensor(
            np.concatenate(ins_tgt_list), dtype=torch.float32, device=dev
        )

        del_loss = F.binary_cross_entropy_with_logits(
            log_rate.log_lambda_del[not_pad], del_tgt[not_pad]
        )
        ins_loss = F.binary_cross_entropy_with_logits(
            log_rate.log_lambda_ins[not_last], ins_tgt[not_last]
        )

        loss = self.rate_penalty * penalty.mean() + del_loss + ins_loss
        return loss

    @torch.no_grad()
    def sample(self, model, noise_batch, n_steps=200, max_seq_len=600, generator=None):
        """Same Euler loop as EditFlow.sample; starts from HPP noise (many events)."""
        x_t = noise_batch
        h = 1.0 / n_steps
        dev = x_t.device
        B = x_t.batch_size
        total_ins, total_del = 0, 0

        for k in range(n_steps):
            t_val = k / n_steps
            t = torch.full((B,), t_val, device=dev)
            log_rate = model(x_t, t)
            op_ins, op_del, tok_ins, ins_type = log_rate.sample_euler_ops(h, generator=generator)
            x_t, ni, nd = apply_ops(x_t, op_ins, op_del, tok_ins, ins_type,
                                     self.n_ins_bins, self.d_max,
                                     max_seq_len=max_seq_len, generator=generator)
            total_ins += ni
            total_del += nd

        return x_t.unwrap(), total_ins, total_del
