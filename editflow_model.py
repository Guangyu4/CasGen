"""Transformer model for depth-aware Edit Flow.
Input: events with (time, depth, parent_time). Output: LogRate with depth/parent_time predictions.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from editflow import DataBatch, LogRate


class SinusoidalEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[..., None] * freqs
        return torch.cat([args.sin(), args.cos()], dim=-1)


class EditFlowTransformer(nn.Module):
    def __init__(self, hidden_dim=64, n_heads=4, n_layers=4, n_ins_bins=64,
                 d_max=2, t_max=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_ins_bins = n_ins_bins
        self.d_max = d_max
        self.t_max = t_max

        # Flow time conditioning
        self.time_emb = SinusoidalEmb(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim),
        )

        # Per-event input: time + depth + parent_time -> hidden
        self.event_time_emb = SinusoidalEmb(hidden_dim // 2)
        self.depth_emb = nn.Embedding(d_max + 1, hidden_dim // 4)
        self.pt_emb = SinusoidalEmb(hidden_dim // 4)
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Output heads
        self.head_ins_rate = nn.Linear(hidden_dim, 1)
        self.head_del_rate = nn.Linear(hidden_dim, 1)
        self.head_ins_logits = nn.Linear(hidden_dim, n_ins_bins)
        # Context heads: predict current token's own attributes (helps model understand x_t)
        self.head_depth = nn.Linear(hidden_dim, d_max + 1)
        self.head_parent_time = nn.Linear(hidden_dim, 1)
        # Structured insertion type: >0 → child of left, <0 → sibling of left
        self.head_ins_type = nn.Linear(hidden_dim, 1)

    def forward(self, x_t: DataBatch, t: torch.Tensor) -> LogRate:
        dev = x_t.device
        B = x_t.batch_size
        time_seqs = x_t.split_sequences()
        depth_seqs = x_t.split_depth()
        pt_seqs = x_t.split_parent_time()
        max_len = max(len(s) for s in time_seqs)

        # Pad to (B, max_len)
        pad_times = torch.zeros(B, max_len, device=dev)
        pad_depths = torch.zeros(B, max_len, dtype=torch.long, device=dev)
        pad_pts = torch.zeros(B, max_len, device=dev)
        pad_mask = torch.ones(B, max_len, dtype=torch.bool, device=dev)
        for i in range(B):
            n = len(time_seqs[i])
            pad_times[i, :n] = time_seqs[i]
            pad_depths[i, :n] = depth_seqs[i].clamp(max=self.d_max)
            pad_pts[i, :n] = pt_seqs[i]
            pad_mask[i, :n] = False

        # Encode events: concat(time_emb, depth_emb, pt_emb) -> proj
        h_time = self.event_time_emb(pad_times)           # (B, L, H/2)
        h_depth = self.depth_emb(pad_depths)                # (B, L, H/4)
        h_pt = self.pt_emb(pad_pts)                         # (B, L, H/4)
        h = self.input_proj(torch.cat([h_time, h_depth, h_pt], dim=-1))

        # Add flow time conditioning
        t_cond = self.time_mlp(self.time_emb(t))
        h = h + t_cond.unsqueeze(1)

        h = self.transformer(h, src_key_padding_mask=pad_mask)

        # Flatten back to 1D
        outputs = []
        for i in range(B):
            outputs.append(h[i, :len(time_seqs[i])])
        h_flat = torch.cat(outputs, dim=0)

        return LogRate(
            log_lambda_ins=self.head_ins_rate(h_flat).squeeze(-1),
            log_lambda_del=self.head_del_rate(h_flat).squeeze(-1),
            Q_ins_logits=self.head_ins_logits(h_flat),
            depth_logits=self.head_depth(h_flat),
            parent_time_pred=self.head_parent_time(h_flat).squeeze(-1).sigmoid(),
            ins_type_logit=self.head_ins_type(h_flat).squeeze(-1),
            seq_idx=x_t.seq_idx,
        )
