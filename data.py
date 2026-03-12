"""Data classes for cascade diffusion model.
CascadeBatch supports thinning, superposition (add_events), and depth/parent_time tracking.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split


class CascadeSequence:
    def __init__(self, time, depth, parent_time, tmax, text_tokens, device='cpu'):
        self.time = torch.as_tensor(time, dtype=torch.float32, device=device)
        self.depth = torch.as_tensor(depth, dtype=torch.long, device=device)
        self.parent_time = torch.as_tensor(parent_time, dtype=torch.float32, device=device)
        self.tmax = torch.as_tensor(tmax, dtype=torch.float32, device=device) if not isinstance(tmax, torch.Tensor) else tmax
        self.text_tokens = torch.as_tensor(text_tokens, dtype=torch.long, device=device)
        tau = torch.diff(self.time, prepend=torch.tensor([0.0], device=device))
        self.tau = tau

    def __len__(self):
        return len(self.time)


class CascadeBatch:
    def __init__(self, time, tau, depth, parent_time, mask, tmax,
                 unpadded_length, text_tokens=None, text_mask=None,
                 kept=None, D_max=10):
        self.time = time
        self.tau = tau
        self.depth = depth
        self.parent_time = parent_time
        self.mask = mask
        self.tmax = tmax
        self.unpadded_length = unpadded_length
        self.text_tokens = text_tokens
        self.text_mask = text_mask
        self.kept = kept
        self.D_max = D_max

    @property
    def batch_size(self):
        return self.time.shape[0]

    @property
    def seq_len(self):
        return self.time.shape[1]

    def __len__(self):
        return self.batch_size

    def to(self, device):
        self.time = self.time.to(device)
        self.tau = self.tau.to(device)
        self.depth = self.depth.to(device)
        self.parent_time = self.parent_time.to(device)
        self.mask = self.mask.to(device)
        self.tmax = self.tmax.to(device)
        self.unpadded_length = self.unpadded_length.to(device)
        if self.text_tokens is not None:
            self.text_tokens = self.text_tokens.to(device)
        if self.text_mask is not None:
            self.text_mask = self.text_mask.to(device)
        if self.kept is not None:
            self.kept = self.kept.to(device)
        return self

    @staticmethod
    def from_sequence_list(sequences, D_max=10):
        device = sequences[0].time.device
        tmax = max(s.tmax for s in sequences)
        if not isinstance(tmax, torch.Tensor):
            tmax = torch.tensor(tmax, device=device)

        lengths = torch.tensor([len(s) for s in sequences], device=device)
        max_len = int(lengths.max().item())

        time = torch.zeros(len(sequences), max_len, device=device)
        tau = torch.zeros(len(sequences), max_len, device=device)
        depth = torch.zeros(len(sequences), max_len, dtype=torch.long, device=device)
        parent_time = torch.zeros(len(sequences), max_len, device=device)
        mask = torch.zeros(len(sequences), max_len, dtype=torch.bool, device=device)

        for i, s in enumerate(sequences):
            n = len(s)
            time[i, :n] = s.time
            tau[i, :n] = s.tau
            depth[i, :n] = s.depth.clamp(max=D_max)
            parent_time[i, :n] = s.parent_time
            mask[i, :n] = True

        # Pad text tokens
        text_tokens = pad_sequence(
            [s.text_tokens for s in sequences], batch_first=True, padding_value=0
        )
        text_mask = text_tokens > 0

        return CascadeBatch(
            time=time, tau=tau, depth=depth, parent_time=parent_time,
            mask=mask, tmax=tmax, unpadded_length=lengths,
            text_tokens=text_tokens, text_mask=text_mask, D_max=D_max,
        )

    @staticmethod
    def sort_time(time, mask, depth, parent_time, kept, tmax):
        time_sort = time.clone()
        time_sort[~mask] = 2 * tmax
        idx = torch.argsort(time_sort, dim=-1)
        time = torch.take_along_dim(time, idx, dim=-1)
        mask = torch.take_along_dim(mask, idx, dim=-1)
        depth = torch.take_along_dim(depth, idx, dim=-1)
        parent_time = torch.take_along_dim(parent_time, idx, dim=-1)
        if kept is not None:
            kept = torch.take_along_dim(kept, idx, dim=-1)
        time = time * mask
        return time, mask, depth, parent_time, kept

    @staticmethod
    def remove_unnecessary_padding(time, mask, depth, parent_time, kept, tmax,
                                   text_tokens=None, text_mask=None, D_max=10):
        time, mask, depth, parent_time, kept = CascadeBatch.sort_time(
            time, mask, depth, parent_time, kept, tmax
        )
        max_length = max(mask.sum(-1).max().int().item(), 1)
        time = time[:, :max_length + 1]
        mask = mask[:, :max_length + 1]
        depth = depth[:, :max_length + 1]
        parent_time = parent_time[:, :max_length + 1]
        if kept is not None:
            kept = kept[:, :max_length + 1]

        time_tau = torch.where(mask, time, tmax)
        tau = torch.diff(time_tau, prepend=torch.zeros_like(time_tau[:, :1]), dim=-1)
        tau = tau * mask

        return CascadeBatch(
            time=time, tau=tau, depth=depth, parent_time=parent_time,
            mask=mask, tmax=tmax, unpadded_length=mask.sum(-1).long(),
            text_tokens=text_tokens, text_mask=text_mask, kept=kept, D_max=D_max,
        )

    def thin(self, alpha):
        if alpha.dim() == 1:
            keep = torch.bernoulli(alpha.unsqueeze(1).expand(-1, self.seq_len)).bool()
        elif alpha.dim() == 2:
            keep = torch.bernoulli(alpha).bool()
        else:
            raise ValueError("alpha must be 1D or 2D")

        keep_mask = self.mask & keep
        rem_mask = self.mask & ~keep

        kept_batch = CascadeBatch.remove_unnecessary_padding(
            time=self.time * keep_mask, mask=keep_mask,
            depth=self.depth * keep_mask, parent_time=self.parent_time * keep_mask,
            kept=self.kept * keep_mask if self.kept is not None else None,
            tmax=self.tmax, text_tokens=self.text_tokens,
            text_mask=self.text_mask, D_max=self.D_max,
        )
        rem_batch = CascadeBatch.remove_unnecessary_padding(
            time=self.time * rem_mask, mask=rem_mask,
            depth=self.depth * rem_mask, parent_time=self.parent_time * rem_mask,
            kept=self.kept * rem_mask if self.kept is not None else None,
            tmax=self.tmax, text_tokens=self.text_tokens,
            text_mask=self.text_mask, D_max=self.D_max,
        )
        return kept_batch, rem_batch

    def add_events(self, other):
        assert len(other) == len(self)
        other = other.to(self.time.device)
        tmax = max(self.tmax, other.tmax)

        if self.kept is None:
            kept = torch.cat([
                torch.ones_like(self.time, dtype=torch.bool),
                torch.zeros_like(other.time, dtype=torch.bool),
            ], dim=-1)
        else:
            kept = torch.cat([
                self.kept,
                torch.zeros_like(other.time, dtype=torch.bool),
            ], dim=-1)

        return CascadeBatch.remove_unnecessary_padding(
            time=torch.cat([self.time, other.time], dim=-1),
            mask=torch.cat([self.mask, other.mask], dim=-1),
            depth=torch.cat([self.depth, other.depth], dim=-1),
            parent_time=torch.cat([self.parent_time, other.parent_time], dim=-1),
            kept=kept, tmax=tmax,
            text_tokens=self.text_tokens, text_mask=self.text_mask,
            D_max=self.D_max,
        )

    def to_list(self):
        result = []
        for i in range(self.batch_size):
            m = self.mask[i]
            result.append({
                'time': self.time[i][m].detach().cpu(),
                'depth': self.depth[i][m].detach().cpu(),
                'parent_time': self.parent_time[i][m].detach().cpu(),
            })
        return result


def generate_hpp(tmax, n_sequences, intensity=None, D_max=10):
    device = tmax.device
    if intensity is None:
        intensity = torch.ones(n_sequences, device=device)

    n_samples = torch.poisson(tmax * intensity)
    max_samples = int(torch.max(n_samples).item()) + 1
    max_samples = max(max_samples, 1)

    times = torch.rand((n_sequences, max_samples), device=device) * tmax
    mask = torch.arange(0, max_samples, device=device)[None, :] < n_samples[:, None]
    times = times * mask

    depth = torch.randint(0, D_max + 1, (n_sequences, max_samples), device=device)
    parent_time = torch.rand((n_sequences, max_samples), device=device) * tmax

    depth = depth * mask
    parent_time = parent_time * mask

    return CascadeBatch.remove_unnecessary_padding(
        time=times, mask=mask, depth=depth, parent_time=parent_time,
        kept=None, tmax=tmax, D_max=D_max,
    )


class CascadeDataset(Dataset):
    def __init__(self, processed_path, max_text_len=128, max_events=500, bert_path=None):
        data = torch.load(processed_path, map_location='cpu', weights_only=False)
        self.cascades = data['cascades']
        self.vocab = data['vocab']
        self.stats = data['stats']
        self.max_text_len = max_text_len
        self.max_events = max_events
        self.D_max = self.stats['D_max']
        self.n_max = min(self.stats['n_max'], max_events)

        if bert_path:
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        else:
            self.tokenizer = None

    def __len__(self):
        return len(self.cascades)

    def _tokenize(self, text):
        if self.tokenizer is not None:
            enc = self.tokenizer(
                text, max_length=self.max_text_len, truncation=True,
                padding=False, return_tensors='pt',
            )
            return enc['input_ids'].squeeze(0)
        unk = self.vocab.get('<UNK>', 1)
        tokens = [self.vocab.get(ch, unk) for ch in text[:self.max_text_len]]
        if not tokens:
            tokens = [unk]
        return torch.tensor(tokens, dtype=torch.long)

    def __getitem__(self, idx):
        c = self.cascades[idx]
        text_tokens = self._tokenize(c['text'])
        times = c['times'][:self.max_events]
        depths = c['depths'][:self.max_events]
        ptimes = c['parent_times'][:self.max_events]
        return CascadeSequence(
            time=times, depth=depths,
            parent_time=ptimes,
            tmax=1.0,
            text_tokens=text_tokens,
        )


def get_dataloaders(processed_path, batch_size=64, train_ratio=0.8,
                    val_ratio=0.1, seed=42, max_text_len=128, D_max=10,
                    max_events=500, bert_path=None):
    dataset = CascadeDataset(processed_path, max_text_len=max_text_len,
                             max_events=max_events, bert_path=bert_path)
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    def collate(seqs):
        return CascadeBatch.from_sequence_list(seqs, D_max=D_max)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=min(batch_size * 4, n_val),
                            shuffle=False, collate_fn=collate, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=min(batch_size * 4, n_test),
                             shuffle=False, collate_fn=collate, num_workers=0)

    return train_loader, val_loader, test_loader, dataset.stats, dataset.vocab
