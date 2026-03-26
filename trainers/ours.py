"""OURS model trainer: Edit Flow cascade generation.

Variants (only active when --model OURS):
  uncond  - tree noise + Flow Matching, no conditioning
  bert    - tree noise + Flow Matching + BERT text conditioning
  motive  - tree noise + Flow Matching + 9-dim burst score conditioning
  flat    - flat Poisson noise, no tree structure  (ablation: W/o Tree)
  ddpm    - HPP noise + DDPM-style denoising       (ablation: W/o Flowmatching)
"""
import functools
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.stats import wasserstein_distance
from editflow import DataBatch, EditFlow, DDPMFlow, D_MAX, GAMMA
from editflow_model import EditFlowTransformer


# ---- Raw cascade processing helpers ----

def _process_cascade_raw(cascade):
    root_user = cascade['root_user']
    nodes = cascade['nodes']
    t_max = cascade.get('t_max', 1.0)
    if t_max <= 0:
        t_max = 1.0
    user_times = {}
    times, depths, parent_times = [], [], []
    for node in nodes:
        uid = node['user_id']
        t = node['time'] / t_max
        depth = len(node['path']) - 1
        puid = node['parent_user_id']
        if puid is None or puid == root_user:
            pt = 0.0
        elif puid in user_times:
            cands = [tp for tp in user_times[puid] if tp <= t]
            pt = max(cands) if cands else 0.0
        else:
            pt = 0.0
        times.append(t)
        depths.append(depth)
        parent_times.append(pt)
        user_times.setdefault(uid, []).append(t)
    return {
        'times': torch.tensor(times, dtype=torch.float32),
        'depths': torch.tensor(depths, dtype=torch.long),
        'parent_times': torch.tensor(parent_times, dtype=torch.float32),
    }


def _encode_bert(texts, bert_name='bert-base-uncased', device='cpu', batch_size=64):
    from transformers import AutoTokenizer, AutoModel
    print(f'Loading BERT model: {bert_name}')
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    bert_model = AutoModel.from_pretrained(bert_name).to(device)
    bert_model.eval()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = bert_model(**enc)
        all_embs.append(out.last_hidden_state[:, 0, :].cpu())
        if (i // batch_size) % 10 == 0:
            print(f'  BERT encoding {i}/{len(texts)}...')
    del bert_model
    if device != 'cpu':
        torch.cuda.empty_cache()
    embs = torch.cat(all_embs, dim=0)
    print(f'BERT encoding done: {embs.shape}')
    return list(embs)


# ---- Datasets ----

class CascadeSeqDataset(Dataset):
    def __init__(self, processed_path, max_events=500):
        data = torch.load(processed_path, map_location='cpu', weights_only=False)
        self.items = []
        for c in data['cascades']:
            if len(c['times']) == 0:
                continue
            t = c['times'][:max_events]
            d = c['depths'][:max_events]
            p = c['parent_times'][:max_events]
            self.items.append((t.tolist(), d.tolist(), p.tolist()))
        self.stats = data['stats']
        counts = [len(item[0]) for item in self.items]
        hist = torch.zeros(max(counts) + 1)
        for c in counts:
            hist[c] += 1
        self.count_dist = hist / hist.sum()
        self.t_max = 1.0
        self.d_max = self.stats['D_max']
        self.cond_dim = 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class CondCascadeDataset(Dataset):
    """Conditioning dataset.

    cond_type='bert':   768-dim BERT CLS embedding (raw cascade format)
    cond_type='motive': 9-dim burst scores from *_burst.pt processed format
    """
    COND_DIM = {'bert': 768, 'motive': 9}

    def __init__(self, cond_path, cond_type, max_events=500,
                 bert_name='bert-base-uncased', bert_device='cpu'):
        assert cond_type in self.COND_DIM, f'Unknown cond_type: {cond_type}'
        self.items, self.conds = [], []

        if cond_type == 'motive':
            data = torch.load(cond_path, map_location='cpu', weights_only=False)
            for c in data['cascades']:
                if len(c['times']) == 0:
                    continue
                t = c['times'][:max_events].tolist()
                if not t:
                    continue
                self.items.append((
                    t,
                    c['depths'][:max_events].tolist(),
                    c['parent_times'][:max_events].tolist(),
                ))
                self.conds.append(c.get('burst_scores', torch.zeros(9)).float())
            self.stats = data['stats']
            counts = [len(it[0]) for it in self.items]
            all_d = [d for _, ds, _ in self.items for d in ds]
            self.d_max = self.stats.get('D_max', max(all_d) if all_d else 2)
        else:
            texts = []
            raw = torch.load(cond_path, map_location='cpu', weights_only=False)
            for c in raw:
                if len(c.get('nodes', [])) == 0:
                    continue
                proc = _process_cascade_raw(c)
                t = proc['times'][:max_events].tolist()
                if not t:
                    continue
                self.items.append((
                    t,
                    proc['depths'][:max_events].tolist(),
                    proc['parent_times'][:max_events].tolist(),
                ))
                texts.append(c.get('text', ''))
            self.conds = _encode_bert(texts, bert_name, bert_device)
            counts = [len(it[0]) for it in self.items]
            all_d = [d for _, ds, _ in self.items for d in ds]
            self.d_max = max(all_d) if all_d else 2
            self.stats = {
                'D_max': self.d_max, 'n_max': max(counts),
                'n_cascades': len(self.items), 'avg_len': sum(counts) / len(counts),
            }

        counts = [len(it[0]) for it in self.items]
        hist = torch.zeros(max(counts) + 1)
        for c in counts:
            hist[c] += 1
        self.count_dist = hist / hist.sum()
        self.t_max = 1.0
        self.cond_dim = self.COND_DIM[cond_type]
        self.stats.setdefault('n_max', max(counts))
        self.stats.setdefault('n_cascades', len(self.items))
        self.stats.setdefault('avg_len', sum(counts) / len(counts))
        print(f'CondCascadeDataset: {len(self.items)} cascades, cond_dim={self.cond_dim}')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        t, d, p = self.items[idx]
        return t, d, p, self.conds[idx]


def collate_fn(batch):
    ts = [item[0] for item in batch]
    ds = [item[1] for item in batch]
    ps = [item[2] for item in batch]
    return DataBatch.from_sequences(ts, ds, ps)


def collate_fn_cond(batch):
    ts = [item[0] for item in batch]
    ds = [item[1] for item in batch]
    ps = [item[2] for item in batch]
    cs = [item[3] for item in batch]
    return DataBatch.from_sequences(ts, ds, ps), torch.stack(cs)


# ---- Noise generators ----

class TreeNoise:
    def __init__(self, count_dist, t_max, d_max):
        self.count_dist = count_dist
        self.t_max = t_max
        self.d_max = d_max

    def _build_tree(self, count, rng):
        if count == 0:
            return np.empty(0), np.empty(0, dtype=np.int64), np.empty(0)
        times = np.sort(rng.random(count)) * self.t_max
        depths = np.zeros(count, dtype=np.int64)
        parent_times = np.zeros(count)
        for i in range(1, count):
            parent = rng.integers(0, i)
            depths[i] = min(int(depths[parent]) + 1, self.d_max)
            parent_times[i] = times[parent]
        return times, depths, parent_times

    def sample(self, n, generator=None):
        dev = generator.device if generator else 'cpu'
        counts = torch.multinomial(
            self.count_dist.to(dev), n, replacement=True, generator=generator
        )
        cpu_gen = torch.Generator(device='cpu')
        seed = int(torch.randint(0, 2**31, (1,), generator=cpu_gen).item())
        rng = np.random.default_rng(seed)
        t_seqs, d_seqs, p_seqs = [], [], []
        for count in counts.cpu().tolist():
            t, d, p = self._build_tree(int(count), rng)
            t_seqs.append(torch.tensor(t, dtype=torch.float32))
            d_seqs.append(torch.tensor(d, dtype=torch.long))
            p_seqs.append(torch.tensor(p, dtype=torch.float32))
        return DataBatch.from_sequences(t_seqs, d_seqs, p_seqs).to(dev)


class FlatNoise:
    def __init__(self, count_dist, t_max):
        self.count_dist = count_dist
        self.t_max = t_max

    def sample(self, n, generator=None):
        dev = generator.device if generator else 'cpu'
        counts = torch.multinomial(
            self.count_dist.to(dev), n, replacement=True, generator=generator
        )
        t_seqs = []
        for count in counts.cpu().tolist():
            count = int(count)
            if count == 0:
                t_seqs.append(torch.empty(0, dtype=torch.float32))
            else:
                t_seqs.append(torch.sort(torch.rand(count) * self.t_max)[0])
        return DataBatch.from_sequences(t_seqs).to(dev)


class HPPNoise:
    def __init__(self, lambda_hpp, t_max):
        self.lambda_hpp = lambda_hpp
        self.t_max = t_max

    def sample(self, n, generator=None):
        dev = generator.device if generator else 'cpu'
        seed = int(torch.randint(0, 2**31, (1,)).item())
        rng = np.random.default_rng(seed)
        t_seqs = []
        for _ in range(n):
            count = rng.poisson(self.lambda_hpp * self.t_max)
            if count == 0:
                t_seqs.append(torch.empty(0, dtype=torch.float32))
            else:
                times = np.sort(rng.uniform(0.0, self.t_max, count))
                t_seqs.append(torch.tensor(times, dtype=torch.float32))
        return DataBatch.from_sequences(t_seqs).to(dev)


def compute_w1(gen_lens, ref_lens):
    g = np.array(gen_lens, dtype=float)
    r = np.array(ref_lens, dtype=float)
    if len(g) == 0 or len(r) == 0:
        return 1.0
    mean_r = max(r.mean(), 1)
    return float(wasserstein_distance(g / mean_r, r / mean_r))


def _make_flat_batch(batch):
    seqs = batch.split_sequences()
    return DataBatch.from_sequences(seqs)


# ---- Public interface ----

def add_args(parser):
    parser.add_argument('--variant', choices=['uncond', 'bert', 'motive', 'flat', 'ddpm'],
                        default='uncond',
                        help='uncond=no cond, bert=BERT text cond, motive=9-dim burst score cond, '
                             'flat=W/o Tree, ddpm=W/o Flowmatching')
    parser.add_argument('--data', default='dataset/socialnet_processed.pt',
                        help='Pre-processed data path (used for uncond/flat/ddpm)')
    parser.add_argument('--cond_data', default='dataset/APS_burst.pt',
                        help='*_burst.pt for motive; raw motivation pt for bert '
                             '(e.g. APS_burst.pt / WeiboCov_burst.pt / RedditM_burst.pt)')
    parser.add_argument('--bert_model', default='pretrained/bert-base-chinese',
                        help='BERT model path or HuggingFace name (bert variant)')
    parser.add_argument('--bert_device', default='cpu',
                        help='Device for pre-computing BERT embeddings')
    parser.add_argument('--outdir', default='runs_editflow')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_ins_bins', type=int, default=64)
    parser.add_argument('--delta', type=float, default=0.05)
    parser.add_argument('--gamma', type=float, default=0.3)
    parser.add_argument('--rate_penalty', type=float, default=1.0)
    parser.add_argument('--lambda_hpp', type=float, default=50.0)
    parser.add_argument('--max_events', type=int, default=500)
    parser.add_argument('--max_steps', type=int, default=30000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--n_sample_steps', type=int, default=100)
    parser.add_argument('--n_val_samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    is_cond = args.variant in ('bert', 'motive')
    print(f'Model: OURS  Variant: {args.variant}')

    print('Loading data...')
    if is_cond:
        dataset = CondCascadeDataset(
            args.cond_data, cond_type=args.variant,
            max_events=args.max_events,
            bert_name=args.bert_model, bert_device=args.bert_device,
        )
        cond_dim = dataset.cond_dim
    else:
        dataset = CascadeSeqDataset(args.data, max_events=args.max_events)
        cond_dim = 0

    n = len(dataset)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )

    cf = collate_fn_cond if is_cond else collate_fn
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=cf, drop_last=True)

    ref_lens = [len(dataset.items[i][0]) for i in val_set.indices]
    d_max = dataset.d_max
    print(f'Data: {n} cascades, train={n_train}, val={n_val}, test={n_test}, '
          f'D_max={d_max}, cond_dim={cond_dim}')
    print(f'Ref lengths: mean={np.mean(ref_lens):.1f}, median={np.median(ref_lens):.0f}')

    if is_cond:
        n_eval = min(args.n_val_samples, len(val_set))
        val_cond_list = [dataset.conds[val_set.indices[i]] for i in range(n_eval)]
        val_conds = torch.stack(val_cond_list)
    else:
        n_eval = args.n_val_samples
        val_conds = None

    if args.variant in ('uncond', 'bert', 'motive'):
        noise = TreeNoise(dataset.count_dist, dataset.t_max, d_max)
        flow = EditFlow(n_ins_bins=args.n_ins_bins, delta=args.delta,
                        rate_penalty=args.rate_penalty, gamma=args.gamma, d_max=d_max)
    elif args.variant == 'flat':
        noise = FlatNoise(dataset.count_dist, dataset.t_max)
        flow = EditFlow(n_ins_bins=args.n_ins_bins, delta=args.delta,
                        rate_penalty=args.rate_penalty, gamma=0.0, d_max=0)
        d_max = 0
    elif args.variant == 'ddpm':
        mean_n = float(np.mean([len(item[0]) for item in dataset.items]))
        lambda_hpp = args.lambda_hpp if args.lambda_hpp > 0 else 5.0 * mean_n
        noise = HPPNoise(lambda_hpp=lambda_hpp, t_max=dataset.t_max)
        flow = DDPMFlow(n_ins_bins=args.n_ins_bins, lambda_hpp=lambda_hpp,
                        t_max=dataset.t_max, rate_penalty=args.rate_penalty, d_max=d_max)
        print(f'HPP rate: {lambda_hpp:.1f} events/unit-time')

    model = EditFlowTransformer(
        hidden_dim=args.hidden_dim, n_heads=args.n_heads, n_layers=args.n_layers,
        n_ins_bins=args.n_ins_bins, d_max=d_max, t_max=dataset.t_max, cond_dim=cond_dim,
    ).to(args.device)
    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.lr * 0.05
    )
    gen = torch.Generator(device=args.device)
    gen.manual_seed(args.seed)

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

        if is_cond:
            x1_batch, cond_batch = batch
            cond_batch = cond_batch.to(args.device)
        else:
            x1_batch = batch
            cond_batch = None

        x1_batch = x1_batch.to(args.device)

        if args.variant == 'flat':
            x1_batch = _make_flat_batch(x1_batch).to(args.device)

        model_fn = functools.partial(model, cond=cond_batch) if cond_batch is not None else model

        if args.variant == 'ddpm':
            x1_wrapped = x1_batch.wrap(prefix=0.0, suffix=dataset.t_max)
            loss = flow.compute_loss(model_fn, x1_wrapped, generator=gen)
        else:
            x0_batch = noise.sample(x1_batch.batch_size, generator=gen).to(args.device)
            idx0 = sorted(range(x0_batch.batch_size), key=lambda i: x0_batch.seq_lens[i])
            idx1 = sorted(range(x1_batch.batch_size), key=lambda i: x1_batch.seq_lens[i])
            seqs0 = x0_batch.split_sequences()
            deps0 = x0_batch.split_depth()
            pts0 = x0_batch.split_parent_time()
            paired = [None] * x0_batch.batch_size
            for rank, orig1 in enumerate(idx1):
                orig0 = idx0[rank]
                paired[orig1] = (seqs0[orig0], deps0[orig0], pts0[orig0])
            x0_batch = DataBatch.from_sequences(
                [p[0] for p in paired], [p[1] for p in paired], [p[2] for p in paired]
            )
            x1_wrapped = x1_batch.wrap(prefix=0.0, suffix=dataset.t_max)
            x0_wrapped = x0_batch.wrap(prefix=0.0, suffix=dataset.t_max)
            loss = flow.compute_loss(model_fn, x0_wrapped, x1_wrapped, generator=gen)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % 100 == 0:
            print(f'[Step {step:5d}] loss={loss.item():.4f}')

        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                n_s = n_eval if is_cond else args.n_val_samples
                if is_cond:
                    eval_conds = val_conds[:n_s].to(args.device)
                    model_eval = functools.partial(model, cond=eval_conds)
                else:
                    model_eval = model
                x0_sample = noise.sample(n_s, generator=gen)
                x0_sample = x0_sample.to(args.device).wrap(prefix=0.0, suffix=dataset.t_max)
                result, n_ins, n_del = flow.sample(model_eval, x0_sample,
                                                    n_steps=args.n_sample_steps, generator=gen)

            gen_lens = list(result.seq_lens)
            wd = compute_w1(gen_lens, ref_lens)

            gen_depths = result.depth.cpu().numpy()
            depth_dist = np.bincount(gen_depths.clip(0, d_max), minlength=d_max + 1)
            depth_str = '/'.join(str(int(d)) for d in depth_dist)
            print(f'  [Eval] gen_mean={np.mean(gen_lens):.1f} gen_median={np.median(gen_lens):.0f} '
                  f'ref_mean={np.mean(ref_lens):.1f} W1={wd:.4f} ins={n_ins} del={n_del} '
                  f'depth_dist=[{depth_str}]')

            if wd < best_wd:
                best_wd = wd
                torch.save({
                    'step': step, 'model': model.state_dict(), 'args': vars(args),
                    'best_wd': best_wd, 'stats': dataset.stats, 'd_max': d_max,
                    'variant': args.variant, 'cond_dim': cond_dim,
                }, os.path.join(args.outdir, 'best.pt'))

            model.train()

    print(f'Training complete. Best W1: {best_wd:.4f}')
    test(args)


def test(args):
    """Evaluate the best checkpoint on the held-out test split."""
    import functools
    from .metrics import eval_metrics, print_and_save, get_test_split, to_numpy_seq
    torch.manual_seed(args.seed)

    ckpt_path = os.path.join(args.outdir, 'best.pt')
    if not os.path.exists(ckpt_path):
        print(f'No checkpoint at {ckpt_path}, skipping test.')
        return None

    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    saved = ckpt.get('args', {})
    variant  = saved.get('variant', args.variant)
    is_cond  = variant in ('bert', 'motive')
    cond_dim = ckpt.get('cond_dim', saved.get('cond_dim', 0))
    d_max    = ckpt.get('d_max', saved.get('d_max', 2))

    print('Loading data for test...')
    if is_cond:
        dataset = CondCascadeDataset(
            saved.get('cond_data', args.cond_data),
            cond_type=variant,
            max_events=saved.get('max_events', args.max_events),
            bert_name=saved.get('bert_model', args.bert_model),
            bert_device=args.bert_device,
        )
    else:
        dataset = CascadeSeqDataset(
            saved.get('data', args.data),
            max_events=saved.get('max_events', args.max_events),
        )

    test_set = get_test_split(dataset, args)
    ref_seqs = [to_numpy_seq(dataset.items[i][0]) for i in test_set.indices]
    n_test   = len(ref_seqs)

    model = EditFlowTransformer(
        hidden_dim=saved.get('hidden_dim', args.hidden_dim),
        n_heads=saved.get('n_heads', args.n_heads),
        n_layers=saved.get('n_layers', args.n_layers),
        n_ins_bins=saved.get('n_ins_bins', args.n_ins_bins),
        d_max=d_max,
        t_max=dataset.t_max,
        cond_dim=cond_dim,
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    n_ins_bins = saved.get('n_ins_bins', args.n_ins_bins)
    delta      = saved.get('delta', args.delta)
    rate_pen   = saved.get('rate_penalty', args.rate_penalty)
    gamma      = saved.get('gamma', args.gamma)

    if variant in ('uncond', 'bert', 'motive'):
        noise = TreeNoise(dataset.count_dist, dataset.t_max, d_max)
        flow  = EditFlow(n_ins_bins=n_ins_bins, delta=delta,
                         rate_penalty=rate_pen, gamma=gamma, d_max=d_max)
    elif variant == 'flat':
        noise = FlatNoise(dataset.count_dist, dataset.t_max)
        flow  = EditFlow(n_ins_bins=n_ins_bins, delta=delta,
                         rate_penalty=rate_pen, gamma=0.0, d_max=0)
    else:  # ddpm
        mean_n     = float(np.mean([len(item[0]) for item in dataset.items]))
        lambda_hpp = saved.get('lambda_hpp', args.lambda_hpp) or 5.0 * mean_n
        noise = HPPNoise(lambda_hpp=lambda_hpp, t_max=dataset.t_max)
        flow  = DDPMFlow(n_ins_bins=n_ins_bins, lambda_hpp=lambda_hpp,
                         t_max=dataset.t_max, rate_penalty=rate_pen, d_max=d_max)

    gen = torch.Generator(device=args.device)
    gen.manual_seed(args.seed + 1)

    # build conditioning tensors for conditional variants
    if is_cond:
        test_conds = torch.stack(
            [dataset.conds[i] for i in test_set.indices]
        )
    else:
        test_conds = None

    n_sample_steps = saved.get('n_sample_steps', args.n_sample_steps)
    batch_size     = args.batch_size
    gen_seqs = []

    print(f'Generating for {n_test} test cascades (variant={variant})...')
    with torch.no_grad():
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            bs  = end - start
            x0  = noise.sample(bs, generator=gen).to(args.device)
            x0  = x0.wrap(prefix=0.0, suffix=dataset.t_max)
            if is_cond:
                cond_b = test_conds[start:end].to(args.device)
                model_fn = functools.partial(model, cond=cond_b)
            else:
                model_fn = model
            if variant == 'flat':
                x0_batch = DataBatch.from_sequences(x0.split_sequences())
                x0 = x0_batch.to(args.device).wrap(prefix=0.0, suffix=dataset.t_max)
            result, _, _ = flow.sample(model_fn, x0,
                                       n_steps=n_sample_steps, generator=gen)
            for seq in result.split_sequences():
                gen_seqs.append(to_numpy_seq(seq))

    m = eval_metrics(gen_seqs, ref_seqs, t_max=dataset.t_max)
    print_and_save(m, args.outdir)
    return m
