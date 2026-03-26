"""LangTPP trainer for CasGen: Qwen2.5-0.5B fine-tuned as a cascade sequence generator.

Each cascade is formatted as a causal LM sequence:

  {root_text_tokens}
  <|start_of_event|><|time_prefix|>B1 B2 B3 B4<|end_of_event|>
  <|start_of_event|><|time_prefix|>...
  <eos>

B1-B4 are the big-endian float32 byte tokens of the event time (normalised to [0,1]).
The model is trained with standard next-token cross-entropy loss, with the text prefix
masked from the loss so learning focuses entirely on generating event tokens.

Architecture: Qwen2.5-0.5B base weights + LangTPP extended vocabulary
(256 byte tokens + 8 event special tokens).  The embedding table is resized
to include these tokens before fine-tuning.
"""
import os
import sys
import math
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ── Special token IDs (fixed by the LangTPP tokenizer vocabulary) ─────────────
SOE_ID   = 151665   # <|start_of_event|>
EOE_ID   = 151666   # <|end_of_event|>
TIME_ID  = 151667   # <|time_prefix|>
BYTE_BASE = 151673  # <|byte_0|>; <|byte_k|> = BYTE_BASE + k

TOKENS_PER_EVENT = 7  # SOE + TIME_PFX + 4 bytes + EOE

# Path to the Language-TPP tokenizer (carries the full special-token vocabulary)
_HERE       = os.path.dirname(os.path.abspath(__file__))
LANGTPP_DIR = os.path.realpath(os.path.join(
    _HERE, '..', '..', 'CodeRef', 'Language-TPP', 'Language_TPP_0___5B'
))


# ── Byte-encoding helpers ──────────────────────────────────────────────────────

def _time_to_byte_ids(t: float):
    """float32 event time → list of 4 token IDs (big-endian byte order)."""
    b = np.array([t], dtype=np.float32).tobytes()   # little-endian memory
    # Encode as [b[3], b[2], b[1], b[0]] (MSB first)
    return [BYTE_BASE + b[3], BYTE_BASE + b[2], BYTE_BASE + b[1], BYTE_BASE + b[0]]


def _byte_ids_to_time(ids) -> float:
    """4 token IDs (big-endian) → float32 event time."""
    b = [int(i) - BYTE_BASE for i in ids]   # [b[3], b[2], b[1], b[0]]
    le = bytes([b[3], b[2], b[1], b[0]])    # restore little-endian
    return float(np.frombuffer(le, dtype=np.float32)[0])


def _decode_gen_ids(token_ids) -> np.ndarray:
    """Parse a generated token-ID list into a sorted array of event times."""
    times = []
    i = 0
    n = len(token_ids)
    while i < n:
        if token_ids[i] == SOE_ID and i + 6 < n:
            if (token_ids[i + 1] == TIME_ID
                    and all(BYTE_BASE <= token_ids[i + j] < BYTE_BASE + 256
                            for j in range(2, 6))):
                t = _byte_ids_to_time(token_ids[i + 2: i + 6])
                if 0.0 <= t <= 1.0:
                    times.append(t)
                i += 7
                continue
        i += 1
    return np.array(times, dtype=np.float64)


# ── Dataset ────────────────────────────────────────────────────────────────────

class CascadeTokenDataset(Dataset):
    def __init__(self, cascades, tokenizer, max_events=500,
                 max_text_tokens=256, max_seq_len=4096):
        self.samples = []
        for cas in cascades:
            times = cas['times']
            if hasattr(times, 'numpy'):
                times = times.numpy()
            times = np.asarray(times, dtype=np.float32)[:max_events]
            text  = cas.get('text', '')

            # Text prefix (no special tokens, truncated)
            pfx = tokenizer.encode(text, add_special_tokens=False)[:max_text_tokens]

            # Event tokens
            evt = []
            for t in times:
                evt += [SOE_ID, TIME_ID] + _time_to_byte_ids(float(t)) + [EOE_ID]

            # Cap to max_seq_len (keep prefix, truncate events)
            budget = max_seq_len - len(pfx) - 1  # -1 for eos
            if len(evt) > budget:
                budget = (budget // TOKENS_PER_EVENT) * TOKENS_PER_EVENT
                evt = evt[:budget]

            ids = pfx + evt + [tokenizer.eos_token_id]
            self.samples.append((ids, len(pfx)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, pfx_len = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), pfx_len


def _collate(batch):
    ids_list, pfx_lens = zip(*batch)
    B       = len(ids_list)
    max_len = max(len(x) for x in ids_list)

    input_ids = torch.zeros(B, max_len, dtype=torch.long)
    attn_mask = torch.zeros(B, max_len, dtype=torch.long)
    labels    = torch.full((B, max_len), -100, dtype=torch.long)

    for i, (ids, pfx) in enumerate(zip(ids_list, pfx_lens)):
        L = len(ids)
        input_ids[i, :L] = ids
        attn_mask[i, :L] = 1
        labels[i, pfx:L] = ids[pfx:L]   # only learn event tokens

    return input_ids, attn_mask, labels


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_tokenizer(langtpp_dir=None):
    from transformers import AutoTokenizer
    d = langtpp_dir or LANGTPP_DIR
    tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
    tok.padding_side = 'left'
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def _load_model(qwen_path, vocab_size, gradient_checkpointing=False):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        qwen_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if model.get_input_embeddings().weight.shape[0] != vocab_size:
        model.resize_token_embeddings(vocab_size)
        model.config.vocab_size = vocab_size
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


# ── Generation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def _generate(model, tokenizer, text_ids_list, max_events, batch_size, device):
    model.eval()
    gen_seqs = []
    max_new  = max_events * TOKENS_PER_EVENT + 1

    for start in range(0, len(text_ids_list), batch_size):
        chunk = text_ids_list[start: start + batch_size]
        B     = len(chunk)
        plen  = max(len(t) for t in chunk)

        # Left-pad prompts
        inp  = torch.full((B, plen), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros(B, plen, dtype=torch.long)
        for i, t in enumerate(chunk):
            inp[i,  plen - len(t):] = torch.tensor(t, dtype=torch.long)
            mask[i, plen - len(t):] = 1

        inp  = inp.to(device)
        mask = mask.to(device)

        out = model.generate(
            input_ids=inp,
            attention_mask=mask,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        for i in range(B):
            gen_ids = out[i, plen:].cpu().tolist()
            gen_seqs.append(_decode_gen_ids(gen_ids))

    return gen_seqs


# ── Data split helper ──────────────────────────────────────────────────────────

def _split_cascades(cascades, seed):
    n       = len(cascades)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    n_test  = n - n_train - n_val
    g       = torch.Generator().manual_seed(seed)
    idx     = random_split(range(n), [n_train, n_val, n_test], generator=g)
    return (
        [cascades[i] for i in idx[0]],
        [cascades[i] for i in idx[1]],
        [cascades[i] for i in idx[2]],
    )


# ── Trainer interface ──────────────────────────────────────────────────────────

def add_args(parser):
    parser.add_argument('--data',     required=True,
                        help='*_burst.pt data file')
    parser.add_argument('--qwen_model', default='pretrained/Qwen2.5-0.5B',
                        help='Path to Qwen2.5-0.5B base weights')
    parser.add_argument('--langtpp_dir', default='',
                        help='Override path to Language_TPP_0___5B tokenizer dir '
                             '(defaults to ../../CodeRef/Language-TPP/Language_TPP_0___5B)')
    parser.add_argument('--outdir',   default='runs_langtpp/run')
    parser.add_argument('--max_events',      type=int, default=500)
    parser.add_argument('--max_text_tokens', type=int, default=256,
                        help='Max root-text tokens prepended to each sequence')
    parser.add_argument('--max_seq_len',     type=int, default=4096)
    parser.add_argument('--max_steps',       type=int, default=500)
    parser.add_argument('--batch_size',      type=int, default=1,
                        help='Per-GPU micro-batch size')
    parser.add_argument('--grad_accum',      type=int, default=32,
                        help='Gradient accumulation steps '
                             '(effective batch = batch_size × grad_accum)')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True,
                        help='Enable gradient checkpointing to save GPU memory')
    parser.add_argument('--lr',              type=float, default=2e-5)
    parser.add_argument('--warmup_steps',    type=int, default=50)
    parser.add_argument('--eval_every',      type=int, default=100)
    parser.add_argument('--n_val_samples',   type=int, default=200)
    parser.add_argument('--seed',            type=int, default=42)


def train(args):
    from transformers import get_cosine_schedule_with_warmup
    from .metrics import eval_metrics, print_and_save, to_numpy_seq

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    langtpp_dir = args.langtpp_dir or LANGTPP_DIR

    print(f'Loading tokenizer from {langtpp_dir}')
    tok = _load_tokenizer(langtpp_dir)

    print(f'Loading model from {args.qwen_model}')
    gc = getattr(args, 'gradient_checkpointing', True)
    model = _load_model(args.qwen_model, len(tok), gradient_checkpointing=gc).to(device)

    print(f'Loading data from {args.data}')
    raw      = torch.load(args.data, map_location='cpu', weights_only=False)
    cascades = raw['cascades']
    train_cas, val_cas, _ = _split_cascades(cascades, args.seed)

    ds_kw = dict(
        tokenizer=tok,
        max_events=args.max_events,
        max_text_tokens=args.max_text_tokens,
        max_seq_len=args.max_seq_len,
    )
    train_ds = CascadeTokenDataset(train_cas, **ds_kw)
    loader   = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=_collate, num_workers=2, pin_memory=True,
                          drop_last=True)

    # Validation text prefixes & reference sequences
    n_val      = min(args.n_val_samples, len(val_cas))
    val_txt    = [tok.encode(c.get('text', ''), add_special_tokens=False)
                      [:args.max_text_tokens]
                  for c in val_cas[:n_val]]
    val_ref    = [to_numpy_seq(val_cas[i]['times'][:args.max_events])
                  for i in range(n_val)]

    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler  = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    log_path = os.path.join(args.outdir, 'train.log')
    log_f    = open(log_path, 'w', buffering=1)

    def log(msg):
        print(msg)
        log_f.write(msg + '\n')

    log(f'train={len(train_ds)}  val={n_val}  '
        f'eff_batch={args.batch_size * args.grad_accum}  '
        f'steps={args.max_steps}')

    best_wd   = float('inf')
    step      = 0
    accum     = 0
    run_loss  = 0.0

    model.train()
    optimizer.zero_grad()

    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break

            input_ids, attn_mask, labels = [x.to(device) for x in batch]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = model(input_ids=input_ids,
                             attention_mask=attn_mask,
                             labels=labels).loss / args.grad_accum
            loss.backward()
            run_loss += loss.item()
            accum    += 1

            if accum == args.grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum    = 0
                step    += 1

                if step % 50 == 0:
                    log(f'step={step:5d}  loss={run_loss:.4f}'
                        f'  lr={scheduler.get_last_lr()[0]:.2e}')
                run_loss = 0.0

                if step % args.eval_every == 0:
                    log(f'\n--- eval @ step {step} ---')
                    gen = _generate(model, tok, val_txt,
                                    max_events=args.max_events,
                                    batch_size=args.batch_size,
                                    device=device)
                    m   = eval_metrics(gen, val_ref)
                    wd  = m['w1_l'] + m['w1_t']
                    log(f'  MMD={m["mmd"]:.4f}  W1l={m["w1_l"]:.4f}'
                        f'  W1t={m["w1_t"]:.4f}  wd={wd:.4f}')

                    if wd < best_wd:
                        best_wd = wd
                        best_dir = os.path.join(args.outdir, 'best_model')
                        model.save_pretrained(best_dir)
                        tok.save_pretrained(best_dir)
                        torch.save({'step': step, 'best_wd': best_wd,
                                    'args': vars(args)},
                                   os.path.join(args.outdir, 'best.pt'))
                        log(f'  → saved best  wd={best_wd:.4f}')

                    model.train()

    log_f.close()
    test(args)


def test(args):
    from .metrics import eval_metrics, print_and_save, to_numpy_seq

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    langtpp_dir = getattr(args, 'langtpp_dir', '') or LANGTPP_DIR
    tok = _load_tokenizer(langtpp_dir)

    best_dir = os.path.join(args.outdir, 'best_model')
    src      = best_dir if os.path.isdir(best_dir) else args.qwen_model
    print(f'Loading model from {src}')
    model = _load_model(src, len(tok)).to(device)

    raw      = torch.load(args.data, map_location='cpu', weights_only=False)
    cascades = raw['cascades']
    _, _, test_cas = _split_cascades(cascades, args.seed)

    max_text = getattr(args, 'max_text_tokens', 256)
    max_evt  = getattr(args, 'max_events', 500)

    test_txt = [tok.encode(c.get('text', ''), add_special_tokens=False)[:max_text]
                for c in test_cas]
    ref_seqs = [to_numpy_seq(c['times'][:max_evt]) for c in test_cas]

    print(f'Generating for {len(test_cas)} test cascades ...')
    gen_seqs = _generate(model, tok, test_txt,
                         max_events=max_evt,
                         batch_size=getattr(args, 'batch_size', 4),
                         device=device)

    m = eval_metrics(gen_seqs, ref_seqs)
    print_and_save(m, args.outdir)
