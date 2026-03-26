"""Single-GPU vLLM inference for LAMP cascade feature prediction.

Called by trainers/lamp.py via subprocess using the vllm_env Python:
  <vllm_env>/bin/python scripts/lamp_llm_infer.py <input.jsonl> <output.jsonl>

Input  (one JSON per line): {"id": "...", "system": "...", "prompt": "..."}
Output (one JSON per line): {"id": "...", "response": "..."}

Supports resume: already-completed ids in the output file are skipped.
Uses tensor_parallel_size=1 (single GPU).
"""
import json
import sys
import time
from pathlib import Path

MODEL_PATH      = '/gpfsnyu/spack/share/models/Qwen3-8B'
NUM_GPUS        = 1
MAX_TOKENS      = 256
GPU_MEM_UTIL    = 0.90
ENABLE_THINKING = False
MAX_MODEL_LEN   = 4096
WRITE_EVERY     = 5000


def load_done_ids(output_file: str) -> set:
    p = Path(output_file)
    if not p.exists():
        return set()
    done = set()
    for line in p.read_text(encoding='utf-8').splitlines():
        if line.strip():
            try:
                done.add(str(json.loads(line)['id']))
            except Exception:
                pass
    return done


def main(input_file: str, output_file: str) -> None:
    from vllm import LLM, SamplingParams

    all_items = [
        json.loads(line)
        for line in Path(input_file).read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]

    done_ids = load_done_ids(output_file)
    items    = [x for x in all_items if str(x['id']) not in done_ids]
    if done_ids:
        print(f'Resuming: {len(done_ids)} done, {len(items)} remaining', flush=True)
    if not items:
        print('All items already processed.', flush=True)
        return

    print(f'Loading {MODEL_PATH}  (tensor_parallel_size={NUM_GPUS})...', flush=True)
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=NUM_GPUS,
        dtype='bfloat16',
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=MAX_MODEL_LEN,
        disable_log_stats=True,
    )

    tokenizer = llm.get_tokenizer()
    params    = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    MAX_PROMPT_TOKENS = MAX_MODEL_LEN - MAX_TOKENS - 32

    def build_prompt(item):
        messages = []
        if item.get('system'):
            messages.append({'role': 'system', 'content': item['system']})
        messages.append({'role': 'user', 'content': item['prompt']})
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > MAX_PROMPT_TOKENS:
            text = tokenizer.decode(ids[:MAX_PROMPT_TOKENS], skip_special_tokens=False)
        return text

    formatted = [build_prompt(item) for item in items]

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    t0, total_toks = time.time(), 0

    for chunk_start in range(0, len(items), WRITE_EVERY):
        chunk_items = items[chunk_start: chunk_start + WRITE_EVERY]
        chunk_fmt   = formatted[chunk_start: chunk_start + WRITE_EVERY]
        outputs     = llm.generate(chunk_fmt, params, use_tqdm=True)

        with open(output_file, 'a', encoding='utf-8') as f:
            for item, out in zip(chunk_items, outputs):
                f.write(json.dumps(
                    {'id': item['id'], 'response': out.outputs[0].text.strip()},
                    ensure_ascii=False,
                ) + '\n')

        total_toks += sum(len(o.outputs[0].token_ids) for o in outputs)
        done    = chunk_start + len(chunk_items)
        elapsed = time.time() - t0
        print(f'[{done}/{len(items)}]  {total_toks/elapsed:.0f} tok/s', flush=True)

    elapsed = time.time() - t0
    print(f'\nDone: {len(items)} items in {elapsed:.1f}s', flush=True)
    print(f'Saved → {output_file}', flush=True)


if __name__ == '__main__':
    inp = sys.argv[1] if len(sys.argv) > 1 else 'llm_inp.jsonl'
    out = sys.argv[2] if len(sys.argv) > 2 else 'llm_out.jsonl'
    main(inp, out)
