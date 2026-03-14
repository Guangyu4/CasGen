import asyncio
import json
import os
import re
import httpx
import torch
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

# --- config ---
API_KEY      = "sk-9dc1c0b0e5444eefaa6e00a8a0f5564e"
BASE_URL     = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL        = "qwen3.5-plus"
CONCURRENCY  = 50
MAX_RETRIES  = 3
SAVE_EVERY   = 50
RATE_LIMIT   = None   # max requests per minute; set to None to disable
RATE_PERIOD  = 60.0

INPUT_PT  = "/scratch/gw2556/ODE/CasGen/dataset/socialnet_cascades.pt"
OUTPUT_PT = "/scratch/gw2556/ODE/CasGen/dataset/socialnet_cascades_with_motivation.pt"
CACHE_JSON = os.path.join(os.path.dirname(__file__), "motivation_cache.json")

SCORE_KEYS = [
    "self_enhancement",
    "identity_signaling",
    "filling_conversational_space",
    "generating_social_support",
    "venting",
    "facilitating_sense_making",
    "reducing_dissonance",
    "taking_vengeance",
    "encouraging_rehearsal",
    "seeking_advice",
    "resolving_problems",
    "reinforcing_shared_views",
    "reducing_loneliness",
    "persuading_others",
]

SYSTEM_PROMPT = open(os.path.join(os.path.dirname(__file__), "Pro.md")).read()

# strip the "input:" section at the bottom (template placeholder)
SYSTEM_PROMPT = re.split(r"\ninput:", SYSTEM_PROMPT)[0].strip()

USER_TEMPLATE = '{{\n  "post_id": "{post_id}",\n  "text": "{text}"\n}}'


class RateLimiter:
    """Sliding-window rate limiter: at most `rate` calls per `period` seconds."""
    def __init__(self, rate: int, period: float = 60.0):
        self._rate = rate
        self._period = period
        self._lock = asyncio.Lock()
        self._timestamps: list[float] = []

    async def acquire(self):
        if self._rate is None:
            return
        async with self._lock:
            loop = asyncio.get_event_loop()
            now = loop.time()
            self._timestamps = [t for t in self._timestamps if now - t < self._period]
            if len(self._timestamps) >= self._rate:
                wait = self._period - (now - self._timestamps[0])
                if wait > 0:
                    await asyncio.sleep(wait)
                now = loop.time()
                self._timestamps = [t for t in self._timestamps if now - t < self._period]
            self._timestamps.append(loop.time())


def parse_scores(content: str) -> list[float] | None:
    """Extract 14 integer scores from LLM JSON response."""
    try:
        # find the JSON block
        match = re.search(r"\{[\s\S]*\}", content)
        if not match:
            return None
        data = json.loads(match.group())
        scores = []
        for key in SCORE_KEYS:
            found = False
            for category in data.values():
                if isinstance(category, dict) and key in category:
                    val = category[key]
                    score = val["score"] if isinstance(val, dict) else float(val)
                    scores.append(float(score))
                    found = True
                    break
            if not found:
                return None
        return scores
    except Exception:
        return None


async def score_one(client: AsyncOpenAI, item: dict, sem: asyncio.Semaphore, rl: RateLimiter) -> list[float]:
    post_id = item.get("id", "unknown")
    text = item.get("text", "").replace('"', '\\"').replace("\n", "\\n")
    user_msg = USER_TEMPLATE.format(post_id=post_id, text=text)

    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                await rl.acquire()
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0,
                    max_tokens=1024,
                )
                scores = parse_scores(resp.choices[0].message.content)
                if scores is not None:
                    return scores
                print(f"[WARN] parse failed for {post_id} (attempt {attempt+1}), raw: {resp.choices[0].message.content[:200]}")
            except Exception as e:
                print(f"[WARN] API error for {post_id} (attempt {attempt+1}): {e}")
            await asyncio.sleep(1.5 * (attempt + 1))

    print(f"[ERROR] giving up on {post_id}, using zeros")
    return [0.0] * 14


async def main():
    cascades = torch.load(INPUT_PT)
    print(f"Loaded {len(cascades)} cascades from {INPUT_PT}")

    # load checkpoint
    cache: dict[str, list[float]] = {}
    if os.path.exists(CACHE_JSON):
        with open(CACHE_JSON) as f:
            cache = json.load(f)
        print(f"Resuming from cache: {len(cache)} already done")

    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL,
                         http_client=httpx.AsyncClient())
    sem = asyncio.Semaphore(CONCURRENCY)
    rl = RateLimiter(RATE_LIMIT, RATE_PERIOD)

    pending = [(i, item) for i, item in enumerate(cascades) if item.get("id") not in cache]
    print(f"Pending: {len(pending)}")

    pbar = atqdm(total=len(pending), desc="Scoring")
    done_count = 0
    save_lock = asyncio.Lock()

    async def score_and_store(idx: int, item: dict):
        nonlocal done_count
        scores = await score_one(client, item, sem, rl)
        post_id = item.get("id", str(idx))
        async with save_lock:
            cache[post_id] = scores
            done_count += 1
            pbar.update(1)
            if done_count % SAVE_EVERY == 0:
                with open(CACHE_JSON, "w") as f:
                    json.dump(cache, f)

    await asyncio.gather(*[score_and_store(i, item) for i, item in pending])
    pbar.close()

    # final cache save
    with open(CACHE_JSON, "w") as f:
        json.dump(cache, f)

    # attach scores to cascades
    for item in cascades:
        post_id = item.get("id")
        scores = cache.get(post_id, [0.0] * 14)
        item["motivation_scores"] = torch.tensor(scores, dtype=torch.float32)

    torch.save(cascades, OUTPUT_PT)
    print(f"Saved {len(cascades)} cascades to {OUTPUT_PT}")


if __name__ == "__main__":
    asyncio.run(main())
