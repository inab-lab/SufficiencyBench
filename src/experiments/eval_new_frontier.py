"""
eval_new_frontier.py

Evaluates GPT-5.5 and Claude Opus 4.7 on the FULL SufficiencyBench test set (1,112 examples).
Uses the Batch API by default (50% cost reduction).

Methods evaluated per model:
  - LLM self-judge (YES/NO)
  - Verbalized confidence (0-100 score)
  - GPT-5.5 only: judge + VC with reasoning_effort=high (tests whether reasoning breaks the ceiling)

Usage:
  export OPENAI_API_KEY=sk-...
  export ANTHROPIC_API_KEY=sk-ant-...

  # Run one model first to check results before committing to full plan:
  python src/experiments/eval_new_frontier.py --model claude-opus-4-7

  # Full Anthropic plan (~$11):
  python src/experiments/eval_new_frontier.py --model claude-opus-4-7
  python src/experiments/eval_new_frontier.py --model claude-sonnet-4-6
  python src/experiments/eval_new_frontier.py --model claude-opus-4-7-cot

  # Full plan including OpenAI (when credits available):
  python src/experiments/eval_new_frontier.py --all

  # Live mode (immediate results, 2x cost):
  python src/experiments/eval_new_frontier.py --all --live

Cost estimate (batch mode, 1,112 examples, ~973K input + ~22K output tokens):
  GPT-5.5 judge+VC:            ~$2.50  ($2.50/M in, $15/M out, batch)
  GPT-5.5 reasoning=high:      ~$4-8   (reasoning tokens billed extra)
  Claude Opus 4.7 judge+VC:    ~$2.71  ($2.50/M in, $12.50/M out, batch)
  Total all runs:              ~$10-15
"""

import json
import os
import sys
import time
import re
import argparse
from pathlib import Path
from sklearn.metrics import roc_auc_score
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import BENCH_DIR, RESULTS_DIR

# Load .env from project root if present (safe fallback; env vars take precedence)
_env_file = Path(__file__).resolve().parents[2] / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

OUT_DIR = RESULTS_DIR / "api_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_EXAMPLES = 0          # 0 = use full test set (1,112 examples)
RANDOM_SEED = 42
BATCH_POLL_INTERVAL = 30  # seconds

JUDGE_PROMPT = (
    "Does the context contain sufficient information to answer the question? "
    "Answer YES or NO only.\n\n"
    "Context: {context}\n\n"
    "Question: {question}"
)

VC_PROMPT = (
    "Rate from 0 to 100 how sufficient the context is for answering the question. "
    "0 = no relevant information; 100 = fully contains the answer. "
    "Reply with a single integer.\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n\nScore:"
)

COT_JUDGE_PROMPT = (
    "Does the context contain sufficient information to answer the question?\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "Think step by step: identify what information the question requires, "
    "then check whether the context contains it. "
    "End your response with 'Answer: YES' or 'Answer: NO'."
)


# ─── Sample loading ────────────────────────────────────────────────────────────

def load_sample(n: int = N_EXAMPLES, seed: int = RANDOM_SEED) -> list:
    with open(BENCH_DIR / "test.json") as f:
        data = json.load(f)

    flat = []
    for ex in data:
        for cond in ex["conditions"]:
            flat.append({
                "question": ex["question"],
                "gold_answer": ex["gold_answer"],
                "question_type": ex["question_type"],
                "context": cond["context"],
                "condition_id": cond["condition_id"],
                "quadrant": cond["quadrant"],
                "sufficient": cond["sufficient"],
                "label": int(cond["sufficient"]),
            })

    if n and n > 0:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(flat), size=min(n, len(flat)), replace=False)
        flat = [flat[i] for i in idx]
    return flat


# ─── Response parsing ───────────────────────────────────────────────────────────

def extract_yes_no(text: str) -> str:
    if not text:
        return "UNCLEAR"
    t = text.strip().upper()
    # Prefer an explicit "Answer: YES/NO" tail (CoT prompt)
    m = re.search(r"ANSWER:\s*(YES|NO)", t)
    if m:
        return m.group(1)
    if t[:20].strip().startswith("YES"):
        return "YES"
    if t[:20].strip().startswith("NO"):
        return "NO"
    if re.search(r"\bYES\b", t):
        return "YES"
    if re.search(r"\bNO\b", t):
        return "NO"
    return "UNCLEAR"


def yes_no_to_score(verdict: str) -> float:
    return 1.0 if verdict == "YES" else (0.5 if verdict == "UNCLEAR" else 0.0)


def extract_vc_score(text: str) -> float:
    """Parse a 0-100 verbalized-confidence integer into a 0-1 score."""
    if not text:
        return 0.5
    nums = re.findall(r"\d+", text)
    if not nums:
        return 0.5
    return min(max(float(nums[0]) / 100.0, 0.0), 1.0)


# ─── Prompt building ────────────────────────────────────────────────────────────

def build_prompts(items: list, method: str) -> list:
    """Return the user-message string for each item under a given method."""
    if method == "judge":
        template = JUDGE_PROMPT
    elif method == "vc":
        template = VC_PROMPT
    elif method == "cot":
        template = COT_JUDGE_PROMPT
    else:
        raise ValueError(f"Unknown method: {method}")
    return [template.format(context=it["context"], question=it["question"]) for it in items]


def score_method(method: str, texts: list, items: list) -> dict:
    """Convert raw model outputs into per-example scores + AUROC."""
    labels = [it["label"] for it in items]
    if method == "vc":
        scores = [extract_vc_score(t) for t in texts]
    else:
        scores = [yes_no_to_score(extract_yes_no(t)) for t in texts]
    auroc = (float(roc_auc_score(labels, scores))
             if len(set(labels)) > 1 else 0.5)
    return {"method": method, "auroc": auroc, "scores": scores, "labels": labels}


# ─── OpenAI Batch API ───────────────────────────────────────────────────────────

def run_openai_batch(client, model: str, items: list, method: str,
                     max_tokens: int, reasoning_effort: str = None) -> list:
    """Submit a batch of /v1/chat/completions requests, poll, return ordered texts."""
    import tempfile

    prompts = build_prompts(items, method)
    lines = []
    for i, prompt in enumerate(prompts):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        if reasoning_effort:
            # GPT-5.5 reasoning models bill reasoning tokens separately.
            body["reasoning_effort"] = reasoning_effort
        lines.append({
            "custom_id": f"{method}-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        })

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
        input_path = f.name

    batch_input_file = client.files.create(file=open(input_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"    OpenAI batch {batch.id} submitted ({len(lines)} requests)")

    while True:
        batch = client.batches.retrieve(batch.id)
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(BATCH_POLL_INTERVAL)
    print(f"    OpenAI batch {batch.status}")

    if batch.status != "completed":
        return ["ERROR"] * len(items)

    out_text = client.files.content(batch.output_file_id).text
    by_id = {}
    for row in out_text.splitlines():
        if not row.strip():
            continue
        rec = json.loads(row)
        cid = rec["custom_id"]
        try:
            by_id[cid] = rec["response"]["body"]["choices"][0]["message"]["content"]
        except Exception:
            by_id[cid] = "ERROR"
    return [by_id.get(f"{method}-{i}", "ERROR") for i in range(len(items))]


# ─── Anthropic Message Batches API ──────────────────────────────────────────────

def run_anthropic_batch(client, model: str, items: list, method: str,
                        max_tokens: int) -> list:
    """Submit an Anthropic message batch, poll to completion, return ordered texts."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    prompts = build_prompts(items, method)
    requests = [
        Request(
            custom_id=f"{method}-{i}",
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            ),
        )
        for i, prompt in enumerate(prompts)
    ]

    batch = client.messages.batches.create(requests=requests)
    print(f"    Anthropic batch {batch.id} submitted ({len(requests)} requests)")

    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        time.sleep(BATCH_POLL_INTERVAL)
    print("    Anthropic batch ended")

    by_id = {}
    for result in client.messages.batches.results(batch.id):
        cid = result.custom_id
        if result.result.type == "succeeded":
            try:
                by_id[cid] = result.result.message.content[0].text
            except Exception:
                by_id[cid] = "ERROR"
        else:
            by_id[cid] = "ERROR"
    return [by_id.get(f"{method}-{i}", "ERROR") for i in range(len(items))]


# ─── Live (non-batch) fallback ──────────────────────────────────────────────────

def run_live(provider, client, model: str, items: list, method: str,
             max_tokens: int) -> list:
    """Immediate per-request evaluation (2x cost) — used with --live."""
    prompts = build_prompts(items, method)
    texts = []
    for i, prompt in enumerate(prompts):
        try:
            if provider == "openai":
                r = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens, temperature=0.0,
                )
                texts.append(r.choices[0].message.content)
            else:
                r = client.messages.create(
                    model=model, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                texts.append(r.content[0].text)
        except Exception:
            texts.append("ERROR")
        if (i + 1) % 50 == 0:
            print(f"    live [{i+1}/{len(prompts)}]")
        time.sleep(0.05)
    return texts


# ─── Per-model orchestration ────────────────────────────────────────────────────

def resolve_model(name: str):
    """Map a CLI model name to (provider, api_model, methods, reasoning_effort)."""
    cot = name.endswith("-cot")
    base = name[:-4] if cot else name

    if base.startswith("gpt"):
        provider = "openai"
    elif base.startswith("claude"):
        provider = "anthropic"
    else:
        raise ValueError(f"Unknown model: {name}")

    methods = ["cot"] if cot else ["judge", "vc"]
    # GPT-5.5 supports a high-reasoning variant that tests whether reasoning
    # breaks the ceiling.
    reasoning = "high" if (cot and provider == "openai") else None
    return provider, base, methods, reasoning


def run_model(name: str, live: bool = False):
    provider, api_model, methods, reasoning = resolve_model(name)
    items = load_sample()
    print(f"\n=== eval_new_frontier: {name} ({provider}, {len(items)} examples) ===")

    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            print("pip install openai")
            return
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    else:
        try:
            import anthropic
        except ImportError:
            print("pip install anthropic")
            return
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    results = {}
    for method in methods:
        max_tokens = 512 if method == "cot" else (5 if method == "vc" else 8)
        if live:
            texts = run_live(provider, client, api_model, items, method, max_tokens)
        elif provider == "openai":
            texts = run_openai_batch(client, api_model, items, method,
                                     max_tokens, reasoning_effort=reasoning)
        else:
            texts = run_anthropic_batch(client, api_model, items, method, max_tokens)

        r = score_method(method, texts, items)
        results[method] = r
        print(f"  {method:<6s} AUROC = {r['auroc']:.4f}")

    out = OUT_DIR / f"{name.replace('/', '_')}_frontier.json"
    with open(out, "w") as f:
        json.dump({
            "model": name,
            "provider": provider,
            "n_examples": len(items),
            "results": results,
        }, f, indent=2)
    print(f"Saved to {out}")
    return results


ALL_MODELS = ["claude-opus-4-7", "claude-sonnet-4-6", "claude-opus-4-7-cot", "gpt-5.5"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-opus-4-7",
                        help="e.g. gpt-5.5, claude-opus-4-7, claude-opus-4-7-cot")
    parser.add_argument("--all", action="store_true", help="Run the full model plan")
    parser.add_argument("--live", action="store_true",
                        help="Live mode (immediate results, 2x cost)")
    args = parser.parse_args()

    models = ALL_MODELS if args.all else [args.model]
    for m in models:
        try:
            run_model(m, live=args.live)
        except Exception as e:
            print(f"Error on {m}: {e}")
            import traceback
            traceback.print_exc()