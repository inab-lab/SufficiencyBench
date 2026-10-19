#!/usr/bin/env python
"""Verbalized confidence from Claude Opus 4.7 with extended thinking ENABLED, on the full
SufficiencyBench test split, using the same VC prompt and parser as eval_new_frontier.py.

Answers the reviewer question "does native reasoning break the output-based ceiling?" for a
frontier model. The main-paper Opus 4.7 numbers were obtained with thinking off.

Usage:
  python src/experiments/eval_opus_thinking.py --key-file ~/.secrets/anthropic_key \
      [--budget 1024] [--limit 0] [--workers 6] [--dry-run]

The key is read from the file and never printed or logged. Raw responses are cached in
results/api_models/claude-opus-4-7-thinking_vc.jsonl (resumable). Summary goes to
results/api_models/claude-opus-4-7-thinking_summary.json.
Note: the Anthropic API requires temperature=1 when extended thinking is enabled.
"""
import argparse, json, os, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "experiments"))
from eval_new_frontier import VC_PROMPT, extract_vc_score, load_sample  # noqa: E402

MODEL = "claude-opus-4-7"
OUT_DIR = ROOT / "results" / "api_models"
CACHE = OUT_DIR / "claude-opus-4-7-thinking_vc.jsonl"
SUMMARY = OUT_DIR / "claude-opus-4-7-thinking_summary.json"
PK_FILE = OUT_DIR / "claude-opus-4-7_results.json"


def load_pk_by_question():
    """Per-question PK flag for Opus 4.7 from the earlier run (tolerant of NaN)."""
    try:
        raw = re.sub(r"\bNaN\b", "null", open(PK_FILE).read())
        d = json.loads(raw)
    except Exception as e:  # pragma: no cover
        print(f"[warn] could not read PK flags: {e}")
        return {}
    pk = {}
    def walk(o):
        if isinstance(o, dict):
            cid = o.get("condition_id") or o.get("id")
            flag = o.get("pk", o.get("model_knows_answer", o.get("has_pk")))
            if cid and flag is not None:
                pk[str(cid).rsplit("_", 1)[0]] = bool(flag) if not isinstance(flag, str) else flag.lower() == "true"
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(d)
    return pk


def boot_ci(y, s, groups, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed); uq = np.unique(groups)
    idx = {q: np.flatnonzero(groups == q) for q in uq}; out = []
    for _ in range(n_boot):
        pick = rng.choice(uq, size=len(uq), replace=True)
        sel = np.concatenate([idx[q] for q in pick])
        if len(set(y[sel])) > 1: out.append(roc_auc_score(y[sel], s[sel]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--budget", type=int, default=1024, help="thinking budget_tokens (min 1024)")
    ap.add_argument("--limit", type=int, default=0, help="0 = full test split")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true", help="build prompts, estimate cost, no API calls")
    a = ap.parse_args()

    items = load_sample(a.limit, 42)
    print(f"{len(items)} items, model {MODEL}, thinking budget {a.budget}")
    prompts = {it["condition_id"]: VC_PROMPT.format(context=it["context"], question=it["question"]) for it in items}
    in_tok = sum(len(p) / 4 for p in prompts.values())
    print(f"~{in_tok/1e6:.2f}M input tokens; worst-case output {(a.budget+50)*len(items)/1e6:.2f}M tokens "
          f"-> rough cost at $5/$25 per MTok: ${in_tok/1e6*5 + (a.budget+50)*len(items)/1e6*25:.0f} (upper bound)")
    if a.dry_run:
        return

    key = open(os.path.expanduser(a.key_file)).read().strip()
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    del key

    done = {}
    if CACHE.exists():
        for line in open(CACHE):
            if line.strip():
                r = json.loads(line); done[r["condition_id"]] = r
    todo = [it for it in items if it["condition_id"] not in done]
    print(f"cached {len(done)}, to run {len(todo)}")
    lock = threading.Lock()

    def call(it):
        cid = it["condition_id"]
        for attempt in range(6):
            try:
                resp = client.messages.create(
                    model=MODEL, max_tokens=a.budget + 200, temperature=1,
                    thinking={"type": "enabled", "budget_tokens": a.budget},
                    messages=[{"role": "user", "content": prompts[cid]}])
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                think = sum(len(getattr(b, "thinking", "") or "") for b in resp.content if getattr(b, "type", "") == "thinking")
                rec = {"condition_id": cid, "label": it["label"], "question_type": it["question_type"],
                       "quadrant": it["quadrant"], "text": text, "score": extract_vc_score(text),
                       "thinking_chars": think, "usage": {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens},
                       "stop": resp.stop_reason}
                with lock:
                    with open(CACHE, "a") as f: f.write(json.dumps(rec) + "\n")
                return rec
            except Exception as e:
                wait = 2 ** attempt
                print(f"[{cid}] attempt {attempt+1} failed: {type(e).__name__}: {str(e)[:80]} -> sleep {wait}s")
                time.sleep(wait)
        return None

    t0 = time.time(); n_ok = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(call, it) for it in todo]
        for i, f in enumerate(as_completed(futs), 1):
            if f.result(): n_ok += 1
            if i % 50 == 0: print(f"  {i}/{len(todo)} done, {time.time()-t0:.0f}s")
    print(f"finished: {n_ok}/{len(todo)} new calls ok")

    recs = {}
    for line in open(CACHE):
        if line.strip():
            r = json.loads(line); recs[r["condition_id"]] = r
    recs = [recs[it["condition_id"]] for it in items if it["condition_id"] in recs]
    y = np.array([r["label"] for r in recs]); s = np.array([r["score"] for r in recs])
    g = np.array([r["condition_id"].rsplit("_", 1)[0] for r in recs]); qt = np.array([r["question_type"] for r in recs])
    lo, hi = boot_ci(y, s, g)
    out = {"model": MODEL, "thinking_budget": a.budget, "temperature": 1, "n": int(len(y)),
           "auroc": float(roc_auc_score(y, s)), "ci95": [lo, hi],
           "per_type": {t: float(roc_auc_score(y[qt == t], s[qt == t])) for t in sorted(set(qt)) if len(set(y[qt == t])) > 1},
           "mean_thinking_chars": float(np.mean([r["thinking_chars"] for r in recs])),
           "truncated_rate": float(np.mean([r["stop"] == "max_tokens" for r in recs])),
           "unparsed_rate": float(np.mean([not re.search(r"\d", r["text"] or "") for r in recs])),
           "tokens": {"in": int(sum(r["usage"]["in"] for r in recs)), "out": int(sum(r["usage"]["out"] for r in recs))}}
    pk = load_pk_by_question()
    if pk:
        m0 = np.array([not pk.get(q, True) for q in g]); m1 = ~m0
        if m0.sum() and len(set(y[m0])) > 1: out["auroc_pk0"] = float(roc_auc_score(y[m0], s[m0])); out["n_pk0"] = int(m0.sum())
        if m1.sum() and len(set(y[m1])) > 1: out["auroc_pk1"] = float(roc_auc_score(y[m1], s[m1])); out["n_pk1"] = int(m1.sum())
    else:
        out["note"] = "per-question PK flags for Opus 4.7 not found; PK=0 AUROC not computed"
    json.dump(out, open(SUMMARY, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
