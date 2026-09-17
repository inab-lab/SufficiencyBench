#!/usr/bin/env python
"""build_pk_cache_safe.py — Non-destructive per-model parametric-knowledge (PK) cache.

The canonical SufficiencyBench split (train/val/test.json) was built once using
llama as the PK reference. For per-model quadrant stratification we need the SAME
closed-book PK test run with mistral and qwen — WITHOUT rebuilding / overwriting
the dataset (which `build_sufficiency_bench.py --model <m>` would do).

This script:
  * reads the questions that actually made it into the built dataset,
  * runs the EXACT `test_parametric_knowledge` routine from the builder (so the
    'knows' criterion and cache format are byte-identical to the llama cache),
  * writes only data/experiments/sufficiency_bench/pk_cache_<model>.json.

Usage:
  python scripts/build_pk_cache_safe.py --model mistral --device cuda:0
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from configs.paths import BENCH_DIR, MODEL_CONFIGS  # noqa: E402
from data.build_sufficiency_bench import test_parametric_knowledge  # noqa: E402


def load_dataset_questions():
    """Collect the unique (id, question, answer) tuples in the built dataset.

    `test_parametric_knowledge` expects each question dict to expose 'id',
    'question' and 'answer'; the dataset stores the gold answer as 'gold_answer'.
    """
    seen, questions = set(), []
    for split in ("train", "val", "test"):
        path = BENCH_DIR / f"{split}.json"
        with open(path) as f:
            data = json.load(f)
        for ex in data:
            qid = ex["id"]
            if qid in seen:
                continue
            seen.add(qid)
            questions.append({
                "id": qid,
                "question": ex["question"],
                "answer": ex.get("gold_answer", ""),
            })
    return questions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only process this many questions (smoke test).")
    args = ap.parse_args()

    questions = load_dataset_questions()
    if args.limit:
        questions = questions[:args.limit]
    print(f"Loaded {len(questions)} unique dataset questions for PK test.")

    cache_path = BENCH_DIR / f"pk_cache_{args.model}.json"
    model_path = MODEL_CONFIGS[args.model]["path"]
    print(f"Loading {args.model} from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=args.device,
    )
    model.eval()

    results = test_parametric_knowledge(
        model, tokenizer, questions, device=args.device, cache_path=cache_path,
    )

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)

    knows = sum(1 for v in results.values() if v["knows"])
    print(f"\nPK cache written: {cache_path}")
    print(f"  {args.model} knows {knows}/{len(results)} "
          f"({100*knows/max(len(results),1):.1f}%)")


if __name__ == "__main__":
    main()
