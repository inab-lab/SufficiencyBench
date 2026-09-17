#!/usr/bin/env python3
"""
Run parametric knowledge (PK) tests on all three models.

Produces per-model PK annotations for the existing benchmark questions.
The benchmark contexts/questions don't change — only the quadrant assignments
become model-specific.

Usage:
  python scripts/run_pk_all_models.py --model llama --device cuda:0
  python scripts/run_pk_all_models.py --model qwen --device cuda:1
  python scripts/run_pk_all_models.py --model all
"""

import json
import sys
import os
import argparse
import signal
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.paths import BENCH_DIR, MODEL_CONFIGS

# Ignore SIGTERM (this system sends it to long-running processes)
for _sig in (signal.SIGTERM, signal.SIGHUP):
    try:
        signal.signal(_sig, signal.SIG_IGN)
    except (OSError, ValueError):
        pass


def load_model(model_key: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    cfg = MODEL_CONFIGS[model_key]
    print(f"Loading {model_key} from {cfg['path']}...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def test_pk_single(model, tokenizer, question: str, gold_answer: str, device: str):
    """Test if model can answer a question without context."""
    prompt = (
        f"Answer this question in one sentence. "
        f"If you don't know, say 'I don't know.'\n\n"
        f"Question: {question}\nAnswer:"
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=60,
            temperature=0.0, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
        )

    response = tokenizer.decode(
        outputs.sequences[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    response_lower = response.lower()
    answer_lower = gold_answer.lower()

    abstain_phrases = [
        "i don't know", "i'm not sure", "i cannot",
        "don't have enough", "not sure", "i do not know",
        "unable to", "no information", "cannot determine",
    ]
    abstains = any(p in response_lower for p in abstain_phrases)
    knows = (answer_lower in response_lower) and not abstains

    # Confidence from token probs
    if outputs.scores:
        confidences = []
        for score in outputs.scores[:5]:
            probs = torch.softmax(score[0], dim=-1)
            confidences.append(probs.max().item())
        confidence = float(np.mean(confidences))
    else:
        confidence = 0.5

    return {
        "knows": knows,
        "closed_book_answer": response[:300],
        "confidence": confidence,
    }


def run_pk_for_model(model_key: str, device: str):
    """Run PK test on all benchmark questions for a single model."""
    cache_path = BENCH_DIR / f"pk_cache_{model_key}.json"

    # Load cached results
    pk_results = {}
    if cache_path.exists():
        with open(cache_path) as f:
            pk_results = json.load(f)
        print(f"Loaded {len(pk_results)} cached PK results for {model_key}")

    # Load all benchmark questions
    all_questions = []
    for split in ["train", "val", "test"]:
        path = BENCH_DIR / f"{split}.json"
        if path.exists():
            data = json.load(open(path))
            all_questions.extend(data)
    print(f"Total questions: {len(all_questions)}")

    # Filter to remaining
    remaining = [q for q in all_questions if q["id"] not in pk_results]
    print(f"Already done: {len(pk_results)}, remaining: {len(remaining)}")

    if not remaining:
        print("All done!")
        return pk_results

    # Load model
    model, tokenizer = load_model(model_key, device)

    for i, q in enumerate(tqdm(remaining, desc=f"PK test ({model_key})")):
        result = test_pk_single(
            model, tokenizer, q["question"], q["gold_answer"], device
        )
        pk_results[q["id"]] = result

        # Save cache every 50 questions
        if (i + 1) % 50 == 0:
            with open(cache_path, "w") as f:
                json.dump(pk_results, f)

    # Final save
    with open(cache_path, "w") as f:
        json.dump(pk_results, f)

    # Summary
    knows = sum(1 for v in pk_results.values() if v["knows"])
    total = len(pk_results)
    print(f"\n{model_key} PK Results: {knows}/{total} knows ({knows/total*100:.1f}%)")

    # Free GPU
    del model
    torch.cuda.empty_cache()

    return pk_results


def update_benchmark_with_model_pk(model_key: str, pk_results: dict):
    """Add model-specific PK annotations to each benchmark split."""
    for split in ["train", "val", "test"]:
        path = BENCH_DIR / f"{split}.json"
        if not path.exists():
            continue

        data = json.load(open(path))
        for ex in data:
            pk = pk_results.get(ex["id"], {"knows": False, "confidence": 0.5})
            # Add model-specific fields
            ex[f"pk_{model_key}"] = {
                "model_knows_answer": pk["knows"],
                "confidence": pk["confidence"],
                "closed_book_answer": pk.get("closed_book_answer", ""),
            }
            # Update conditions with model-specific quadrants
            for cond in ex["conditions"]:
                model_confident = pk["knows"]
                if cond["sufficient"]:
                    cond[f"quadrant_{model_key}"] = "Q1" if model_confident else "Q2"
                else:
                    cond[f"quadrant_{model_key}"] = "Q3" if model_confident else "Q4"

        # Save back
        with open(path, "w") as f:
            json.dump(data, f, indent=1)
        print(f"Updated {split}.json with {model_key} PK annotations")

    # Report quadrant agreement with Mistral
    test_data = json.load(open(BENCH_DIR / "test.json"))
    agree = 0
    total = 0
    for ex in test_data:
        if f"pk_{model_key}" in ex:
            mistral_knows = ex.get("model_knows_answer", False)
            model_knows = ex[f"pk_{model_key}"]["model_knows_answer"]
            if mistral_knows == model_knows:
                agree += 1
            total += 1
    if total > 0:
        print(f"\nQuadrant agreement with Mistral: {agree}/{total} ({agree/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llama", "qwen", "mistral", "all"], required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.environ["MALLOC_ARENA_MAX"] = "1"

    models = [args.model] if args.model != "all" else ["llama", "qwen", "mistral"]

    for model_key in models:
        print(f"\n{'='*60}")
        print(f"PK TEST: {model_key}")
        print(f"{'='*60}")
        pk_results = run_pk_for_model(model_key, args.device)
        update_benchmark_with_model_pk(model_key, pk_results)
        print(f"\nDone with {model_key}!")


if __name__ == "__main__":
    main()
