"""
validate_benchmark_llm.py

Uses an LLM as a proxy annotator to validate SufficiencyBench labels.
For each sampled example, asks the model to judge whether the context
contains enough information to answer the question.

Then compares LLM annotations against the benchmark labels to compute
agreement rate and Cohen's kappa.

Usage:
  python scripts/validate_benchmark_llm.py --model mistral --device cuda:0
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
from tqdm import tqdm
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.paths import RESULTS_DIR, MODEL_CONFIGS


VALIDATION_PROMPT = """You are a careful annotator. Your task is to determine whether the given context contains sufficient information to correctly answer the question.

Rules:
- ONLY consider information explicitly stated in the context
- Do NOT use your own knowledge to fill gaps
- If the context provides the answer or enough clues to derive it, answer SUFFICIENT
- If the answer requires information NOT in the context, answer INSUFFICIENT

Context: {context}

Question: {question}

Gold answer (for reference): {gold_answer}

Does the context contain sufficient information to answer the question correctly?
Answer SUFFICIENT or INSUFFICIENT only."""


def run_validation(model_key="mistral", device="cuda:0"):
    print(f"\n{'='*60}")
    print(f"BENCHMARK VALIDATION — {model_key}")
    print(f"{'='*60}")

    # Load model
    cfg = MODEL_CONFIGS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        quantization_config=bnb_config,
        device_map=device,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    # Load validation samples and key
    val_dir = RESULTS_DIR / "validation"
    with open(val_dir / "validation_samples.json") as f:
        samples = json.load(f)
    with open(val_dir / "validation_key.json") as f:
        key = {item["id"]: item["label"] for item in json.load(f)}

    print(f"Validating {len(samples)} examples...")

    llm_labels = []
    true_labels = []
    raw_responses = []

    for item in tqdm(samples, desc="Validating"):
        prompt = VALIDATION_PROMPT.format(
            context=item["context"],
            question=item["question"],
            gold_answer=item["gold_answer"],
        )

        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=2048,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=10,
                temperature=0.0, do_sample=False,
            )

        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip().upper()
        raw_responses.append(response[:50])

        # Parse
        if "SUFFICIENT" in response and "INSUFFICIENT" not in response:
            llm_label = True
        elif "INSUFFICIENT" in response:
            llm_label = False
        else:
            # Ambiguous - default to sufficient (conservative)
            llm_label = True

        llm_labels.append(llm_label)
        true_labels.append(key[item["id"]])

    # Compute agreement
    llm_binary = [1 if l else 0 for l in llm_labels]
    true_binary = [1 if l else 0 for l in true_labels]

    agreement = accuracy_score(true_binary, llm_binary)
    kappa = cohen_kappa_score(true_binary, llm_binary)
    cm = confusion_matrix(true_binary, llm_binary)

    # Per question type
    per_type = {}
    for item, llm_l, true_l in zip(samples, llm_binary, true_binary):
        qtype = item["question_type"]
        if qtype not in per_type:
            per_type[qtype] = {"correct": 0, "total": 0}
        per_type[qtype]["total"] += 1
        if llm_l == true_l:
            per_type[qtype]["correct"] += 1

    results = {
        "model": model_key,
        "n_examples": len(samples),
        "agreement_rate": float(agreement),
        "cohen_kappa": float(kappa),
        "confusion_matrix": cm.tolist(),
        "per_question_type": {
            qtype: {
                "n": v["total"],
                "agreement": v["correct"] / v["total"] if v["total"] > 0 else 0,
            }
            for qtype, v in per_type.items()
        },
    }

    print(f"\nResults:")
    print(f"  Agreement rate: {agreement:.3f}")
    print(f"  Cohen's kappa: {kappa:.3f}")
    print(f"  Confusion matrix:\n{cm}")
    for qtype, v in results["per_question_type"].items():
        print(f"  {qtype}: {v['agreement']:.3f} (n={v['n']})")

    out_path = val_dir / f"human_validation_{model_key}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    del model
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run_validation(model_key=args.model, device=args.device)
