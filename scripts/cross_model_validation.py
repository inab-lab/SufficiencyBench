"""
cross_model_validation.py

Uses ALL 3 models to independently judge whether benchmark contexts are sufficient.
Computes cross-model agreement and majority-vote validation rate.

This addresses the concern that self-validation (Mistral judging its own labels)
only achieved 55.5% agreement.

Usage:
  python scripts/cross_model_validation.py --model mistral --device cuda:0
  python scripts/cross_model_validation.py --model llama --device cuda:1
  python scripts/cross_model_validation.py --model qwen --device cuda:0
  python scripts/cross_model_validation.py --aggregate  # after all models done
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import cohen_kappa_score, accuracy_score
from tqdm import tqdm
import argparse
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.paths import BENCH_DIR, RESULTS_DIR, MODEL_CONFIGS


VALIDATION_PROMPT = """You are a careful annotator evaluating whether a context passage contains sufficient information to answer a question.

Rules:
- ONLY consider information explicitly stated in the context
- Do NOT use your own knowledge to fill gaps
- If the context contains the answer or enough information to derive it, say SUFFICIENT
- If key information is missing from the context, say INSUFFICIENT

Context: {context}

Question: {question}

Does the context contain sufficient information to correctly answer this question?
Answer with SUFFICIENT or INSUFFICIENT only."""


def run_single_model_validation(model_key, device="cuda:0", n_examples=500, seed=42):
    print(f"\n{'='*60}")
    print(f"CROSS-MODEL VALIDATION — {model_key}")
    print(f"{'='*60}")

    cfg = MODEL_CONFIGS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], quantization_config=bnb_config,
        device_map=device, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
    )
    model.eval()

    # Load test data and sample
    with open(BENCH_DIR / "test.json") as f:
        test_data = json.load(f)

    conditions = []
    for ex in test_data:
        for cond in ex["conditions"]:
            conditions.append({
                "id": cond["condition_id"],
                "question": ex["question"],
                "context": cond["context"],
                "gold_answer": ex["gold_answer"],
                "question_type": ex["question_type"],
                "sufficient": cond["sufficient"],
                "quadrant": cond["quadrant"],
            })

    random.seed(seed)
    if n_examples > 0 and n_examples < len(conditions):
        # Stratified sample: equal sufficient/insufficient
        suf = [c for c in conditions if c["sufficient"]]
        insuf = [c for c in conditions if not c["sufficient"]]
        n_per = n_examples // 2
        sampled = random.sample(suf, min(n_per, len(suf))) + \
                  random.sample(insuf, min(n_per, len(insuf)))
        random.shuffle(sampled)
        conditions = sampled

    print(f"Evaluating {len(conditions)} examples...")

    predictions = []
    for cond in tqdm(conditions, desc=f"Validating ({model_key})"):
        prompt = VALIDATION_PROMPT.format(context=cond["context"], question=cond["question"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10, temperature=0.0, do_sample=False)

        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip().upper()

        if "INSUFFICIENT" in response:
            pred = False
        elif "SUFFICIENT" in response:
            pred = True
        else:
            pred = True  # Default to sufficient if ambiguous

        predictions.append({
            "id": cond["id"],
            "true_label": cond["sufficient"],
            "predicted": pred,
            "correct": pred == cond["sufficient"],
            "quadrant": cond["quadrant"],
            "question_type": cond["question_type"],
            "response": response[:50],
        })

    # Compute metrics
    true_labels = [p["true_label"] for p in predictions]
    pred_labels = [p["predicted"] for p in predictions]
    agreement = accuracy_score(true_labels, pred_labels)
    kappa = cohen_kappa_score(
        [1 if t else 0 for t in true_labels],
        [1 if p else 0 for p in pred_labels]
    )

    # Per-quadrant
    per_quad = {}
    for quad in ["Q1", "Q2", "Q3", "Q4"]:
        qp = [p for p in predictions if p["quadrant"] == quad]
        if len(qp) < 5:
            continue
        per_quad[quad] = {
            "n": len(qp),
            "agreement": sum(1 for p in qp if p["correct"]) / len(qp),
            "pred_sufficient_rate": sum(1 for p in qp if p["predicted"]) / len(qp),
        }

    result = {
        "model": model_key,
        "n_examples": len(predictions),
        "agreement_rate": float(agreement),
        "cohen_kappa": float(kappa),
        "pred_sufficient_rate": sum(1 for p in predictions if p["predicted"]) / len(predictions),
        "per_quadrant": per_quad,
        "predictions": predictions,
    }

    print(f"\n  Agreement: {agreement:.3f}, Kappa: {kappa:.3f}")
    print(f"  Predicted sufficient: {result['pred_sufficient_rate']:.3f}")
    for q, v in per_quad.items():
        print(f"  {q}: agreement={v['agreement']:.3f}, pred_suf_rate={v['pred_sufficient_rate']:.3f}")

    out_dir = RESULTS_DIR / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cross_validation_{model_key}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {out_path}")

    del model
    torch.cuda.empty_cache()
    return result


def aggregate_results():
    """Aggregate cross-model validation results."""
    print(f"\n{'='*60}")
    print("AGGREGATING CROSS-MODEL VALIDATION")
    print(f"{'='*60}")

    val_dir = RESULTS_DIR / "validation"
    results = {}
    all_preds = {}

    for model_key in ["llama", "mistral", "qwen"]:
        path = val_dir / f"cross_validation_{model_key}.json"
        if not path.exists():
            print(f"  Missing: {path}")
            continue
        with open(path) as f:
            results[model_key] = json.load(f)
        # Index predictions by ID
        for p in results[model_key]["predictions"]:
            if p["id"] not in all_preds:
                all_preds[p["id"]] = {"true_label": p["true_label"], "quadrant": p["quadrant"]}
            all_preds[p["id"]][model_key] = p["predicted"]

    # Compute majority vote
    n_majority_correct = 0
    n_total = 0
    for pid, info in all_preds.items():
        models_with_pred = [m for m in ["llama", "mistral", "qwen"] if m in info]
        if len(models_with_pred) < 2:
            continue
        votes = [info[m] for m in models_with_pred]
        majority = sum(votes) > len(votes) / 2
        if majority == info["true_label"]:
            n_majority_correct += 1
        n_total += 1

    # Inter-model agreement
    inter_model = {}
    for m1 in ["llama", "mistral", "qwen"]:
        for m2 in ["llama", "mistral", "qwen"]:
            if m1 >= m2:
                continue
            shared = [pid for pid, info in all_preds.items() if m1 in info and m2 in info]
            if len(shared) < 10:
                continue
            agree = sum(1 for pid in shared if all_preds[pid][m1] == all_preds[pid][m2])
            inter_model[f"{m1}_vs_{m2}"] = {
                "n": len(shared),
                "agreement": agree / len(shared),
            }

    aggregate = {
        "n_examples_with_majority": n_total,
        "majority_vote_accuracy": n_majority_correct / max(n_total, 1),
        "per_model_agreement": {m: results[m]["agreement_rate"] for m in results},
        "per_model_kappa": {m: results[m]["cohen_kappa"] for m in results},
        "per_model_pred_sufficient_rate": {m: results[m]["pred_sufficient_rate"] for m in results},
        "inter_model_agreement": inter_model,
    }

    print(f"\n  Per-model agreement with ground truth:")
    for m, a in aggregate["per_model_agreement"].items():
        print(f"    {m}: {a:.3f}")
    print(f"\n  Majority vote accuracy: {aggregate['majority_vote_accuracy']:.3f} (n={n_total})")
    print(f"\n  Inter-model agreement:")
    for pair, info in inter_model.items():
        print(f"    {pair}: {info['agreement']:.3f} (n={info['n']})")

    out_path = val_dir / "cross_model_validation.json"
    with open(out_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\n  Saved to {out_path}")
    return aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_examples", type=int, default=500)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    if args.aggregate:
        aggregate_results()
    elif args.model:
        run_single_model_validation(args.model, args.device, args.n_examples)
    else:
        print("Specify --model or --aggregate")
