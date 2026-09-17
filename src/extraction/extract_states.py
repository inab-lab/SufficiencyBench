"""
extract_states.py — Hidden-state extraction for SufficiencyBench (Paper 1).

For each benchmark example we run the frozen LLM forward TWICE and cache the
LAST-token residual-stream state at EVERY layer:

  1. h(q + c)  — question + context (the standard RAG input)  -> h_with_context.npy
  2. h(q)      — question alone, no context                    -> h_question_only.npy

The DECO / CSP (Context Subtraction Probe) input is the delta d = h(q+c) - h(q),
which isolates what the context CONTRIBUTES, removing the model's baseline
(parametric) confidence.  The Standard probe uses h(q+c) directly.

Models are loaded from LOCAL dirs via configs.paths.MODEL_CONFIGS with 8-bit
quantization (bfloat16 compute).  All layers are extracted; the per-layer sweep
and validation-based layer selection happen downstream in methods/deco.py.

Caches are written to:
  data/experiments/hidden_states/<model>/<split>/{h_with_context.npy,
                                                  h_question_only.npy,
                                                  labels.npy,
                                                  metadata.json}

Usage:
  conda run -n csp python src/extraction/extract_states.py --model mistral --device cuda:0
  conda run -n csp python src/extraction/extract_states.py --model llama   --device cuda:1
  conda run -n csp python src/extraction/extract_states.py --model qwen    --device cuda:0
"""

import argparse
import json
import os
import sys
import gc
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import BENCH_DIR, HIDDEN_STATES_DIR, MODEL_CONFIGS

MAX_SEQ_LEN = 2048


def format_question_only(question: str) -> str:
    """Prompt with NO context — for the h(q) baseline.

    Mirrors the phrasing used to build the with-context prompts so that the only
    difference between h(q+c) and h(q) is the presence of the context itself.
    """
    return (
        f"Based on the following context, answer the question. "
        f"If the context does not contain enough information, "
        f"say 'I cannot answer this from the provided context.'\n\n"
        f"Context: [No context provided]\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def extract_last_token_hidden(model, tokenizer, text: str, device) -> np.ndarray:
    """Last-token hidden state at every layer. Returns (n_layers, hidden_dim)."""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states[1:]  # skip embedding layer
    n_layers = len(hidden_states)
    hidden_dim = hidden_states[0].shape[-1]

    result = np.zeros((n_layers, hidden_dim), dtype=np.float32)
    for i, hs in enumerate(hidden_states):
        result[i] = hs[0, -1, :].float().cpu().numpy()
    return result


def extract_hidden_states(model, tokenizer, examples, device=None):
    """Convenience API (used in the README quick-start).

    Runs both forward passes for a list of flattened condition dicts (each with a
    'prompt' for h(q+c) and a 'question' for h(q)) and returns two arrays:
        h_with_ctx : (N, n_layers, hidden_dim)
        h_question : (N, n_layers, hidden_dim)
    The DECO delta is then h_with_ctx - h_question.
    """
    if device is None:
        device = next(model.parameters()).device
    h_with_ctx, h_question = [], []
    for ex in tqdm(examples, desc="extract_hidden_states"):
        prompt = ex["prompt"] if "prompt" in ex else ex["question"]
        h_with_ctx.append(extract_last_token_hidden(model, tokenizer, prompt, device))
        q_only = format_question_only(ex["question"])
        h_question.append(extract_last_token_hidden(model, tokenizer, q_only, device))
    return np.stack(h_with_ctx), np.stack(h_question)


def load_model(cfg, device):
    """Load a local model in 8-bit (bfloat16 compute), eval mode."""
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        quantization_config=bnb_config,
        device_map=device,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


def extract_split(split, model, tokenizer, model_key, device):
    """Extract and cache hidden states for one benchmark split."""
    data_path = BENCH_DIR / f"{split}.json"
    if not data_path.exists():
        print(f"  {split}: {data_path} not found, skipping")
        return

    with open(data_path) as f:
        examples = json.load(f)

    out_dir = HIDDEN_STATES_DIR / model_key / split
    out_dir.mkdir(parents=True, exist_ok=True)

    all_with_context = []   # h(q + c)
    all_question_only = []  # h(q)
    all_labels = []         # 1 = sufficient, 0 = insufficient
    all_quadrants = []
    all_metadata = []

    for ex in tqdm(examples, desc=split):
        question = ex["question"]

        # h(question only) — identical across all conditions of this example
        q_only_prompt = format_question_only(question)
        h_q = extract_last_token_hidden(model, tokenizer, q_only_prompt, device)

        for cond in ex["conditions"]:
            try:
                h_qc = extract_last_token_hidden(
                    model, tokenizer, cond["prompt"], device
                )
                all_with_context.append(h_qc)
                all_question_only.append(h_q)
                all_labels.append(1 if cond["sufficient"] else 0)
                all_quadrants.append(cond["quadrant"])
                all_metadata.append({
                    "id": cond["condition_id"],
                    "question_type": ex["question_type"],
                    "quadrant": cond["quadrant"],
                    "sufficient": cond["sufficient"],
                    "model_confident": cond["model_confident"],
                })
            except Exception as e:  # noqa: BLE001 — keep going on a single bad example
                print(f"Error on {cond.get('condition_id', '?')}: {e}")
                continue

        if len(all_labels) % 200 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    np.save(out_dir / "h_with_context.npy", np.stack(all_with_context))
    np.save(out_dir / "h_question_only.npy", np.stack(all_question_only))
    np.save(out_dir / "labels.npy", np.array(all_labels))
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(all_metadata, f)

    n = len(all_labels)
    print(f"Saved {n} conditions to {out_dir}")
    print(f"  Sufficient: {sum(all_labels)}, Insufficient: {n - sum(all_labels)}")
    print(f"  Q1: {all_quadrants.count('Q1')}, Q2: {all_quadrants.count('Q2')}, "
          f"Q3: {all_quadrants.count('Q3')}, Q4: {all_quadrants.count('Q4')}")


def extract_for_model(model_key, device, splits):
    cfg = MODEL_CONFIGS[model_key]
    print(f"\n{'='*60}")
    print(f"Extracting states for {model_key} from {cfg['path']}")
    print(f"{'='*60}")
    model, tokenizer = load_model(cfg, device)

    for split in splits:
        print(f"\nProcessing {split}...")
        extract_split(split, model, tokenizer, model_key, device)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    extract_for_model(args.model, args.device, args.splits)
    print("\nAll extractions done.")


if __name__ == "__main__":
    main()
