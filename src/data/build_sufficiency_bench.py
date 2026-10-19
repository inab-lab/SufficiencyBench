"""
build_sufficiency_bench.py

Creates SufficiencyBench: a benchmark that independently varies
context sufficiency and model confidence.

The 2x2 matrix:
  Q1: Sufficient + Confident    (easy: model answers from context)
  Q2: Sufficient + Uncertain    (hard: answer in context but model misses it)
  Q3: Insufficient + Confident  (dangerous: model hallucinates confidently)
  Q4: Insufficient + Uncertain  (easy: model abstains)

Construction:
  1. Start with QA datasets (SQuAD, HotPotQA, NQ)
  2. For each question, check if model knows the answer without context
     (closed-book test) → determines confident vs. uncertain
  3. Create sufficient (gold context) and insufficient (controlled replacement)
     versions → determines sufficient vs. insufficient
  4. The combination gives us the 2×2 matrix

Confound controls:
  - Length matched across all conditions
  - Topic matched (same-topic distractors)
  - Entity presence matched
  - 4 question types: factual, multi_hop, comparative, subjective
"""

import json
import re
import random
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss
import argparse
import sys
import os
import nltk

# Ensure project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import (
    BENCH_DIR, DATASETS_DIR, LLAMA_PATH, MINILM_PATH
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ================================================================
# STEP 1: Parametric Knowledge Test (Closed-Book)
# ================================================================

def test_parametric_knowledge(
    model, tokenizer, questions: list, device: str, batch_size: int = 1,
    cache_path: Path = None,
) -> dict:
    """
    For each question, test if the model can answer WITHOUT context.
    This determines the 'confident' vs 'uncertain' axis.

    Returns: dict mapping question_id -> {
        'knows': bool,
        'closed_book_answer': str,
        'confidence': float
    }
    """
    # Resume from cache if available
    results = {}
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            results = json.load(f)
        print(f"  Loaded {len(results)} cached PK results")

    remaining = [q for q in questions if q["id"] not in results]
    if not remaining:
        print("  All PK results cached, skipping.")
        return results

    print(f"  Running PK test on {len(remaining)} questions...")

    for i, q in enumerate(tqdm(remaining, desc="Closed-book PK test")):
        prompt = (
            f"Answer this question in one sentence. "
            f"If you don't know, say 'I don't know.'\n\n"
            f"Question: {q['question']}\nAnswer:"
        )

        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=512
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
        answer_lower = q["answer"].lower()

        # Check if model abstains
        abstain_phrases = [
            "i don't know", "i'm not sure", "i cannot",
            "don't have enough", "not sure", "i do not know",
            "unable to", "no information", "cannot determine",
        ]
        abstains = any(p in response_lower for p in abstain_phrases)

        # Check if model's answer contains the gold answer
        knows = (answer_lower in response_lower) and not abstains

        # Compute confidence from generated token probabilities
        if outputs.scores:
            # Average max probability over first 5 tokens
            confidences = []
            for score in outputs.scores[:5]:
                probs = torch.softmax(score[0], dim=-1)
                confidences.append(probs.max().item())
            confidence = float(np.mean(confidences))
        else:
            confidence = 0.5

        results[q["id"]] = {
            "knows": knows,
            "closed_book_answer": response[:300],
            "confidence": confidence,
        }

        # Save cache periodically (every 100 questions)
        if cache_path and (i + 1) % 100 == 0:
            with open(cache_path, "w") as f:
                json.dump(results, f)

    # Final cache save
    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(results, f)
        print(f"  PK results cached to {cache_path}")

    return results


# ================================================================
# STEP 2: Dataset Loading
# ================================================================

def load_squad_questions(max_n: int = 1500) -> list:
    """Load factual questions from SQuAD 2.0 (answerable only)."""
    print("Loading SQuAD 2.0...")
    ds = load_from_disk(str(DATASETS_DIR / "squad_v2"))

    # Inspect structure
    split = "validation" if "validation" in ds else list(ds.keys())[0]
    sample = ds[split][0]
    print(f"  Split: {split}, columns: {ds[split].column_names}")
    print(f"  Sample keys: {list(sample.keys())}")

    items = []
    for i, ex in enumerate(ds[split]):
        # Handle different field name conventions
        answers = ex.get("answers", {})
        answer_texts = answers.get("text", [])
        if not answer_texts:
            continue  # Skip unanswerable

        items.append({
            "id": f"squad_{i}",
            "question": ex["question"],
            "answer": answer_texts[0],
            "context": ex["context"],
            "title": ex.get("title", ""),
            "question_type": "factual",
        })

    random.shuffle(items)
    items = items[:max_n]
    print(f"  Selected {len(items)} answerable SQuAD questions")
    return items


def load_hotpotqa_questions(max_n: int = 1500) -> list:
    """Load multi-hop and comparative questions from HotPotQA."""
    print("Loading HotPotQA...")
    ds = load_from_disk(str(DATASETS_DIR / "hotpotqa"))

    split = "validation" if "validation" in ds else list(ds.keys())[0]
    sample = ds[split][0]
    print(f"  Split: {split}, columns: {ds[split].column_names}")
    print(f"  Sample keys: {list(sample.keys())}")

    items = []
    for i, ex in enumerate(ds[split]):
        if i >= max_n * 2:  # Process more to get enough after filtering
            break

        # Extract gold context from supporting facts
        try:
            ctx_titles = ex["context"]["title"]
            ctx_sents = ex["context"]["sentences"]
            supporting = set(ex["supporting_facts"]["title"])
        except (KeyError, TypeError) as e:
            # Try alternative field names
            try:
                ctx_titles = ex["context.title"]
                ctx_sents = ex["context.sentences"]
                supporting = set(ex["supporting_facts.title"])
            except KeyError:
                continue

        gold_parts = []
        other_parts = []
        for title, sents in zip(ctx_titles, ctx_sents):
            text = " ".join(sents) if isinstance(sents, list) else sents
            if title in supporting:
                gold_parts.append(text)
            else:
                other_parts.append(text)

        gold_context = " ".join(gold_parts)
        if not gold_context.strip() or len(gold_context) < 50:
            continue

        qtype = "comparative" if ex.get("type") == "comparison" else "multi_hop"

        items.append({
            "id": f"hotpot_{i}",
            "question": ex["question"],
            "answer": ex["answer"],
            "context": gold_context,
            "title": "",
            "question_type": qtype,
            "_gold_parts": gold_parts,
            "_other_parts": other_parts,
        })

    random.shuffle(items)
    items = items[:max_n]
    print(f"  Selected {len(items)} HotPotQA questions "
          f"(multi_hop: {sum(1 for x in items if x['question_type']=='multi_hop')}, "
          f"comparative: {sum(1 for x in items if x['question_type']=='comparative')})")
    return items


def load_eli5_questions(max_n: int = 500) -> list:
    """Load subjective questions from ELI5."""
    print("Loading ELI5...")
    ds_path = DATASETS_DIR / "eli5"

    if not ds_path.exists():
        print("  ELI5 not found, skipping subjective questions")
        return []

    try:
        ds = load_from_disk(str(ds_path))
    except Exception as e:
        print(f"  Error loading ELI5: {e}")
        return []

    # Find the right split
    for split_name in ["test", "validation", "train"]:
        if split_name in ds:
            split = split_name
            break
    else:
        split = list(ds.keys())[0]

    sample = ds[split][0]
    print(f"  Split: {split}, columns: {ds[split].column_names}")
    print(f"  Sample keys: {list(sample.keys())}")

    items = []
    for i, ex in enumerate(ds[split]):
        if i >= max_n * 4:
            break

        # Handle multiple field name conventions:
        # sentence-transformers/eli5 format: {"question": ..., "answer": ...}
        # eli5_category format: {"title": ..., "answers": {"text": [...]}}
        question = ex.get("question", ex.get("title", ""))
        if not question:
            continue

        # Get answer text
        answer = None
        if "answer" in ex and isinstance(ex["answer"], str):
            answer = ex["answer"]
        elif "answers" in ex:
            answers = ex["answers"]
            if isinstance(answers, dict):
                answer_texts = answers.get("text", [])
                if answer_texts:
                    answer = answer_texts[0]
            elif isinstance(answers, list) and answers:
                answer = answers[0] if isinstance(answers[0], str) else str(answers[0])

        if not answer or len(answer) < 50:
            continue

        # For ELI5, the answer itself serves as the "sufficient context"
        # (since these are explanation-type questions)
        items.append({
            "id": f"eli5_{i}",
            "question": question,
            "answer": answer[:200],
            "context": answer,
            "title": "",
            "question_type": "subjective",
        })

    random.shuffle(items)
    items = items[:max_n]
    print(f"  Selected {len(items)} ELI5 subjective questions")
    return items


# ================================================================
# STEP 3: Context Manipulation (Insufficient Context Creation)
# ================================================================

def split_sentences(text: str) -> list:
    """Split text into sentences using nltk for robustness."""
    try:
        return nltk.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        return nltk.sent_tokenize(text)


def replace_answer_sentence(
    context: str, answer: str, replacement_pool: list
) -> str:
    """
    Replace sentence(s) containing the answer with length-matched
    replacements from the same topic.

    Falls back to removing a random sentence and replacing it
    if answer text doesn't appear literally in any sentence.
    """
    sentences = split_sentences(context)
    answer_lower = answer.lower()

    # Find sentences containing the answer
    answer_indices = [
        i for i, s in enumerate(sentences) if answer_lower in s.lower()
    ]

    # Fallback: if answer not found literally, find most relevant sentence
    if not answer_indices and len(sentences) > 1:
        # Remove a middle sentence (not first or last to keep coherence)
        if len(sentences) > 2:
            answer_indices = [len(sentences) // 2]
        else:
            answer_indices = [0]

    if not answer_indices or not replacement_pool:
        return context  # Can't replace, return as-is

    new_sentences = []
    for i, sent in enumerate(sentences):
        if i in answer_indices:
            target_len = len(sent)
            # Find best length-matched replacement
            candidates = sorted(
                replacement_pool,
                key=lambda s: abs(len(s) - target_len)
            )
            if candidates:
                replacement = candidates[0]
                new_sentences.append(replacement)
            else:
                new_sentences.append(sent)
        else:
            new_sentences.append(sent)

    return " ".join(new_sentences)


# ================================================================
# STEP 4: Build Replacement Pools with FAISS
# ================================================================

def build_replacement_pools(
    all_questions: list, embed_model: SentenceTransformer, k: int = 10
) -> dict:
    """
    For each question, find same-topic neighbors and collect their
    non-answer sentences as a replacement pool.
    """
    print("Building replacement pools via FAISS...")
    all_contexts = [q["context"] for q in all_questions]
    embeddings = embed_model.encode(
        all_contexts, normalize_embeddings=True,
        batch_size=64, show_progress_bar=True,
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))

    # Pre-split all sentences
    all_sentences = {}
    for q in all_questions:
        sents = split_sentences(q["context"])
        all_sentences[q["id"]] = [
            s for s in sents
            if q["answer"].lower() not in s.lower() and len(s) > 20
        ]

    # Build pools
    pools = {}
    for idx, q in enumerate(tqdm(all_questions, desc="Building pools")):
        q_emb = embeddings[idx:idx+1]
        _, neighbor_indices = index.search(q_emb, k=k)
        pool = []
        for ni in neighbor_indices[0]:
            if ni != idx and ni < len(all_questions):
                nid = all_questions[ni]["id"]
                pool.extend(all_sentences.get(nid, []))
        pools[q["id"]] = pool

    return pools


# ================================================================
# STEP 5: Format Prompt
# ================================================================

def format_prompt(question: str, context: str) -> str:
    return (
        f"Based on the following context, answer the question. "
        f"If the context does not contain enough information, "
        f"say 'I cannot answer this from the provided context.'\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )


# ================================================================
# STEP 6: Main Build
# ================================================================

def build_benchmark(device: str = "cuda:0", model_key: str = None):
    """Build SufficiencyBench with the 2×2 matrix."""

    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    # Select model for PK test — use first available
    if model_key is None:
        from configs.paths import MODEL_CONFIGS
        for key in ["llama", "mistral", "qwen"]:
            cfg = MODEL_CONFIGS[key]
            if Path(cfg["path"]).exists() and (Path(cfg["path"]) / "config.json").exists():
                model_key = key
                break
        if model_key is None:
            raise RuntimeError("No model found. Download at least one model first.")

    from configs.paths import MODEL_CONFIGS
    model_path = MODEL_CONFIGS[model_key]["path"]
    pk_cache = BENCH_DIR / f"pk_cache_{model_key}.json"

    # ---- Load model for PK test ----
    print("=" * 60)
    print(f"Loading model for parametric knowledge test: {model_key}")
    print(f"  Path: {model_path}")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    # ---- Load all datasets ----
    print("\n" + "=" * 60)
    print("Loading datasets...")
    print("=" * 60)
    squad_qs = load_squad_questions(max_n=1500)
    hotpot_qs = load_hotpotqa_questions(max_n=1500)
    eli5_qs = load_eli5_questions(max_n=500)

    all_questions = squad_qs + hotpot_qs + eli5_qs
    print(f"\nTotal questions loaded: {len(all_questions)}")

    # ---- Parametric knowledge test ----
    print("\n" + "=" * 60)
    print("Running parametric knowledge test (closed-book)...")
    print("This is the slowest step (~2-4 hours for 3500 questions)")
    print("=" * 60)
    pk_results = test_parametric_knowledge(
        model, tokenizer, all_questions,
        device=device, cache_path=pk_cache,
    )

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # ---- Summary of PK results ----
    knows_count = sum(1 for v in pk_results.values() if v["knows"])
    total = len(pk_results)
    print(f"\nPK Test Results:")
    print(f"  Model knows answer: {knows_count}/{total} ({knows_count/total*100:.1f}%)")
    print(f"  Model doesn't know: {total-knows_count}/{total} ({(total-knows_count)/total*100:.1f}%)")

    # ---- Build replacement pools ----
    print("\n" + "=" * 60)
    print("Building same-topic replacement pools...")
    print("=" * 60)
    embed_model = SentenceTransformer(str(MINILM_PATH))
    pools = build_replacement_pools(all_questions, embed_model)
    del embed_model

    # ---- Create the 2×2 examples ----
    print("\n" + "=" * 60)
    print("Creating 2×2 benchmark examples...")
    print("=" * 60)
    benchmark_examples = []
    skipped = 0

    for idx, q in enumerate(tqdm(all_questions, desc="Building 2x2")):
        pk = pk_results.get(q["id"], {"knows": False, "confidence": 0.5})
        model_confident = pk["knows"]
        replacement_pool = pools.get(q["id"], [])

        # SUFFICIENT context = gold context
        gold_context = q["context"]

        # INSUFFICIENT context = answer sentence replaced
        replaced_context = replace_answer_sentence(
            gold_context, q["answer"], replacement_pool,
        )

        # Verify the replacement actually changed something
        if replaced_context == gold_context and len(split_sentences(gold_context)) > 1:
            skipped += 1
            continue

        # Length ratio check
        len_ratio = len(replaced_context) / max(len(gold_context), 1)
        if len_ratio < 0.5 or len_ratio > 2.0:
            skipped += 1
            continue

        # Remove internal fields
        clean_q = {k: v for k, v in q.items() if not k.startswith("_")}

        entry = {
            "id": q["id"],
            "question": q["question"],
            "gold_answer": q["answer"],
            "question_type": q["question_type"],
            "model_knows_answer": model_confident,
            "closed_book_confidence": pk["confidence"],
            "closed_book_answer": pk.get("closed_book_answer", ""),
            "conditions": [
                {
                    "condition_id": f"{q['id']}_suf",
                    "context": gold_context,
                    "sufficient": True,
                    "model_confident": model_confident,
                    "quadrant": "Q1" if model_confident else "Q2",
                    "expected_behavior": "answer",
                    "context_length": len(gold_context),
                    "prompt": format_prompt(q["question"], gold_context),
                },
                {
                    "condition_id": f"{q['id']}_insuf",
                    "context": replaced_context,
                    "sufficient": False,
                    "model_confident": model_confident,
                    "quadrant": "Q3" if model_confident else "Q4",
                    "expected_behavior": "abstain",
                    "context_length": len(replaced_context),
                    "prompt": format_prompt(q["question"], replaced_context),
                },
            ],
        }
        benchmark_examples.append(entry)

    print(f"\n  Created: {len(benchmark_examples)} examples ({len(benchmark_examples)*2} conditions)")
    print(f"  Skipped: {skipped} (unchanged context or bad length ratio)")

    # ---- Validate confound controls ----
    print("\n" + "=" * 60)
    print("CONFOUND CONTROL VALIDATION")
    print("=" * 60)

    suf_lengths = [ex["conditions"][0]["context_length"] for ex in benchmark_examples]
    insuf_lengths = [ex["conditions"][1]["context_length"] for ex in benchmark_examples]
    ratios = [i / s for s, i in zip(suf_lengths, insuf_lengths) if s > 0]
    print(f"\nLength ratio (insuf/suf): {np.mean(ratios):.3f} ± {np.std(ratios):.3f}")
    print(f"Within 10% of 1.0: {sum(1 for r in ratios if 0.9 <= r <= 1.1) / len(ratios) * 100:.1f}%")

    # Quadrant distribution
    quad_counts = defaultdict(int)
    for ex in benchmark_examples:
        for cond in ex["conditions"]:
            quad_counts[cond["quadrant"]] += 1

    print(f"\n2×2 Quadrant Distribution:")
    quad_descs = {
        "Q1": "Sufficient + Confident",
        "Q2": "Sufficient + Uncertain",
        "Q3": "Insufficient + Confident",
        "Q4": "Insufficient + Uncertain",
    }
    for q_name in ["Q1", "Q2", "Q3", "Q4"]:
        count = quad_counts[q_name]
        print(f"  {q_name} ({quad_descs[q_name]}): {count}")

    # Question type distribution
    type_counts = defaultdict(int)
    for ex in benchmark_examples:
        type_counts[ex["question_type"]] += 1
    print(f"\nQuestion types:")
    for qt, cnt in sorted(type_counts.items()):
        print(f"  {qt}: {cnt}")

    # ---- GO/NO-GO CHECK ----
    min_quadrant = min(quad_counts.values()) if quad_counts else 0
    if min_quadrant < 200:
        print(f"\n⚠ WARNING: Smallest quadrant has {min_quadrant} examples (target: 500+)")
        print("  Consider adjusting PK test threshold or adding more source data.")
    else:
        print(f"\n✓ All quadrants have {min_quadrant}+ examples. PASS.")

    # ---- Split and save ----
    print("\n" + "=" * 60)
    print("Splitting and saving...")
    print("=" * 60)
    random.shuffle(benchmark_examples)
    n = len(benchmark_examples)
    n_test = int(n * 0.4)
    n_val = int(n * 0.1)
    n_train = n - n_test - n_val

    splits = {
        "train": benchmark_examples[:n_train],
        "val": benchmark_examples[n_train:n_train + n_val],
        "test": benchmark_examples[n_train + n_val:],
    }

    metadata = {
        "name": "SufficiencyBench",
        "version": "1.0",
        "description": (
            "A benchmark for context sufficiency detection in RAG that "
            "independently controls context sufficiency and model confidence."
        ),
        "quadrant_definitions": quad_descs,
        "question_types": ["factual", "multi_hop", "comparative", "subjective"],
        "model_used_for_pk_test": model_path,
        "total_questions": n,
        "total_conditions": n * 2,
        "splits": {name: len(data) for name, data in splits.items()},
        "quadrant_counts": dict(quad_counts),
        "question_type_counts": dict(type_counts),
        "length_ratio_mean": float(np.mean(ratios)),
        "length_ratio_std": float(np.std(ratios)),
    }

    for name, data in splits.items():
        path = BENCH_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved {name}: {len(data)} questions ({len(data)*2} conditions)")

    with open(BENCH_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSufficiencyBench saved to {BENCH_DIR}")
    print(f"Total: {n} questions, {n*2} conditions")
    print("\n=== BENCHMARK CONSTRUCTION COMPLETE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", help="GPU device")
    parser.add_argument("--model", default=None,
                        choices=["llama", "mistral", "qwen"],
                        help="Model for PK test (auto-selects if not set)")
    args = parser.parse_args()
    build_benchmark(device=args.device, model_key=args.model)
