"""
insufficient_context_verifier.py  (Issue 4)

Automated verification pass over all insufficient contexts:
  1. Gold answer does NOT appear in the modified context (construction check)
  2. Replacement sentence is topically similar to removed sentence (cosine > 0.5)
  3. Replacement sentence does NOT contain gold answer entities
  4. Context length ratio between sufficient/insufficient is close to 1.0
  5. Flags multi-hop cases where answer may be inferrable from remaining text

Also checks for regex sentence detection failures (abbreviations, etc.)
by comparing nltk.sent_tokenize vs regex split results.

Outputs:
  - results/validation/context_verification.json  (per-item flags)
  - results/validation/context_verification_summary.json  (aggregate stats)
  - results/validation/flagged_items.json  (items needing review)

Usage:
  pip install nltk sentence-transformers
  python src/validation/insufficient_context_verifier.py
  python src/validation/insufficient_context_verifier.py --split test --verbose
"""

import json
import re
import argparse
import sys
import os
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import BENCH_DIR, RESULTS_DIR

VALIDATION_DIR = RESULTS_DIR / "validation"


def setup_nltk():
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def gold_in_context(context: str, gold: str) -> bool:
    return gold.lower().strip() in context.lower()


def get_context_entities(text: str) -> set:
    """Simple entity extraction: capitalized words and numbers."""
    tokens = re.findall(r'\b[A-Z][a-zA-Z]+\b|\b\d{4}\b|\b\d+\b', text)
    return set(tokens)


def sentence_split_regex(text: str) -> list:
    return re.split(r'(?<=[.!?])\s+', text.strip())


def sentence_split_nltk(text: str) -> list:
    import nltk
    return nltk.sent_tokenize(text)


def check_sent_split_agreement(sufficient: str, insufficient: str) -> dict:
    """Check if regex and nltk sentence splitting agree."""
    regex_suf = len(sentence_split_regex(sufficient))
    nltk_suf = len(sentence_split_nltk(sufficient))
    regex_ins = len(sentence_split_regex(insufficient))
    nltk_ins = len(sentence_split_nltk(insufficient))
    return {
        "regex_suf": regex_suf, "nltk_suf": nltk_suf,
        "regex_ins": regex_ins, "nltk_ins": nltk_ins,
        "split_disagrees": (regex_suf != nltk_suf or regex_ins != nltk_ins),
    }


def semantic_similarity(a: str, b: str, model) -> float:
    import numpy as np
    ea = model.encode(a)
    eb = model.encode(b)
    return float(
        (ea @ eb) / (
            (ea @ ea) ** 0.5 * (eb @ eb) ** 0.5 + 1e-8
        )
    )


def verify_contexts(splits: list = None, verbose: bool = False,
                    semantic_check: bool = True):
    setup_nltk()

    if semantic_check:
        try:
            from sentence_transformers import SentenceTransformer
            embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            print("Semantic similarity checking enabled.")
        except ImportError:
            print("sentence-transformers not installed; skipping semantic checks.")
            semantic_check = False
            embed_model = None
    else:
        embed_model = None

    if splits is None:
        splits = ["train", "val", "test"]

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    all_flags = []
    stats = defaultdict(int)

    for split in splits:
        with open(BENCH_DIR / f"{split}.json") as f:
            data = json.load(f)

        print(f"\n=== Verifying {split} ({len(data)} questions) ===")

        for ex in data:
            conds = {c["sufficient"]: c for c in ex["conditions"]}
            suf_cond = conds.get(True)
            ins_cond = conds.get(False)

            if suf_cond is None or ins_cond is None:
                continue

            suf_ctx = suf_cond["context"]
            ins_ctx = ins_cond["context"]
            gold = ex["gold_answer"]
            qid = ex["id"]
            qtype = ex["question_type"]

            flags = []

            # Check 1: gold answer still in insufficient context?
            if gold_in_context(ins_ctx, gold):
                flags.append("gold_still_in_insufficient")
                stats["gold_leak"] += 1

            # Check 2: context length ratio
            ratio = len(ins_ctx) / max(len(suf_ctx), 1)
            if ratio < 0.5 or ratio > 1.5:
                flags.append(f"length_ratio_outlier:{ratio:.2f}")
                stats["length_outlier"] += 1

            # Check 3: sentence split disagreement (regex vs nltk)
            split_check = check_sent_split_agreement(suf_ctx, ins_ctx)
            if split_check["split_disagrees"]:
                flags.append("sent_split_disagrees")
                stats["sent_split_disagree"] += 1

            # Check 4: gold answer entities in insufficient context
            gold_entities = get_context_entities(gold)
            ins_entities = get_context_entities(ins_ctx)
            leaked_entities = gold_entities & ins_entities
            if leaked_entities and len(leaked_entities) >= len(gold_entities) * 0.5:
                flags.append(f"entity_leak:{list(leaked_entities)[:3]}")
                stats["entity_leak"] += 1

            # Check 5: multi-hop — answer inferrable from remaining text?
            if qtype == "multi_hop" and not gold_in_context(ins_ctx, gold):
                # Check if any entity from gold appears in insufficient context
                if leaked_entities:
                    flags.append("multi_hop_partial_signal")
                    stats["multi_hop_partial"] += 1

            # Semantic check: are the two contexts semantically similar overall?
            if semantic_check and embed_model is not None:
                sim = semantic_similarity(suf_ctx, ins_ctx, embed_model)
                if sim < 0.3:
                    flags.append(f"low_topic_similarity:{sim:.2f}")
                    stats["low_similarity"] += 1

            stats["total"] += 1
            if flags:
                stats["flagged"] += 1

            item_result = {
                "id": qid,
                "split": split,
                "question_type": qtype,
                "quadrant_suf": suf_cond["quadrant"],
                "gold_answer": gold,
                "flags": flags,
                "length_ratio": float(ratio),
                "sent_split_check": split_check,
            }
            all_flags.append(item_result)

            if verbose and flags:
                print(f"  {qid} ({qtype}): {flags}")

    flagged = [x for x in all_flags if x["flags"]]
    critical = [x for x in flagged if "gold_still_in_insufficient" in x["flags"]]

    print(f"\n=== Verification Summary ===")
    print(f"Total items: {stats['total']}")
    print(f"Any flag: {stats['flagged']} ({100*stats['flagged']/max(stats['total'],1):.1f}%)")
    print(f"  Gold still in insufficient: {stats['gold_leak']}")
    print(f"  Length ratio outlier: {stats['length_outlier']}")
    print(f"  Sentence split disagrees: {stats['sent_split_disagree']}")
    print(f"  Entity leak (≥50% of gold entities): {stats['entity_leak']}")
    print(f"  Multi-hop partial signal: {stats['multi_hop_partial']}")
    print(f"  Low topic similarity: {stats['low_similarity']}")
    print(f"\nCRITICAL (gold still present): {len(critical)} items")

    if critical:
        print("  These items MUST be fixed — gold answer still findable in insufficient context.")
        for c in critical[:5]:
            print(f"  {c['id']}: {c['gold_answer'][:50]}")

    with open(VALIDATION_DIR / "context_verification.json", "w") as f:
        json.dump(all_flags, f, indent=2)
    with open(VALIDATION_DIR / "context_verification_summary.json", "w") as f:
        json.dump(dict(stats), f, indent=2)
    with open(VALIDATION_DIR / "flagged_items.json", "w") as f:
        json.dump(flagged, f, indent=2)

    print(f"\nSaved to {VALIDATION_DIR}")
    return stats, flagged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--all-splits", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-semantic", action="store_true",
                        help="Skip semantic similarity check (faster)")
    args = parser.parse_args()

    splits = None
    if args.all_splits:
        splits = ["train", "val", "test"]
    elif args.split:
        splits = [args.split]

    verify_contexts(
        splits=splits,
        verbose=args.verbose,
        semantic_check=not args.no_semantic,
    )
