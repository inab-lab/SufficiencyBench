"""
pk_gpt4o_judge.py  (Issue 3)

Replaces case-insensitive substring match with a GPT-4o-mini semantic
judge for parametric knowledge (PK) assessment.

For each question, judges whether the model's closed-book response
correctly answers it — handling paraphrases and equivalent formulations.

Special handling for ELI5: marks all ELI5 questions as pk=False
(open-ended explanations have no well-defined gold answer for PK testing).

Outputs:
  - data/experiments/sufficiency_bench/pk_cache_gpt4o.json
  - results/validation/pk_agreement_analysis.json  (new vs old labels)

Then rewrites train/val/test.json with updated model_knows_answer fields.

Usage:
  export OPENAI_API_KEY=sk-...
  python src/validation/pk_gpt4o_judge.py --dry-run  # preview 20 cases
  python src/validation/pk_gpt4o_judge.py --split train
  python src/validation/pk_gpt4o_judge.py --all-splits  # runs all 3 splits
"""

import json
import argparse
import sys
import os
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import BENCH_DIR, RESULTS_DIR

VALIDATION_DIR = RESULTS_DIR / "validation"

PK_JUDGE_PROMPT = """\
A language model was asked to answer a question without any reference material (from memory only).

Question: {question}
Gold answer: {gold_answer}
Model's response: {model_response}

Did the model correctly answer the question?
- Answer YES if the model's response contains the correct answer, even if worded differently.
- Answer NO if the model's response is wrong, incomplete, or the model said it doesn't know.
- Answer YES for partial matches only if the key fact is present and correct.

Respond with exactly one word: YES or NO"""


def call_gpt4o_mini(client, prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip().upper()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  API error: {e}")
                return "ERROR"


def substring_pk(response: str, gold: str) -> bool:
    return gold.lower().strip() in response.lower().strip()


def load_split(split: str) -> list:
    with open(BENCH_DIR / f"{split}.json") as f:
        return json.load(f)


def run_pk_judge(splits: list = None, dry_run: bool = False):
    try:
        from openai import OpenAI
    except ImportError:
        print("pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    if splits is None:
        splits = ["train", "val", "test"]

    # Load existing PK cache for Mistral (the reference model)
    cache_path = BENCH_DIR / "pk_cache_mistral.json"
    with open(cache_path) as f:
        mistral_cache = json.load(f)

    all_results = {}
    agreement_stats = {
        "total": 0, "agree": 0, "new_true_old_false": 0,
        "new_false_old_true": 0, "eli5_excluded": 0, "errors": 0
    }

    for split in splits:
        data = load_split(split)
        print(f"\n=== Processing {split} ({len(data)} questions) ===")

        if dry_run:
            data = data[:20]
            print("  DRY RUN: processing 20 items only")

        updated = []
        for i, ex in enumerate(data):
            qid = ex["id"]
            question = ex["question"]
            gold = ex["gold_answer"]
            qtype = ex["question_type"]
            old_pk = ex["model_knows_answer"]

            # ELI5: always mark as not-knows (open-ended, no definable PK)
            if qtype == "subjective":
                new_pk = False
                verdict = "EXCLUDED_ELI5"
                agreement_stats["eli5_excluded"] += 1
            else:
                # Get stored closed-book response for Mistral
                cb_response = mistral_cache.get(qid, {}).get("response", "")
                if not cb_response:
                    # Fallback to old substring result
                    new_pk = old_pk
                    verdict = "NO_RESPONSE"
                else:
                    prompt = PK_JUDGE_PROMPT.format(
                        question=question,
                        gold_answer=gold,
                        model_response=cb_response,
                    )
                    verdict = call_gpt4o_mini(client, prompt)
                    if verdict == "ERROR":
                        new_pk = old_pk  # fallback
                        agreement_stats["errors"] += 1
                    else:
                        new_pk = (verdict == "YES")

                    time.sleep(0.05)

            # Track agreement
            agreement_stats["total"] += 1
            if new_pk == old_pk:
                agreement_stats["agree"] += 1
            elif new_pk and not old_pk:
                agreement_stats["new_true_old_false"] += 1
            else:
                agreement_stats["new_false_old_true"] += 1

            all_results[f"{split}_{qid}"] = {
                "qid": qid,
                "split": split,
                "question_type": qtype,
                "old_pk": old_pk,
                "new_pk": new_pk,
                "verdict": verdict,
                "old_method": "substring",
                "new_method": "gpt4o-mini",
            }

            updated.append({**ex, "model_knows_answer_gpt4o": new_pk,
                            "pk_verdict_gpt4o": verdict})

            if (i + 1) % 100 == 0:
                agree_rate = agreement_stats["agree"] / max(agreement_stats["total"], 1)
                print(f"  [{i+1}/{len(data)}] Agreement so far: {agree_rate:.3f}")

        if not dry_run:
            # Write updated split (keeps original model_knows_answer, adds new field)
            out_path = BENCH_DIR / f"{split}_with_gpt4o_pk.json"
            with open(out_path, "w") as f:
                json.dump(updated, f, indent=2)
            print(f"  Saved updated {split} to {out_path}")

    # Agreement summary
    total = agreement_stats["total"]
    agree_rate = agreement_stats["agree"] / total if total > 0 else 0

    print(f"\n=== PK Agreement Summary ===")
    print(f"Total assessed: {total}")
    print(f"Agreement: {agreement_stats['agree']}/{total} = {agree_rate:.4f}")
    print(f"New=True, Old=False (false negatives in old): {agreement_stats['new_true_old_false']}")
    print(f"New=False, Old=True (false positives in old): {agreement_stats['new_false_old_true']}")
    print(f"ELI5 excluded from PK: {agreement_stats['eli5_excluded']}")
    print(f"Errors (fell back to old): {agreement_stats['errors']}")

    # Spot-check: print 10 disagreements for manual review
    disagreements = [v for v in all_results.values()
                     if v["new_pk"] != v["old_pk"]]
    print(f"\n=== Sample disagreements (up to 10 of {len(disagreements)}) ===")
    for v in disagreements[:10]:
        print(f"  {v['split']}/{v['qid']} [{v['question_type']}]: "
              f"old={v['old_pk']} -> new={v['new_pk']} (verdict={v['verdict']})")

    # Persist a verdict cache (per-qid) and the agreement analysis.
    if not dry_run:
        cache_path = BENCH_DIR / "pk_cache_gpt4o.json"
        with open(cache_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved GPT-4o PK cache to {cache_path}")

    analysis = {
        "agreement_stats": agreement_stats,
        "agreement_rate": agree_rate,
        "n_disagreements": len(disagreements),
        "splits": splits,
        "old_method": "substring",
        "new_method": "gpt4o-mini",
        "results": all_results,
    }
    analysis_path = VALIDATION_DIR / "pk_agreement_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved PK agreement analysis to {analysis_path}")

    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=None,
                        choices=["train", "val", "test"],
                        help="Process a single split")
    parser.add_argument("--all-splits", action="store_true",
                        help="Process train, val and test")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview on the first 20 items per split (no writes)")
    args = parser.parse_args()

    if args.all_splits:
        splits = ["train", "val", "test"]
    elif args.split:
        splits = [args.split]
    else:
        splits = ["train", "val", "test"]

    run_pk_judge(splits=splits, dry_run=args.dry_run)