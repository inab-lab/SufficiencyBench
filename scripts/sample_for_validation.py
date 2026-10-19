"""
sample_for_validation.py

Samples 200 examples from SufficiencyBench test set for human validation.
Outputs a JSON file with question, context, gold_answer, and label (sufficient/insufficient)
for manual annotation.

Usage:
  python scripts/sample_for_validation.py
"""

import json
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.paths import BENCH_DIR, RESULTS_DIR


def sample_for_validation(n_per_class=100, seed=42):
    random.seed(seed)

    with open(BENCH_DIR / "test.json") as f:
        test_data = json.load(f)

    sufficient = []
    insufficient = []

    for ex in test_data:
        for cond in ex["conditions"]:
            item = {
                "id": cond["condition_id"],
                "question": ex["question"],
                "question_type": ex["question_type"],
                "context": cond["context"],
                "gold_answer": ex["gold_answer"],
                "label_sufficient": cond["sufficient"],
                "quadrant": cond["quadrant"],
                # Annotation fields (to be filled)
                "human_label": None,
                "human_notes": "",
            }
            if cond["sufficient"]:
                sufficient.append(item)
            else:
                insufficient.append(item)

    print(f"Total sufficient: {len(sufficient)}, insufficient: {len(insufficient)}")

    sampled_suf = random.sample(sufficient, min(n_per_class, len(sufficient)))
    sampled_insuf = random.sample(insufficient, min(n_per_class, len(insufficient)))

    all_sampled = sampled_suf + sampled_insuf
    random.shuffle(all_sampled)

    # Remove label for blind annotation
    for item in all_sampled:
        item["_hidden_label"] = item.pop("label_sufficient")

    out_dir = RESULTS_DIR / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full file for annotation
    annotation_path = out_dir / "validation_samples.json"
    with open(annotation_path, "w") as f:
        json.dump(all_sampled, f, indent=2, ensure_ascii=False)

    # Key file (labels only, for scoring)
    key = [{"id": item["id"], "label": item["_hidden_label"]} for item in all_sampled]
    key_path = out_dir / "validation_key.json"
    with open(key_path, "w") as f:
        json.dump(key, f, indent=2)

    print(f"\nSampled {len(all_sampled)} examples:")
    print(f"  Sufficient: {len(sampled_suf)}")
    print(f"  Insufficient: {len(sampled_insuf)}")
    print(f"  Annotation file: {annotation_path}")
    print(f"  Answer key: {key_path}")

    # Also create a quick-annotation version with shorter contexts
    quick = []
    for item in all_sampled[:20]:  # first 20 for quick check
        quick.append({
            "id": item["id"],
            "question": item["question"],
            "context_preview": item["context"][:300] + "..." if len(item["context"]) > 300 else item["context"],
            "gold_answer": item["gold_answer"],
        })

    quick_path = out_dir / "validation_quick_sample.json"
    with open(quick_path, "w") as f:
        json.dump(quick, f, indent=2, ensure_ascii=False)
    print(f"  Quick sample (20): {quick_path}")

    return all_sampled


if __name__ == "__main__":
    sample_for_validation()
