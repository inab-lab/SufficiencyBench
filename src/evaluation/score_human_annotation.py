"""
score_human_annotation.py — Score the 30-example human annotation against ground truth.

Usage (after filling human_annotation_30.csv):
  python src/evaluation/score_human_annotation.py

Expects 'Is_Sufficient' column values: YES or NO (case-insensitive).
Saves results/validation/human_annotation_30_results.json.
"""

import csv
import json
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

VAL_DIR    = Path(__file__).resolve().parents[2] / "results" / "validation"
ANN_FILE   = VAL_DIR / "human_annotation_30.csv"
KEY_FILE   = VAL_DIR / "human_annotation_30_key.csv"
OUT_FILE   = VAL_DIR / "human_annotation_30_results.json"


def main():
    with open(KEY_FILE) as f:
        key = {r["id"]: r["_truth"] for r in csv.DictReader(f)}

    rows = list(csv.DictReader(open(ANN_FILE)))
    filled = [r for r in rows if r["Is_Sufficient"].strip()]
    missing = [r["id"] for r in rows if not r["Is_Sufficient"].strip()]

    if missing:
        print(f"WARNING: {len(missing)} rows without a label: {missing}")
    if not filled:
        print("No rows filled yet. Please open human_annotation_30.csv and fill 'Is_Sufficient'.")
        return

    human   = [r["Is_Sufficient"].strip().upper() for r in filled]
    ground  = [key[r["id"]] for r in filled]
    qtypes  = [r["question_type"] for r in filled]
    ids     = [r["id"] for r in filled]

    # Convert to binary
    h_bin = [1 if v == "YES" else 0 for v in human]
    g_bin = [1 if v == "YES" else 0 for v in ground]

    agreement   = sum(h == g for h, g in zip(h_bin, g_bin)) / len(h_bin)
    kappa       = float(cohen_kappa_score(g_bin, h_bin))

    per_type = {}
    for qt in ["factual", "multi_hop", "comparative", "subjective"]:
        idxs = [i for i, q in enumerate(qtypes) if q == qt]
        if idxs:
            ag = sum(h_bin[i] == g_bin[i] for i in idxs) / len(idxs)
            per_type[qt] = {"n": len(idxs), "agreement": round(ag, 4)}

    disagreements = [
        {"id": ids[i], "question_type": qtypes[i],
         "human": human[i], "ground_truth": ground[i],
         "notes": filled[i].get("Notes", "")}
        for i in range(len(filled)) if h_bin[i] != g_bin[i]
    ]

    results = {
        "n_annotated": len(filled),
        "n_total": len(rows),
        "overall_agreement": round(agreement, 4),
        "cohen_kappa": round(kappa, 4),
        "per_type": per_type,
        "n_disagreements": len(disagreements),
        "disagreements": disagreements,
    }

    print(f"Annotated: {len(filled)}/{len(rows)}")
    print(f"Agreement: {agreement:.4f}")
    print(f"Kappa:     {kappa:.4f}")
    print("Per type:")
    for qt, v in per_type.items():
        print(f"  {qt:15s}: {v['agreement']:.4f} (n={v['n']})")
    if disagreements:
        print(f"\nDisagreements ({len(disagreements)}):")
        for d in disagreements:
            print(f"  [{d['question_type']}] {d['id']}: human={d['human']}, truth={d['ground_truth']}")

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()
