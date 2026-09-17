#!/usr/bin/env python3
"""
Score Google Forms annotation responses.

Usage:
    1. Export Google Form responses as CSV (Responses tab → three dots → Download .csv)
    2. Run: python scripts/score_google_form.py results/validation/responses.csv

The script expects:
    - Column 1: Timestamp
    - Column 2: Annotator name/initials
    - Columns 3-35: YES/NO responses for each of the 33 items (in form order)
"""

import sys
import csv
import json
import numpy as np
from pathlib import Path
from collections import Counter

def load_key(key_path="results/validation/google_form_key.json"):
    with open(key_path) as f:
        return json.load(f)

def parse_responses(csv_path):
    """Parse Google Forms CSV export."""
    responses = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            if len(row) < 3:
                continue
            name = row[1].strip()
            # Answers start at column 2 (after Timestamp and Name)
            answers = []
            for cell in row[2:2+33]:
                val = cell.strip().upper()
                if val in ('YES', 'SÍ', 'SI'):
                    answers.append('YES')
                elif val in ('NO'):
                    answers.append('NO')
                else:
                    answers.append(None)
            responses.append({'name': name, 'answers': answers})

    return responses

def compute_agreement_with_key(responses, key):
    """Compute each annotator's agreement with ground truth."""
    results = []
    real_items = [(i, k) for i, k in enumerate(key) if not k['is_attention_check']]
    attn_items = [(i, k) for i, k in enumerate(key) if k['is_attention_check']]

    for resp in responses:
        # Real questions
        correct = 0
        total = 0
        suf_correct = 0
        suf_total = 0
        insuf_correct = 0
        insuf_total = 0

        for idx, k in real_items:
            if idx < len(resp['answers']) and resp['answers'][idx] is not None:
                total += 1
                if resp['answers'][idx] == k['ground_truth']:
                    correct += 1
                if k['ground_truth'] == 'YES':
                    suf_total += 1
                    if resp['answers'][idx] == 'YES':
                        suf_correct += 1
                else:
                    insuf_total += 1
                    if resp['answers'][idx] == 'NO':
                        insuf_correct += 1

        # Attention checks
        attn_pass = 0
        attn_total = 0
        for idx, k in attn_items:
            if idx < len(resp['answers']) and resp['answers'][idx] is not None:
                attn_total += 1
                if resp['answers'][idx] == k['ground_truth']:
                    attn_pass += 1

        results.append({
            'name': resp['name'],
            'agreement': correct / total if total > 0 else 0,
            'correct': correct,
            'total': total,
            'suf_agreement': suf_correct / suf_total if suf_total > 0 else 0,
            'insuf_agreement': insuf_correct / insuf_total if insuf_total > 0 else 0,
            'attention_checks': f"{attn_pass}/{attn_total}",
            'attention_pass': attn_pass == attn_total
        })

    return results

def compute_fleiss_kappa(responses, key):
    """Compute Fleiss' kappa on real (non-attention-check) items."""
    real_indices = [i for i, k in enumerate(key) if not k['is_attention_check']]
    n_items = len(real_indices)
    n_raters = len(responses)
    n_categories = 2  # YES, NO

    # Build rating matrix: n_items × n_categories
    # Each cell = number of raters who assigned that category to that item
    matrix = np.zeros((n_items, n_categories))

    for item_idx, real_idx in enumerate(real_indices):
        for resp in responses:
            if real_idx < len(resp['answers']) and resp['answers'][real_idx] is not None:
                if resp['answers'][real_idx] == 'YES':
                    matrix[item_idx, 0] += 1
                else:
                    matrix[item_idx, 1] += 1

    # Fleiss' kappa computation
    N = n_items
    n = n_raters
    k = n_categories

    # Proportion of assignments to each category
    p_j = matrix.sum(axis=0) / (N * n)

    # Per-item agreement
    P_i = (np.sum(matrix ** 2, axis=1) - n) / (n * (n - 1)) if n > 1 else np.zeros(N)

    P_bar = np.mean(P_i)
    P_e = np.sum(p_j ** 2)

    if P_e == 1.0:
        kappa = 1.0
    else:
        kappa = (P_bar - P_e) / (1 - P_e)

    return kappa, P_bar, P_e

def compute_krippendorff_alpha(responses, key):
    """Compute Krippendorff's alpha (nominal) on real items."""
    real_indices = [i for i, k in enumerate(key) if not k['is_attention_check']]

    # Build reliability matrix: n_raters × n_items
    # Values: 0=YES, 1=NO, nan=missing
    n_raters = len(responses)
    n_items = len(real_indices)

    matrix = np.full((n_raters, n_items), np.nan)
    for r, resp in enumerate(responses):
        for item_idx, real_idx in enumerate(real_indices):
            if real_idx < len(resp['answers']) and resp['answers'][real_idx] is not None:
                matrix[r, item_idx] = 0 if resp['answers'][real_idx] == 'YES' else 1

    # Compute alpha
    # Count observed disagreements
    n_values = 2
    coincidence = np.zeros((n_values, n_values))

    for u in range(n_items):
        col = matrix[:, u]
        valid = col[~np.isnan(col)].astype(int)
        m_u = len(valid)
        if m_u < 2:
            continue
        for i in range(len(valid)):
            for j in range(len(valid)):
                if i != j:
                    coincidence[valid[i], valid[j]] += 1 / (m_u - 1)

    n_total = coincidence.sum()
    if n_total == 0:
        return float('nan')

    # Expected disagreement
    n_c = coincidence.sum(axis=1)
    D_o = 1 - np.trace(coincidence) / n_total
    D_e = 1 - np.sum(n_c * (n_c - 1)) / (n_total * (n_total - 1)) if n_total > 1 else 0

    if D_e == 0:
        return 1.0

    alpha = 1 - D_o / D_e
    return alpha

def main():
    if len(sys.argv) < 2:
        print("Usage: python score_google_form.py <responses.csv>")
        print("       Download CSV from Google Forms → Responses → ⋮ → Download (.csv)")
        sys.exit(1)

    csv_path = sys.argv[1]
    key = load_key()
    responses = parse_responses(csv_path)

    print(f"\n{'='*60}")
    print(f"ANNOTATION RESULTS")
    print(f"{'='*60}")
    print(f"Responses: {len(responses)} annotators")
    print(f"Items: 30 real + 3 attention checks = 33 total")

    # Per-annotator results
    annotator_results = compute_agreement_with_key(responses, key)

    print(f"\n{'─'*60}")
    print(f"PER-ANNOTATOR AGREEMENT WITH BENCHMARK LABELS")
    print(f"{'─'*60}")
    print(f"{'Name':<20} {'Overall':>8} {'Sufficient':>11} {'Insufficient':>13} {'Attn':>6} {'OK':>4}")
    print(f"{'─'*60}")

    flagged = []
    for r in annotator_results:
        ok = "✓" if r['attention_pass'] else "✗"
        if not r['attention_pass']:
            flagged.append(r['name'])
        print(f"{r['name']:<20} {r['agreement']:>7.1%} {r['suf_agreement']:>10.1%} {r['insuf_agreement']:>12.1%} {r['attention_checks']:>6} {ok:>4}")

    if flagged:
        print(f"\n⚠ Flagged annotators (failed attention checks): {', '.join(flagged)}")
        print(f"  Consider excluding them from IAA computation.")

    # Overall agreement
    agreements = [r['agreement'] for r in annotator_results if r['attention_pass']]
    suf_agreements = [r['suf_agreement'] for r in annotator_results if r['attention_pass']]
    insuf_agreements = [r['insuf_agreement'] for r in annotator_results if r['attention_pass']]

    print(f"\n{'─'*60}")
    print(f"AGGREGATE (excluding flagged annotators)")
    print(f"{'─'*60}")
    print(f"Mean agreement with labels:  {np.mean(agreements):.1%} (std: {np.std(agreements):.1%})")
    print(f"  Sufficient items:          {np.mean(suf_agreements):.1%}")
    print(f"  Insufficient items:        {np.mean(insuf_agreements):.1%}")

    # Inter-annotator agreement
    valid_responses = [r for r, ar in zip(responses, annotator_results) if ar['attention_pass']]

    if len(valid_responses) >= 2:
        kappa, P_bar, P_e = compute_fleiss_kappa(valid_responses, key)
        alpha = compute_krippendorff_alpha(valid_responses, key)

        print(f"\n{'─'*60}")
        print(f"INTER-ANNOTATOR AGREEMENT")
        print(f"{'─'*60}")
        print(f"Fleiss' kappa:          {kappa:.3f}")
        print(f"Krippendorff's alpha:   {alpha:.3f}")
        print(f"Observed agreement:     {P_bar:.3f}")
        print(f"Expected agreement:     {P_e:.3f}")
        print(f"Number of raters:       {len(valid_responses)}")

        # Interpretation
        if kappa >= 0.81:
            interp = "Almost perfect agreement"
        elif kappa >= 0.61:
            interp = "Substantial agreement"
        elif kappa >= 0.41:
            interp = "Moderate agreement"
        elif kappa >= 0.21:
            interp = "Fair agreement"
        else:
            interp = "Slight/poor agreement"
        print(f"Interpretation:         {interp} (Landis & Koch)")

    # Paper-ready paragraph
    print(f"\n{'─'*60}")
    print(f"PAPER-READY PARAGRAPH")
    print(f"{'─'*60}")
    n_valid = len(valid_responses)
    mean_agr = np.mean(agreements) * 100
    suf_agr = np.mean(suf_agreements) * 100
    insuf_agr = np.mean(insuf_agreements) * 100

    if len(valid_responses) >= 2:
        print(f'"To validate benchmark labels, {n_valid} annotators independently judged')
        print(f'30 question-context pairs for information sufficiency without seeing')
        print(f'the gold answer. Mean agreement with benchmark labels was {mean_agr:.0f}%.')
        print(f'Fleiss\' kappa was {kappa:.2f} ({interp.lower()}), with Krippendorff\'s')
        print(f'alpha = {alpha:.2f}. Agreement was higher for sufficient contexts ({suf_agr:.0f}%)')
        print(f'than for insufficient contexts ({insuf_agr:.0f}%)."')

    # Save results as JSON
    output = {
        'n_annotators': len(responses),
        'n_valid_annotators': len(valid_responses),
        'n_items': 30,
        'n_attention_checks': 3,
        'mean_agreement': float(np.mean(agreements)),
        'std_agreement': float(np.std(agreements)),
        'suf_agreement': float(np.mean(suf_agreements)),
        'insuf_agreement': float(np.mean(insuf_agreements)),
        'per_annotator': annotator_results,
    }
    if len(valid_responses) >= 2:
        output['fleiss_kappa'] = float(kappa)
        output['krippendorff_alpha'] = float(alpha)

    out_path = Path(csv_path).parent / 'annotation_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()
