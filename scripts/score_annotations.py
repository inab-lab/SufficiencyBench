"""Score human annotation results from Google Form CSV."""
import numpy as np
from pathlib import Path

CSV_PATH = Path("/home/inab/Downloads/Formulario sin título.csv")

# Ground truth (33 items) from GOOGLE_FORM_CONTENT.txt
GROUND_TRUTH = [
    "Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes",
    "No","Yes","No","No","Yes","Yes","No","No","No","Yes",
    "No","No","No","No","Yes","No","No","Yes","No","No",
    "No","No","Yes",
]
ATTENTION_CHECKS = [7, 17, 27]  # 0-indexed: Q8, Q18, Q28

def parse_csv():
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    responses = []
    for line in content.strip().split('\n'):
        if line.startswith('"2026/'):
            parts = line.split('","')
            answers = [p.strip().strip('"') for p in parts[2:]][:33]
            level = parts[1].strip().strip('"')
            responses.append({'level': level or 'C1', 'answers': answers})
    return responses

def fleiss_kappa(ratings):
    n_items, n_raters = len(ratings), len(ratings[0])
    counts = np.zeros((n_items, 2))
    for i in range(n_items):
        for j in range(n_raters):
            counts[i, int(ratings[i][j])] += 1
    P_i = (np.sum(counts**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_j = np.sum(counts, axis=0) / (n_items * n_raters)
    P_bar, P_e = np.mean(P_i), np.sum(p_j**2)
    return (P_bar - P_e) / (1.0 - P_e) if P_e < 1.0 else 1.0

def main():
    responses = parse_csv()
    n_ann = len(responses)
    n_items = len(GROUND_TRUTH)
    binary_gt = [1 if g == "Yes" else 0 for g in GROUND_TRUTH]
    binary_resp = [[1 if a.strip().lower() == 'yes' else 0 for a in r['answers']] for r in responses]

    print(f"={'='*70}\nHUMAN ANNOTATION RESULTS ({n_ann} annotators)\n{'='*70}")
    print(f"\nEnglish levels: {[r['level'] for r in responses]}")

    # Attention checks
    print(f"\n--- Attention Checks ---")
    flagged = []
    for i, r in enumerate(responses):
        passed = sum(1 for ac in ATTENTION_CHECKS if r['answers'][ac] == GROUND_TRUTH[ac])
        print(f"  Annotator {i+1}: {'PASS' if passed==3 else f'FAIL ({passed}/3)'}")
        if passed < 3: flagged.append(i)

    # Per-annotator agreement
    print(f"\n--- Per-Annotator Agreement ---")
    agrs = []
    for i, br in enumerate(binary_resp):
        agree = sum(1 for j in range(n_items) if br[j] == binary_gt[j])
        pct = agree / n_items * 100
        agrs.append(pct)
        print(f"  Annotator {i+1}: {agree}/{n_items} = {pct:.1f}%{' [FLAGGED]' if i in flagged else ''}")
    print(f"\n  Mean: {np.mean(agrs):.1f}% +/- {np.std(agrs):.1f}%")

    # Majority vote
    print(f"\n--- Per-Item Analysis ---")
    maj_correct = 0
    for i in range(n_items):
        votes = [br[i] for br in binary_resp]
        n_yes, n_no = sum(votes), len(votes) - sum(votes)
        majority = 1 if n_yes > n_no else 0
        correct = majority == binary_gt[i]
        if correct: maj_correct += 1
        agr_pct = max(n_yes, n_no) / len(votes) * 100
        if not correct or agr_pct < 70:
            gt_l = "YES" if binary_gt[i] else "NO"
            mj_l = "YES" if majority else "NO"
            ac = " [AC]" if i in ATTENTION_CHECKS else ""
            print(f"  Q{i+1}: GT={gt_l} Maj={mj_l} ({n_yes}Y/{n_no}N={agr_pct:.0f}%) {'OK' if correct else 'WRONG'}{ac}")

    suf_idx = [i for i in range(n_items) if binary_gt[i] == 1]
    insuf_idx = [i for i in range(n_items) if binary_gt[i] == 0]
    suf_ok = sum(1 for i in suf_idx if sum(br[i] for br in binary_resp) > n_ann/2)
    insuf_ok = sum(1 for i in insuf_idx if sum(br[i] for br in binary_resp) <= n_ann/2)

    print(f"\n  Majority accuracy: {maj_correct}/{n_items} = {maj_correct/n_items*100:.1f}%")
    print(f"  Sufficient: {suf_ok}/{len(suf_idx)} = {suf_ok/len(suf_idx)*100:.1f}%")
    print(f"  Insufficient: {insuf_ok}/{len(insuf_idx)} = {insuf_ok/len(insuf_idx)*100:.1f}%")

    # Fleiss' kappa
    ratings = [[br[i] for br in binary_resp] for i in range(n_items)]
    kappa = fleiss_kappa(ratings)
    non_ac = [i for i in range(n_items) if i not in ATTENTION_CHECKS]
    kappa_no_ac = fleiss_kappa([ratings[i] for i in non_ac])

    if kappa >= 0.81: interp = "almost perfect"
    elif kappa >= 0.61: interp = "substantial"
    elif kappa >= 0.41: interp = "moderate"
    elif kappa >= 0.21: interp = "fair"
    else: interp = "slight"

    print(f"\n--- Fleiss' kappa ---")
    print(f"  All items: k = {kappa:.3f} ({interp})")
    print(f"  Excl. attention checks: k = {kappa_no_ac:.3f}")

    print(f"\n{'='*70}\nVALUES FOR PAPER\n{'='*70}")
    print(f"N = {n_ann}")
    print(f"Mean agreement = {np.mean(agrs):.1f}%")
    print(f"Fleiss' kappa = {kappa:.3f}")
    print(f"Attention pass rate = {n_ann-len(flagged)}/{n_ann}")
    print(f"Sufficient = {suf_ok/len(suf_idx)*100:.1f}%")
    print(f"Insufficient = {insuf_ok/len(insuf_idx)*100:.1f}%")

if __name__ == "__main__":
    main()
