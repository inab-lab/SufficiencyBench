"""Splice-augmented Standard / CSP probes (Llama): train on original train pairs + spliced-sufficient train contexts
(label 1), same layer / classifier as the paper's probes, threshold from original val; evaluate on test."""
import json, sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "data"))
from evaluation.splice_controls import SPLICE_HS_DIR, SPLICE_DIR, RESULTS_DIR, qid_of, auc, cluster_boot, bal_acc_threshold, load_split
from methods.deco import make_classifier
M = "llama"; B = 1000; SRC = ["squad", "hotpot", "eli5", "all"]
best = json.load(open(RESULTS_DIR / M / "all_results_logreg.json"))["best_layers"]; Ls, Lc = best["standard"], best["DECO"]
tr = load_split(M, "train"); va = load_split(M, "val"); te = load_split(M, "test")

def spliced(split, ref):
    d = SPLICE_HS_DIR / M / split
    H = np.asarray(np.load(d / "h_with_context.npy"), dtype=np.float32); ids = json.load(open(d / "ids.json"))
    pos = {i: k for k, i in enumerate(ref[3])}; q = [qid_of(i) for i in ids]
    return H, np.array([pos[x + "_suf"] for x in q]), q
Hs_tr, isuf_tr, _ = spliced("train", tr)
feat = {"standard": (lambda hqc, hq: np.asarray(hqc[:, Ls]), Ls), "csp": (lambda hqc, hq: np.asarray(hqc[:, Lc] - hq[:, Lc]), Lc)}
pos = {i: k for k, i in enumerate(te[3])}
Hs_te = np.asarray(np.load(SPLICE_HS_DIR / M / "test/h_with_context.npy"), dtype=np.float32); ids_te = json.load(open(SPLICE_HS_DIR / M / "test/ids.json"))
qs = np.array([qid_of(i) for i in ids_te]); ssrc = np.array([i.split("_")[0] for i in qs])
i_suf = np.array([pos[x + "_suf"] for x in qs]); i_ins = np.array([pos[x + "_insuf"] for x in qs])
y = te[2]; q_all = np.array([qid_of(i) for i in te[3]])
R = {"n_train_spliced": int(len(Hs_tr)), "layers": {"standard": Ls, "csp": Lc}}
for name, (fx, L) in feat.items():
    X0 = fx(tr[0], tr[1]); Xs = fx(Hs_tr, tr[1][isuf_tr])
    base = make_classifier("logreg").fit(X0, tr[2])
    aug = make_classifier("logreg").fit(np.vstack([X0, Xs]), np.r_[tr[2], np.ones(len(Xs), dtype=tr[2].dtype)])
    for tag, pr in [("paper", base), ("augmented", aug)]:
        sc = lambda hqc, hq: pr.predict_proba(fx(hqc, hq))[:, 1]
        t = bal_acc_threshold(va[2], sc(va[0], va[1]))
        s_all = sc(te[0], te[1]); p_suf = s_all[i_suf]; p_ins = s_all[i_ins]; p_spl = sc(Hs_te, te[1][i_suf])
        res = {"layer": int(L), "threshold": t, "auroc_original_pairs": {"value": auc(y, s_all), "ci95": cluster_boot(lambda ix: auc(y[ix], s_all[ix]), q_all, B)}, "controlB": {}}
        for s in SRC:
            ix0 = np.where(ssrc == s)[0] if s != "all" else np.arange(len(qs)); g = qs[ix0]
            a2 = lambda pp, pn, ix: auc(np.r_[np.ones(len(ix)), np.zeros(len(ix))], np.r_[pp[ix0][ix], pn[ix0][ix]])
            mm = lambda fn: {"value": fn(np.arange(len(ix0))), "ci95": cluster_boot(fn, g, B)}
            res["controlB"][s] = {"n": int(len(ix0)),
                "mean_p_orig_suf": mm(lambda ix: float(p_suf[ix0][ix].mean())), "mean_p_spliced_suf": mm(lambda ix: float(p_spl[ix0][ix].mean())),
                "mean_p_insuf": mm(lambda ix: float(p_ins[ix0][ix].mean())),
                "auroc_orig_vs_spliced": mm(lambda ix: a2(p_suf, p_spl, ix)), "auroc_spliced_vs_insuf": mm(lambda ix: a2(p_spl, p_ins, ix)),
                "auroc_orig_vs_insuf": mm(lambda ix: a2(p_suf, p_ins, ix)),
                "frac_spliced_below_thr": mm(lambda ix: float((p_spl[ix0][ix] < t).mean())), "frac_orig_below_thr": mm(lambda ix: float((p_suf[ix0][ix] < t).mean()))}
        R.setdefault(name, {})[tag] = res
json.dump(R, open(RESULTS_DIR / "splice_augmented_probe.json", "w"), indent=1)
ci = lambda d: f"{d['value']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"
L = [f"# Splice-augmented probes (Llama, layers standard={Ls}, CSP={Lc}); test split", "",
     f"Training = original train pairs + {R['n_train_spliced']} spliced-sufficient train contexts labelled sufficient. Threshold: balanced accuracy on ORIGINAL val. 95% CI question-clustered bootstrap B={B}.", "",
     "## (i) AUROC on the original test pairs", "", "| Probe | Paper probe | Splice-augmented |", "|---|---|---|"]
for n in feat: L.append(f"| {n} | {ci(R[n]['paper']['auroc_original_pairs'])} | {ci(R[n]['augmented']['auroc_original_pairs'])} |")
L += ["", "## (ii) Control B", "", "| Probe | Training | Source | n | p orig-suf | p spliced-suf | p insuf | AUROC orig vs spliced | AUROC spliced vs insuf | (ref) orig vs insuf | frac spliced < thr | (ref) frac orig < thr |", "|" + "---|" * 12]
for n in feat:
    for tag in ["paper", "augmented"]:
        for s in SRC:
            r = R[n][tag]["controlB"][s]
            L.append(f"| {n} | {tag} | {s} | {r['n']} | " + " | ".join(ci(r[k]) for k in ["mean_p_orig_suf", "mean_p_spliced_suf", "mean_p_insuf", "auroc_orig_vs_spliced", "auroc_spliced_vs_insuf", "auroc_orig_vs_insuf", "frac_spliced_below_thr", "frac_orig_below_thr"]) + " |")
open(RESULTS_DIR / "splice_augmented_probe.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
