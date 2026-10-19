import json, sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "data"))
from evaluation.splice_controls import SPLICE_HS_DIR, SPLICE_DIR, RESULTS_DIR, DATASETS_DIR, MODELS, CTX_ONLY_DIR, qid_of, auc, cluster_boot, bal_acc_threshold, load_split, norm_ws
from methods.deco import make_classifier
from datasets import load_from_disk
B = 1000
ds = load_from_disk(str(DATASETS_DIR / "hotpotqa")); d = ds["validation" if "validation" in ds else list(ds.keys())[0]]
spl = json.load(open(SPLICE_DIR / "test.json")); removed = {}
for e in spl:
    if not e["id"].startswith("hotpot"): continue
    ex = d[int(e["id"].split("_")[1])]; titles = ex["context"]["title"]; sents = ex["context"]["sentences"]; sup = []
    for t, i in zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"]):
        if t in titles and i < len(sents[titles.index(t)]): sup.append(norm_ws(sents[titles.index(t)][i]))
    rs = e["conditions"][0]["replaced_sentence"]
    hit = any(s and (s in rs or rs in s) for s in sup) or any(s and len(set(s.split()) & set(rs.split())) >= 0.5 * min(len(s.split()), len(rs.split())) for s in sup)
    removed[e["id"]] = bool(hit)
print(sum(removed.values()), len(removed) - sum(removed.values()))
ci = lambda v: f"{v['value']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"
rows = []; out = {}
def evaluate(m, tag, pr_std, pr_csp, Ls, Lc, va, te, Hs, qs, strata):
    pos = {i: k for k, i in enumerate(te[3])}; i_suf = np.array([pos[x + "_suf"] for x in qs]); i_ins = np.array([pos[x + "_insuf"] for x in qs])
    fx = {"standard": (pr_std, lambda a, q: np.asarray(a[:, Ls])), "csp": (pr_csp, lambda a, q: np.asarray(a[:, Lc] - q[:, Lc]))}
    for pn, (pr, f) in fx.items():
        sc = lambda a, q: pr.predict_proba(f(a, q))[:, 1]; t = bal_acc_threshold(va[2], sc(va[0], va[1]))
        s_all = sc(te[0], te[1]); p_suf, p_ins, p_spl = s_all[i_suf], s_all[i_ins], sc(Hs, te[1][i_suf])
        for sname, mask in strata.items():
            ix0 = np.where(mask)[0]; g = qs[ix0]
            a2 = lambda pp, pn_, ix: auc(np.r_[np.ones(len(ix)), np.zeros(len(ix))], np.r_[pp[ix0][ix], pn_[ix0][ix]])
            mm = lambda fn: {"value": fn(np.arange(len(ix0))), "ci95": cluster_boot(fn, g, B)}
            r = {"n": int(len(ix0)), "mean_p_orig_suf": mm(lambda ix: float(p_suf[ix0][ix].mean())), "mean_p_spliced_suf": mm(lambda ix: float(p_spl[ix0][ix].mean())),
                 "mean_p_insuf": mm(lambda ix: float(p_ins[ix0][ix].mean())), "auroc_orig_vs_spliced": mm(lambda ix: a2(p_suf, p_spl, ix)),
                 "auroc_spliced_vs_insuf": mm(lambda ix: a2(p_spl, p_ins, ix)), "frac_spliced_below_thr": mm(lambda ix: float((p_spl[ix0][ix] < t).mean())),
                 "frac_orig_below_thr": mm(lambda ix: float((p_suf[ix0][ix] < t).mean()))}
            out[f"{m}|{tag}|{pn}|{sname}"] = r
            rows.append(f"| {m} | {tag} | {pn} | {sname} | {r['n']} | " + " | ".join(ci(r[k]) for k in ["mean_p_orig_suf", "mean_p_spliced_suf", "mean_p_insuf", "auroc_orig_vs_spliced", "auroc_spliced_vs_insuf", "frac_spliced_below_thr", "frac_orig_below_thr"]) + " |")
for m in MODELS:
    best = json.load(open(RESULTS_DIR / m / "all_results_logreg.json"))["best_layers"]; Ls, Lc = best["standard"], best["DECO"]
    tr = load_split(m, "train"); va = load_split(m, "val"); te = load_split(m, "test")
    Hs = np.asarray(np.load(SPLICE_HS_DIR / m / "test/h_with_context.npy"), dtype=np.float32); ids = json.load(open(SPLICE_HS_DIR / m / "test/ids.json"))
    qs = np.array([qid_of(i) for i in ids]); hot = np.array([q.startswith("hotpot") for q in qs]); rem = np.array([removed.get(q, False) for q in qs])
    strata = {"hotpot, swap did NOT remove a supporting fact": hot & ~rem, "hotpot, swap removed a supporting fact": hot & rem}
    std = make_classifier("logreg").fit(np.asarray(tr[0][:, Ls]), tr[2]); csp = make_classifier("logreg").fit(np.asarray(tr[0][:, Lc] - tr[1][:, Lc]), tr[2])
    evaluate(m, "paper", std, csp, Ls, Lc, va, te, Hs, qs, strata)
    if m == "llama":
        dd = SPLICE_HS_DIR / m / "train"; Hst = np.asarray(np.load(dd / "h_with_context.npy"), dtype=np.float32); idt = json.load(open(dd / "ids.json"))
        pos = {i: k for k, i in enumerate(tr[3])}; isuf = np.array([pos[qid_of(i) + "_suf"] for i in idt])
        ones = np.ones(len(Hst), dtype=tr[2].dtype)
        astd = make_classifier("logreg").fit(np.vstack([np.asarray(tr[0][:, Ls]), Hst[:, Ls]]), np.r_[tr[2], ones])
        acsp = make_classifier("logreg").fit(np.vstack([np.asarray(tr[0][:, Lc] - tr[1][:, Lc]), Hst[:, Lc] - np.asarray(tr[1][isuf][:, Lc])]), np.r_[tr[2], ones])
        evaluate(m, "augmented", astd, acsp, Ls, Lc, va, te, Hs, qs, strata)
head = ["", "## HotpotQA by whether the swap removed a supporting fact", "",
        f"Stratum 'removed' = replaced sentence contains/is contained in a dataset supporting-fact sentence or shares >=50% of words with one ({sum(removed.values())} questions); the other stratum has {len(removed)-sum(removed.values())}. CIs: question-clustered bootstrap B={B}.", "",
        "| Model | Training | Probe | Stratum | n | p orig-suf | p spliced-suf | p insuf | AUROC orig vs spliced | AUROC spliced vs insuf | frac spliced < thr | frac orig < thr |", "|" + "---|" * 12]
md = RESULTS_DIR / "splice_controls_by_source.md"; txt = md.read_text()
if "HotpotQA by whether the swap" not in txt: md.write_text(txt + "\n".join(head + rows) + "\n")
js = RESULTS_DIR / "splice_controls_by_source.json"; J = json.load(open(js)); J["hotpot_by_supporting_fact_removed"] = out; json.dump(J, open(js, "w"), indent=1)
print("\n".join(head + rows))
