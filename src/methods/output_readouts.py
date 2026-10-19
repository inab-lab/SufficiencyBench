"""
output_readouts.py

Stronger OUTPUT-based sufficiency predictors on SufficiencyBench (test split), to
test whether the verbalized-confidence (VC) gap to hidden-state probes is an artefact
of the paper's raw-prompt greedy-integer readout.

Per test condition and model, six scores (all use c["prompt"], the exact base prompt of
baselines.py, and the exact VC meta-prompt of baselines.verbalized_confidence_batch):
  vc_raw            greedy VC, raw prompt, no chat template (replicates the paper; sanity check)
  vc_template       same VC meta-prompt as one user turn via apply_chat_template, greedy
  pyes_raw          P(Yes) = pY/(pY+pN) over next-token logits of Yes/' Yes'/No/' No' (raw)
  pyes_template     same, chat template
  vc_expect_raw     E[d]/9 over first generated token restricted to digit tokens "0".."9"
  vc_expect_template same, template
LIMITATION of vc_expect: only the FIRST generated token is used. Mistral/Qwen tokenise
digits one at a time, so for "85" the first token is "8" (tens digit) and "100" starts
with "1": the expectation is a crude proxy of the tens digit, not of the 0-100 value.
Llama-3 tokenises numbers in chunks of up to 3 digits ("85", "100" are single tokens
outside the digit set), so there the digit-restricted mass is small and the readout is
even cruder (digit_mass is stored in metadata).

Usage:
  python src/methods/output_readouts.py --model llama --device cuda:0 --batch-size 8 [--limit 50] [--out PATH]
  python src/methods/output_readouts.py --summarize            # bootstrap summary from results/*/output_readouts.json
"""
import argparse, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import BENCH_DIR, RESULTS_DIR, MODEL_CONFIGS

SCORES = ["vc_raw", "vc_template", "pyes_raw", "pyes_template",
          "vc_expect_raw", "vc_expect_template"]
MODELS = ["llama", "mistral", "qwen"]
YN_SUFFIX = ("\n\nDoes the context contain enough information to answer the question? "
             "Answer Yes or No.\nAnswer:")


def vc_prompt(p):  # identical to baselines.verbalized_confidence_batch
    return (f"{p}\n\nBefore answering, rate from 0 to 100: "
            f"does the context contain enough information to answer? "
            f"Reply with ONLY the number.\nScore:")


def parse_vc(text):  # identical parsing to baselines
    try:
        nums = re.findall(r'\d+', text.strip())
        return min(max(float(nums[0]) / 100, 0.0), 1.0) if nums else 0.5
    except (ValueError, IndexError):
        return 0.5


def token_ids(tok, variants):
    ids = set()
    for v in variants:
        e = tok.encode(v, add_special_tokens=False)
        if e:
            # sentencepiece may emit a bare '▁' first for ' Yes'; take the last piece if so
            ids.add(e[-1] if len(e) > 1 and tok.decode([e[0]]).strip() == "" else e[0])
    return sorted(ids)


def run(model_key, device, batch_size, limit, out_path):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tqdm import tqdm
    from src.methods.baselines import _quant_kwargs, _chunks

    cfg = MODEL_CONFIGS[model_key]
    tok = AutoTokenizer.from_pretrained(cfg["path"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.float16, device_map=device, **_quant_kwargs(cfg))
    model.eval()

    yes_ids = token_ids(tok, ["Yes", " Yes"])
    no_ids = token_ids(tok, ["No", " No"])
    digit_ids = [token_ids(tok, [str(d)])[0] for d in range(10)]
    print(f"[{model_key}] yes_ids={yes_ids} no_ids={no_ids} digit_ids={digit_ids}", flush=True)
    assert not set(yes_ids) & set(no_ids)

    with open(BENCH_DIR / "test.json") as f:
        test = json.load(f)
    conds = []
    for ex in test:
        for c in ex["conditions"]:
            conds.append({**c, "qid": ex["id"], "question_type": ex.get("question_type")})
    if limit:
        conds = conds[:limit]

    def wrap(text):
        return tok.apply_chat_template([{"role": "user", "content": text}],
                                       tokenize=False, add_generation_prompt=True)

    def encode(texts, templated):
        if templated:  # template already contains BOS etc.
            return tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        return tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)

    @torch.no_grad()
    def vc(texts, templated):
        inp = encode(texts, templated)
        out = model.generate(**inp, max_new_tokens=5, temperature=None, top_p=None, top_k=None,
                             do_sample=False, return_dict_in_generate=True, output_scores=True,
                             pad_token_id=tok.pad_token_id)
        gen = out.sequences[:, inp["input_ids"].shape[1]:]
        sc = [parse_vc(tok.decode(r, skip_special_tokens=True)) for r in gen]
        lg = out.scores[0].float()[:, digit_ids]
        full = torch.softmax(out.scores[0].float(), -1)[:, digit_ids].sum(-1)
        p = torch.softmax(lg, -1)
        ex = (p * torch.arange(10, device=p.device).float()).sum(-1) / 9.0
        return sc, ex.cpu().tolist(), full.cpu().tolist()

    @torch.no_grad()
    def pyes(texts, templated):
        inp = encode(texts, templated)
        try:
            lg = model(**inp, logits_to_keep=1).logits[:, -1, :].float()
        except TypeError:
            lg = model(**inp).logits[:, -1, :].float()
        lg = torch.nan_to_num(lg, nan=0.0, posinf=0.0, neginf=0.0)
        pr = torch.softmax(lg, -1)
        py = pr[:, yes_ids].sum(-1); pn = pr[:, no_ids].sum(-1)
        return (py / (py + pn + 1e-12)).cpu().tolist(), (py + pn).cpu().tolist()

    res = {k: [] for k in SCORES}
    extra = {"yn_mass_raw": [], "yn_mass_template": [], "digit_mass_raw": [], "digit_mass_template": []}
    for b in tqdm(list(_chunks(conds, batch_size)), desc=f"readouts[{model_key}]"):
        base = [c["prompt"] for c in b]
        v = [vc_prompt(p) for p in base]
        y = [p + YN_SUFFIX for p in base]
        s, e, dm = vc(v, False); res["vc_raw"] += s; res["vc_expect_raw"] += e; extra["digit_mass_raw"] += dm
        s, e, dm = vc([wrap(t) for t in v], True); res["vc_template"] += s; res["vc_expect_template"] += e; extra["digit_mass_template"] += dm
        s, m = pyes(y, False); res["pyes_raw"] += s; extra["yn_mass_raw"] += m
        s, m = pyes([wrap(t) for t in y], True); res["pyes_template"] += s; extra["yn_mass_template"] += m

    from sklearn.metrics import roc_auc_score
    labels = [int(bool(c["sufficient"])) for c in conds]
    meta = [{"condition_id": c["condition_id"], "id": c["condition_id"], "qid": c["qid"],
             "quadrant": c["quadrant"], "question_type": c["question_type"], "label": l,
             **{k: extra[k][i] for k in extra}} for i, (c, l) in enumerate(zip(conds, labels))]
    out = {}
    for k in SCORES:
        arr = np.asarray(res[k], float)
        arr = np.where(np.isfinite(arr), arr, np.nanmean(arr[np.isfinite(arr)]))
        out[k] = {"metadata": meta, "scores": arr.tolist(), "labels": labels,
                  "overall_auroc": float(roc_auc_score(labels, arr))}
        print(f"[{model_key}] {k}: AUROC={out[k]['overall_auroc']:.4f}", flush=True)
    out["_info"] = {"yes_ids": yes_ids, "no_ids": no_ids, "digit_ids": digit_ids,
                    "n": len(conds), "limit": limit, "batch_size": batch_size}
    Path = __import__("pathlib").Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"))
    print(f"saved {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Summary: AUROC overall / PK strata, clustered bootstrap CIs, paired delta vs vc_raw
# ---------------------------------------------------------------------------
def _auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def summarize(B=1000, seed=0, models=MODELS):
    rng = np.random.default_rng(seed)
    summ = {}
    for m in models:
        p = RESULTS_DIR / m / "output_readouts.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        meta = d["vc_raw"]["metadata"]
        y = np.array(d["vc_raw"]["labels"])
        quad = np.array([x["quadrant"] for x in meta])
        qid = np.array([x["qid"] for x in meta])
        S = {k: np.array(d[k]["scores"]) for k in SCORES}
        strata = {"overall": np.ones(len(y), bool),
                  "PK0 (Q2 vs Q4)": np.isin(quad, ["Q2", "Q4"]),
                  "PK1 (Q1 vs Q3)": np.isin(quad, ["Q1", "Q3"])}
        uq = np.unique(qid)
        groups = {u: np.where(qid == u)[0] for u in uq}
        point = {k: {st: float(_auc(y[mk], S[k][mk])) for st, mk in strata.items()} for k in SCORES}
        boot = {k: {st: [] for st in strata} for k in SCORES}
        for _ in range(B):
            idx = np.concatenate([groups[u] for u in rng.choice(uq, len(uq))])
            for st, mk in strata.items():
                sel = idx[mk[idx]]
                for k in SCORES:
                    boot[k][st].append(_auc(y[sel], S[k][sel]))
        def ci(a):
            a = np.asarray(a, float); a = a[np.isfinite(a)]
            return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [np.nan, np.nan]
        summ[m] = {"n": int(len(y)), "n_questions": int(len(uq)), "scores": {}}
        for k in SCORES:
            summ[m]["scores"][k] = {}
            for st in strata:
                e = {"auroc": point[k][st], "ci": ci(boot[k][st])}
                if k != "vc_raw":
                    dl = np.asarray(boot[k][st]) - np.asarray(boot["vc_raw"][st])
                    e["delta_vs_vc_raw"] = point[k][st] - point["vc_raw"][st]
                    e["delta_ci"] = ci(dl)
                summ[m]["scores"][k][st] = e
        # sanity vs stored baseline scores
        bp = RESULTS_DIR / m / "scores" / "baseline_scores.json"
        if bp.exists():
            b = json.load(open(bp))["verbalized_confidence"]
            bm = {x["condition_id"]: s for x, s in zip(b["metadata"], b["scores"])}
            ours = {x["condition_id"]: s for x, s in zip(meta, S["vc_raw"])}
            common = [c for c in ours if c in bm]
            summ[m]["sanity"] = {"stored_vc_auroc": b.get("overall_auroc"),
                                 "n_common": len(common),
                                 "frac_identical_scores": float(np.mean([abs(ours[c] - bm[c]) < 1e-6 for c in common])) if common else None}
        summ[m]["mean_digit_mass"] = {t: float(np.mean([x[f"digit_mass_{t}"] for x in meta])) for t in ("raw", "template")}
        summ[m]["mean_yn_mass"] = {t: float(np.mean([x[f"yn_mass_{t}"] for x in meta])) for t in ("raw", "template")}
    json.dump(summ, open(RESULTS_DIR / "output_readouts_summary.json", "w"), indent=1)
    L = ["# Output-based readouts on SufficiencyBench test (AUROC, 95%% question-clustered bootstrap CI, B=%d)\n" % B]
    for m, r in summ.items():
        L.append(f"\n## {m}  (n={r['n']} conditions, {r['n_questions']} questions)\n")
        L.append("| score | overall | PK=0 (Q2 vs Q4) | PK=1 (Q1 vs Q3) | Δ overall vs VC-raw | Δ PK=0 | Δ PK=1 |")
        L.append("|---|---|---|---|---|---|---|")
        for k in SCORES:
            row = [k]
            for st in strata:
                e = r["scores"][k][st]; row.append(f"{e['auroc']:.3f} [{e['ci'][0]:.3f}, {e['ci'][1]:.3f}]")
            for st in strata:
                e = r["scores"][k][st]
                row.append("-" if k == "vc_raw" else f"{e['delta_vs_vc_raw']:+.3f} [{e['delta_ci'][0]:+.3f}, {e['delta_ci'][1]:+.3f}]")
            L.append("| " + " | ".join(row) + " |")
        if "sanity" in r:
            L.append(f"\nSanity: stored VC AUROC {r['sanity']['stored_vc_auroc']:.4f}; recomputed vc_raw {r['scores']['vc_raw']['overall']['auroc']:.4f}; "
                     f"fraction of per-example VC scores identical to stored: {r['sanity']['frac_identical_scores']:.3f}.")
        L.append(f"Mean P(Yes)+P(No) mass raw/template: {r['mean_yn_mass']['raw']:.3f}/{r['mean_yn_mass']['template']:.3f}; "
                 f"mean first-token digit mass raw/template: {r['mean_digit_mass']['raw']:.3f}/{r['mean_digit_mass']['template']:.3f} "
                 f"(vc_expect uses first token only; crude, see module docstring).")
    open(RESULTS_DIR / "output_readouts_summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize()
    else:
        assert a.batch_size <= 8
        run(a.model, a.device, a.batch_size, a.limit,
            a.out or str(RESULTS_DIR / a.model / "output_readouts.json"))
