"""Recompute parametric-knowledge (PK) labels with a first-sentence rule (v2) and PK-stratified AUROCs.
v1 = original rule (gold in whole 60-token generation, no abstention anywhere)
v2 = gold substring of first sentence, no abstention phrase in first sentence
v2_wb = as v2 but gold must match as whole token sequence (word boundaries)
Normalisation and abstention list copied from src/data/build_sufficiency_bench.py::test_parametric_knowledge.
"""
import json, re, collections
import numpy as np
from scipy.stats import rankdata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data/experiments/sufficiency_bench"
OUT = ROOT / "results/pk_v2"
OUT.mkdir(exist_ok=True)
MODELS = ["llama", "mistral", "qwen", "qwen3", "qwen14", "deepseek32"]
ABSTAIN = ["i don't know", "i'm not sure", "i cannot", "don't have enough", "not sure",
           "i do not know", "unable to", "no information", "cannot determine"]
ABBR = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "prof", "mt", "no", "inc", "co", "ltd", "etc", "e.g", "i.e", "u.s", "gen", "col", "lt", "capt", "sgt", "rev", "fr"}

def norm(s): return s.lower()  # original normalisation: str.lower() only

def first_sentence(text):
    text = text.strip()
    n = len(text)
    for m in re.finditer(r"[.!?]+(?=\s|$)|\n", text):
        end = m.end()
        if text[m.start()] == "\n":
            if text[:m.start()].strip(): return text[:m.start()].strip()
            continue
        if end >= n: return text
        prev = re.findall(r"[\w.]+", text[:m.start()+1])
        last = prev[-1].rstrip(".").lower() if prev else ""
        if len(last) == 1 and last.isalpha() and text[m.start()] == ".": continue  # initials, "J. K."
        if last in ABBR and text[m.start()] == ".": continue
        if re.match(r"\d", text[end:end+1].strip() or "x") and text[m.start()] == "." and last.isdigit(): continue
        return text[:end].strip()
    return text

def has_abstain(s): return any(p in s for p in ABSTAIN)

def wb_match(gold, s):
    return re.search(r"(?<!\w)" + re.escape(gold) + r"(?!\w)", s) is not None

def rules(gold, gen):
    g, r = norm(gold), norm(gen)
    v1 = (g in r) and not has_abstain(r)
    fs = first_sentence(gen); f = norm(fs)
    ok = not has_abstain(f)
    return v1, (g in f) and ok, wb_match(g, f) and ok, fs

def load(split): return json.load(open(BENCH / f"{split}.json"))
splits = {s: load(s) for s in ["train", "val", "test"]}
test = splits["test"]
tid = [q["id"] for q in test]
gold = {q["id"]: q["gold_answer"] for sp in splits.values() for q in sp}
qtype = {q["id"]: q["question_type"] for sp in splits.values() for q in sp}
suf = {c["condition_id"]: c["sufficient"] for q in test for c in q["conditions"]}

res, pk = {}, {}
for m in MODELS:
    cache = json.load(open(BENCH / f"pk_cache_{m}.json"))
    d = {}
    for qid, r in cache.items():
        if qid not in gold: continue
        v1, v2, wb, fs = rules(gold[qid], r["closed_book_answer"])
        d[qid] = dict(knows_v1=v1, knows_v2=v2, knows_v2_wb=wb, first_sentence=fs, knows_cache=bool(r["knows"]))
    json.dump(d, open(OUT / f"pk_v2_{m}.json", "w"), indent=1)
    pk[m] = d

def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

# --- summary pieces
L = []
L.append("# PK rule v2 recomputation\n")
L.append("Original normalisation: `str.lower()` on both gold and generation (substring test). Abstention phrases (copied exactly): " + ", ".join(f"`{p}`" for p in ABSTAIN) + ".\n")
L.append("v2 = gold substring of lower-cased FIRST sentence and no abstention phrase in that first sentence. First sentence = text up to first `.`/`!`/`?` followed by whitespace/end, or first newline; periods after single-letter initials and common abbreviations (Mr, Dr, St, ...) and decimals do not end a sentence. v2_wb = same, gold must match with non-word characters (or edges) on both sides.\n")
L.append("Sanity: v1 recomputed from the stored closed_book_answer (truncated to 300 chars by the original code, so identical to original only if gold appears in the first 300 chars).\n")
L.append("## PK rate, test split (n=%d questions)\n" % len(tid))
L.append("| model | v1 | v2 | v2_wb | v1 cached | #flips v1->v2 (F->T / T->F) | v1 train+val+test | v2 train+val+test |\n|---|---|---|---|---|---|---|---|")
allids = {m: list(pk[m]) for m in MODELS}
for m in MODELS:
    d = pk[m]; a = np.array([[d[i]["knows_v1"], d[i]["knows_v2"], d[i]["knows_v2_wb"], d[i]["knows_cache"]] for i in tid], int)
    ft = int(((a[:,0]==0)&(a[:,1]==1)).sum()); tf = int(((a[:,0]==1)&(a[:,1]==0)).sum())
    al = np.array([[d[i]["knows_v1"], d[i]["knows_v2"]] for i in d], int)
    L.append(f"| {m} | {a[:,0].mean():.1%} | {a[:,1].mean():.1%} | {a[:,2].mean():.1%} | {a[:,3].mean():.1%} | {ft+tf} ({ft} / {tf}) | {al[:,0].mean():.1%} (n={len(al)}) | {al[:,1].mean():.1%} |")

L.append("\n## Quadrant counts over test conditions (Q1 = knows & sufficient, Q2 = not-knows & sufficient, Q3 = knows & insufficient, Q4 = not-knows & insufficient; 2 conditions per question)\n")
L.append("| model | rule | Q1 | Q2 | Q3 | Q4 |\n|---|---|---|---|---|---|")
for m in MODELS:
    for rule in ["knows_v1", "knows_v2"]:
        k = sum(pk[m][i][rule] for i in tid)
        L.append(f"| {m} | {rule[6:]} | {k} | {len(tid)-k} | {k} | {len(tid)-k} |")
L.append("\n(Every question has one sufficient and one insufficient condition, so Q1=Q3=#knows and Q2=Q4=#not-knows.)\n")

qts = sorted(set(qtype[i] for i in tid))
L.append("## v2 PK rate by question type (test)\n")
L.append("| model | " + " | ".join(f"{t} (n={sum(qtype[i]==t for i in tid)})" for t in qts) + " |\n|---|" + "---|" * len(qts))
for m in MODELS:
    L.append(f"| {m} | " + " | ".join(f"{np.mean([pk[m][i]['knows_v2'] for i in tid if qtype[i]==t]):.1%}" for t in qts) + " |")

L.append("\n## Frontier models\n")
L.append("results/api_models contains only claude-opus-4-7_pk_stratified.json (valid JSON; stores aggregate pk_rate=0.3498, n_pk0=723, n_pk1=389, no closed-book text), claude-opus-4-7_results.json (MALFORMED JSON: begins mid-object at `},`) and claude-opus-4-7-cot_results.json (MALFORMED JSON: error at line 100). The valid/parseable parts contain judge_scores arrays, not closed-book generations; no GPT-5.5 files exist. **The frontier PK rates (Opus 4.7 35.0%, GPT-5.5 29.7%) cannot be recomputed under v2.**\n")

L.append("## Ten example flips per model (v1 vs v2; first 10 in test order with v1 != v2)\n")
for m in MODELS:
    fl = [i for i in tid if pk[m][i]["knows_v1"] != pk[m][i]["knows_v2"]]
    L.append(f"### {m} ({len(fl)} flips)\n")
    q = {x["id"]: x["question"] for x in test}
    step = max(1, len(fl)//10)
    for i in fl[::step][:10]:
        d = pk[m][i]
        L.append(f"- `{i}` v1={d['knows_v1']}→v2={d['knows_v2']} | Q: {q[i]} | gold: **{gold[i]}** | first sentence: {d['first_sentence']!r}")
    L.append("")

# --- strata AUROC
def load_scores(m, method):
    if method in ("Standard (LogReg)", "CSP (LogReg)"):
        p = ROOT / f"results/{m}/scores/probe_scores.json"
        if not p.exists(): return None
        d = json.load(open(p))[method]
        return {md["id"]: s for md, s in zip(d["metadata"], d["scores"])}
    if m == "qwen3":
        out = {}
        for line in open(ROOT / "results/qwen3/vc_thinking/chat-nothink.jsonl"):
            r = json.loads(line); out[r["condition_id"]] = r["score"]
        return out
    d = json.load(open(ROOT / f"results/{m}/scores/baseline_scores.json"))["verbalized_confidence"]
    return {md["condition_id"]: s for md, s in zip(d["metadata"], d["scores"])}

METHODS = [("Standard Probe", "Standard (LogReg)"), ("CSP Probe", "CSP (LogReg)"), ("Verbalized Conf.", "vc")]
rng = np.random.default_rng(0)
B = 2000
idx_all = {}
out = {}
rows = []
for m in MODELS:
    for name, key in METHODS:
        sc = load_scores(m, key)
        if sc is None: continue
        qids = [i for i in tid if f"{i}_suf" in sc and f"{i}_insuf" in sc]
        S = np.array([[sc[f"{i}_suf"], sc[f"{i}_insuf"]] for i in qids], float)
        Y = np.array([[suf[f"{i}_suf"], suf[f"{i}_insuf"]] for i in qids], int)
        nq = len(qids)
        draws = rng.integers(0, nq, size=(B, nq))  # same draws reused per (model,method) across rules
        for ver in ["v1", "v2"]:
            K = np.array([pk[m][i]["knows_" + ver] for i in qids], bool)
            r = {}
            for strat, mask in [("overall", np.ones(nq, bool)), ("PK=0", ~K), ("PK=1", K)]:
                def f(ix):
                    ix = ix[mask[ix]]
                    return auroc(Y[ix].ravel(), S[ix].ravel())
                pt = f(np.arange(nq))
                bs = np.array([f(d) for d in draws])
                ok = bs[~np.isnan(bs)]
                ci = [float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))] if len(ok) > 50 else [None, None]
                r[strat] = dict(auroc=None if np.isnan(pt) else float(pt), ci95=ci, n_questions=int(mask.sum()), valid_boot=int(len(ok)))
            out.setdefault(m, {}).setdefault(name, {})[ver] = r
        def fm(x): return "n/a" if x["auroc"] is None else f"{x['auroc']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]" if x["ci95"][0] is not None else f"{x['auroc']:.3f} [n/a]"
        o = out[m][name]
        rows.append(f"| {m} | {name} | {fm(o['v1']['overall'])} | {fm(o['v1']['PK=0'])} (n={o['v1']['PK=0']['n_questions']}) | {fm(o['v1']['PK=1'])} (n={o['v1']['PK=1']['n_questions']}) | {fm(o['v2']['PK=0'])} (n={o['v2']['PK=0']['n_questions']}) | {fm(o['v2']['PK=1'])} (n={o['v2']['PK=1']['n_questions']}) |")
json.dump(out, open(OUT / "pk_strata_auroc.json", "w"), indent=1)
L.append("## PK-stratified AUROC (sufficient vs insufficient; question-clustered bootstrap, 2000 resamples, seed 0, 95% CI)\n")
L.append("Overall AUROC does not depend on the PK rule. qwen3 VC = chat-nothink.jsonl `score`; qwen3 has no probe. Strata with <50 valid bootstrap resamples (single class) show CI n/a.\n")
L.append("| model | method | overall | v1 PK=0 | v1 PK=1 | v2 PK=0 | v2 PK=1 |\n|---|---|---|---|---|---|---|")
L += rows
(OUT / "summary.md").write_text("\n".join(L) + "\n")
print("done")
