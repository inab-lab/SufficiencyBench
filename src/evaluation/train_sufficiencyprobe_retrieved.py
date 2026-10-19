"""
Supervised, LLM-free text-only baseline on the retrieved-context experiment (bm25_k5, label = support).

SufficiencyProbe v2 architecture (ModernBERT-large, linear head on CLS(q+c) - CLS(q)):
  transfer : the three existing v2 checkpoints applied unchanged to the retrieved test set
  train    : retrain from base ModernBERT-large on the retrieved train split (BCE only; 10% of train questions,
             seed 0, held out for early stopping + threshold), --max_len 512 (primary) or 1024
  final    : bootstrap CIs, PK=0 strata, and paired deltas vs the retrained Standard LLM probes
             (recomputed with retrieved_context_eval functions; verified against analysis.json)
Run through scripts/run_sprobe_retrieved.sh (thermal-guarded).
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from evaluation.train_sufficiencyprobe_v2 import SufficiencyProbeV2, find_best_threshold  # noqa: E402

MODEL_NAME = "answerdotai/ModernBERT-large"
RC = ROOT / "data/experiments/retrieved_ctx/bm25_k5"
OUT = ROOT / "results/sufficiencyprobe_retrieved"
Q_MAX = 128; BS = 16; LR = 2e-5; EPOCHS = 10; PATIENCE = 3; WARM = 0.10
LLMS = ("llama", "mistral", "qwen")
CKPTS = {"v2": "sufficiencyprobe_v2", "v2_crossdataset": "sufficiencyprobe_v2_crossdataset",
         "v2_crosssource": "sufficiencyprobe_v2_crosssource"}


def ctx_from_prompt(p):  # same as retrieved_context_eval._ctx_from_prompt
    a = p.index("Context: ") + len("Context: ")
    return p[a:p.rindex("\n\nQuestion: ")]


def load(split):
    recs = json.load(open(RC / f"{split}.json"))
    return [dict(id=r["id"], q=r["question"], c=ctx_from_prompt(r["prompt"]), y=int(r["sufficient"])) for r in recs]


def encode(tok, recs, max_len):
    qc = tok([r["q"] for r in recs], [r["c"] for r in recs], max_length=max_len, truncation=True)
    q = tok([r["q"] for r in recs], max_length=Q_MAX, truncation=True)
    return qc["input_ids"], q["input_ids"]


def pad(tok, seqs):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
    m = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s); m[i, :len(s)] = 1
    return ids, m


def batch_logits(model, tok, qc, q, idx, dev):
    a, am = pad(tok, [qc[i] for i in idx]); b, bm = pad(tok, [q[i] for i in idx])
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        return model(a.to(dev), am.to(dev), b.to(dev), bm.to(dev)).float()


@torch.no_grad()
def predict(model, tok, qc, q, dev, bs=32):
    model.eval(); out = []
    order = np.argsort([len(s) for s in qc])  # length-sort for speed
    for i in range(0, len(order), bs):
        out.append(batch_logits(model, tok, qc, q, order[i:i + bs], dev).cpu().numpy())
    lg = np.concatenate(out); res = np.empty_like(lg); res[order] = lg
    return 1 / (1 + np.exp(-res))


def trunc_stats(tok, recs, max_len):
    full = tok([r["q"] for r in recs], [r["c"] for r in recs], truncation=False)["input_ids"]
    qlen = [len(x) for x in tok([r["q"] for r in recs], truncation=False)["input_ids"]]
    n = np.array([len(x) for x in full]); cut = np.maximum(n - max_len, 0)
    ctxlen = np.maximum(n - np.array(qlen), 1)
    return {"max_len": max_len, "n": len(recs), "mean_len": float(n.mean()), "median_len": float(np.median(n)),
            "max_total_len": int(n.max()), "frac_truncated": float((n > max_len).mean()),
            "mean_tokens_cut_among_truncated": float(cut[n > max_len].mean()) if (n > max_len).any() else 0.0,
            "mean_frac_context_cut": float((cut / ctxlen).mean())}


def load_model(dev, ckpt=None):
    m = SufficiencyProbeV2(MODEL_NAME)
    if ckpt is not None:
        m.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state_dict"])
    return m.to(dev)


def cmd_transfer(a, dev):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    te = load("test"); qc, q = encode(tok, te, 512)
    res = {"ids": [r["id"] for r in te], "y": [r["y"] for r in te], "max_len": 512, "scores": {}}
    for name, d in CKPTS.items():
        m = load_model(dev, ROOT / "results" / d / "best_checkpoint.pt")
        s = predict(m, tok, qc, q, dev); res["scores"][name] = s.tolist()
        print(name, "AUROC", roc_auc_score(res["y"], s), flush=True); del m; torch.cuda.empty_cache()
    json.dump(res, open(OUT / "transfer_scores.json", "w"))


def cmd_train(a, dev):
    ML = a.max_len; micro = a.micro; acc = BS // micro
    torch.manual_seed(0); np.random.seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tr_all, te = load("train"), load("test")
    perm = np.random.RandomState(0).permutation(len(tr_all)); nv = int(round(0.1 * len(tr_all)))
    val = [tr_all[i] for i in perm[:nv]]; tr = [tr_all[i] for i in perm[nv:]]
    stats = {"train": trunc_stats(tok, tr_all, ML), "test": trunc_stats(tok, te, ML)}
    print("truncation", json.dumps(stats), flush=True)
    enc = {k: encode(tok, v, ML) for k, v in (("tr", tr), ("val", val), ("te", te))}
    ytr = np.array([r["y"] for r in tr], float); yval = np.array([r["y"] for r in val]); yte = np.array([r["y"] for r in te])
    model = load_model(dev)
    steps_ep = (len(tr) + BS - 1) // BS
    opt = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(steps_ep * EPOCHS * WARM), steps_ep * EPOCHS)
    ck = OUT / ("best_checkpoint.pt" if ML == 512 else f"best_checkpoint_len{ML}.pt")
    best, best_ep, pat, log = -1.0, 0, 0, []
    for ep in range(1, EPOCHS + 1):
        model.train(); t0 = time.time(); losses = []
        p = np.random.RandomState(1000 + ep).permutation(len(tr))
        for i in range(0, len(p), BS):
            idx = p[i:i + BS]; opt.zero_grad()
            for j in range(0, len(idx), micro):
                sub = idx[j:j + micro]
                lg = batch_logits(model, tok, enc["tr"][0], enc["tr"][1], sub, dev)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    lg, torch.tensor(ytr[sub], dtype=torch.float, device=dev), reduction="sum") / len(idx)
                loss.backward(); losses.append(loss.item() * len(idx) / len(sub))
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        sv = predict(model, tok, *enc["val"], dev); va = float(roc_auc_score(yval, sv))
        print(f"Epoch {ep} bce={np.mean(losses):.4f} val_auroc={va:.4f} ({(time.time()-t0)/60:.1f}min)", flush=True)
        log.append({"epoch": ep, "bce": float(np.mean(losses)), "val_auroc": va})
        if va > best:
            best, best_ep, pat = va, ep, 0
            torch.save({"model_state_dict": model.state_dict(), "val_auroc": va, "epoch": ep, "max_len": ML}, ck)
        else:
            pat += 1
            if pat >= PATIENCE: print("early stop", flush=True); break
    model.load_state_dict(torch.load(ck, map_location=dev)["model_state_dict"])
    sv = predict(model, tok, *enc["val"], dev); st = predict(model, tok, *enc["te"], dev)
    thr, ba = find_best_threshold(sv, yval)
    pred = (st >= thr).astype(int)
    tpr = float(((pred == 1) & (yte == 1)).sum() / max((yte == 1).sum(), 1)); tnr = float(((pred == 0) & (yte == 0)).sum() / max((yte == 0).sum(), 1))
    res = {"max_len": ML, "n_train": len(tr), "n_val": len(val), "best_epoch": best_ep, "val_auroc": best,
           "threshold": thr, "val_balanced_acc": ba, "test_auroc": float(roc_auc_score(yte, st)),
           "test_balanced_acc_at_thr": (tpr + tnr) / 2, "test_acc_at_thr": float((pred == yte).mean()),
           "truncation": stats, "log": log, "test_ids": [r["id"] for r in te], "test_scores": st.tolist(),
           "val_ids": [r["id"] for r in val]}
    json.dump(res, open(OUT / f"train_len{ML}.json", "w"))
    print("TEST AUROC", res["test_auroc"], flush=True)


# ------------------------------------------------------------------ final analysis
def load_rce():
    spec = importlib.util.spec_from_file_location("rce", ROOT / "src/evaluation/retrieved_context_eval.py")
    m = importlib.util.module_from_spec(spec); sys.modules["rce"] = m; spec.loader.exec_module(m); return m


def llm_probe_scores(rce, model):
    """Retrained Standard probe, support label, exactly as cmd_analyse: returns (test ids kept, scores, y, pk, reproduced auroc)."""
    from src.analysis.store_baseline_scores import load_best_layers
    sc_tr, rec_tr, hqc_tr, _, _ = rce._load_split("bm25_k5", model, "train")
    sc_te, rec_te, hqc_te, _, _ = rce._load_split("bm25_k5", model, "test")
    _, std_layer = load_best_layers(model)
    f32 = lambda x: np.asarray(x, dtype=np.float32)
    m_tr, y_tr = rce._label_arrays(rec_tr, "support"); m_te, y_te = rce._label_arrays(rec_te, "support")
    keep_tr = np.array([not r.get("truncated", 0) for r in sc_tr]); keep_te = np.array([not r.get("truncated", 0) for r in sc_te])
    y_tr = y_tr[keep_tr[m_tr]]; m_tr = m_tr & keep_tr; y_te = y_te[keep_te[m_te]]; m_te = m_te & keep_te
    clf = rce._fit_lr(f32(hqc_tr[:, std_layer, :])[m_tr], y_tr)
    s = clf.predict_proba(f32(hqc_te[:, std_layer, :])[m_te])[:, 1]
    ids = [r["id"] for r, k in zip(sc_te, m_te) if k]
    pk = {r["id"]: int(r["pk"]) for r in sc_te}
    ref = json.load(open(ROOT / f"results/{model}/retrieved_ctx_bm25_k5/analysis_support.json"))["methods"]["Standard probe (retrained)"]["auroc"]
    return ids, s, y_te, pk, rce.auroc(y_te, s), ref


def boot_ci(y, s, B_idx):
    v = np.array([roc_auc_score(y[b], s[b]) for b in B_idx]); return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def cmd_final(a, dev):
    rce = load_rce(); rng = np.random.default_rng(0); B = 1000
    tr_sc = json.load(open(OUT / "transfer_scores.json"))
    runs = {ML: json.load(open(OUT / f"train_len{ML}.json")) for ML in (512, 1024) if (OUT / f"train_len{ML}.json").exists()}
    ids = tr_sc["ids"]; y = np.array(tr_sc["y"]); pos = {i: k for k, i in enumerate(ids)}
    pks = {m: json.load(open(ROOT / f"results/{m}/retrieved_ctx_bm25_k5/test_scores.json")) for m in LLMS}
    pkd = {m: {r["id"]: int(r["pk"]) for r in pks[m]} for m in LLMS}
    sys_scores = {f"transferred_{k}": np.array(v) for k, v in tr_sc["scores"].items()}
    for ML, r in runs.items():
        assert r["test_ids"] == ids
        sys_scores[f"retrained_len{ML}"] = np.array(r["test_scores"])
    n = len(y); Bi = rng.integers(0, n, size=(B, n))
    out = {"n_test": n, "pos_rate": float(y.mean()), "B": B, "label": "support", "systems": {}, "llm_probes": {}, "paired_delta": {}}
    for k, s in sys_scores.items():
        d = {"auroc": float(roc_auc_score(y, s)), "ci95": boot_ci(y, s, Bi)}
        for m in LLMS:
            msk = np.array([pkd[m][i] == 0 for i in ids]); sub = np.nonzero(msk)[0]
            Bs = rng.integers(0, len(sub), size=(B, len(sub)))
            d[f"auroc_pk0_{m}"] = float(roc_auc_score(y[msk], s[msk])); d[f"n_pk0_{m}"] = int(msk.sum())
            d[f"auroc_pk0_{m}_ci95"] = boot_ci(y[msk], s[msk], Bs)
        out["systems"][k] = d
    for m in LLMS:
        pid, ps, py, _, rep, ref = llm_probe_scores(rce, m)
        ok = abs(rep - ref) <= 0.002
        print(f"{m}: reproduced {rep:.4f} vs analysis.md {ref:.4f} ok={ok}", flush=True)
        # align on the questions the LLM probe scored (non-truncated)
        sel = np.array([pos[i] for i in pid]); yy = y[sel]; assert (yy == py).all()
        Bm = rng.integers(0, len(sel), size=(B, len(sel)))
        out["llm_probes"][m] = {"auroc_reproduced": rep, "auroc_analysis_md": ref, "within_0.002": bool(ok), "n": len(sel),
                                "ci95": boot_ci(py, ps, Bm), "scores": dict(zip(pid, ps.tolist()))}
        for k, s in sys_scores.items():
            ss = s[sel]; dd = np.array([roc_auc_score(yy[b], ss[b]) - roc_auc_score(yy[b], ps[b]) for b in Bm])
            pk0 = np.array([pkd[m][i] == 0 for i in pid]); 
            Bp = rng.integers(0, pk0.sum(), size=(B, pk0.sum()))
            d0 = np.array([roc_auc_score(yy[pk0][b], ss[pk0][b]) - roc_auc_score(yy[pk0][b], ps[pk0][b]) for b in Bp])
            out["paired_delta"][f"{k}__minus__{m}_retrained_standard"] = {
                "delta": float(roc_auc_score(yy, ss) - roc_auc_score(yy, ps)), "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
                "p_two_sided": float(min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean()))), "n": len(sel),
                "delta_pk0": float(roc_auc_score(yy[pk0], ss[pk0]) - roc_auc_score(yy[pk0], ps[pk0])),
                "ci95_pk0": [float(np.percentile(d0, 2.5)), float(np.percentile(d0, 97.5))], "n_pk0": int(pk0.sum())}
    out["training"] = {str(ML): {k: v for k, v in r.items() if k not in ("test_scores", "test_ids", "val_ids", "log")} | {"log": r["log"]} for ML, r in runs.items()}
    out["per_example"] = {"ids": ids, "y": y.tolist(), "pk": {m: [pkd[m][i] for i in ids] for m in LLMS},
                          "scores": {k: s.tolist() for k, s in sys_scores.items()}}
    json.dump(out, open(OUT / "results.json", "w"))
    f = lambda x: f"{x:.3f}"; ci = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]"
    L = ["# SufficiencyProbe on retrieved context (bm25_k5, label = support)", "",
         f"Test N={n} (sufficient rate {y.mean():.3f}); bootstrap over questions, B={B}. Text-only ModernBERT-large, no LLM.", "",
         "| System | AUROC [95% CI] | PK=0 llama | PK=0 mistral | PK=0 qwen |", "|---|---|---|---|---|"]
    for k, d in out["systems"].items():
        L.append(f"| {k} | {f(d['auroc'])} {ci(d['ci95'])} | " + " | ".join(f"{f(d[f'auroc_pk0_{m}'])} {ci(d[f'auroc_pk0_{m}_ci95'])} (n={d[f'n_pk0_{m}']})" for m in LLMS) + " |")
    L += ["", "LLM retrained Standard probes (recomputed; analysis.md value in brackets):", ""]
    for m, d in out["llm_probes"].items():
        L.append(f"- {m}: {f(d['auroc_reproduced'])} {ci(d['ci95'])} [{f(d['auroc_analysis_md'])}], within 0.002: {d['within_0.002']}, n={d['n']}")
    L += ["", "Paired delta AUROC (system minus LLM retrained Standard probe), bootstrap over questions:", "",
          "| Contrast | Δ [95% CI] | p | Δ PK=0 [95% CI] (PK of that LLM) |", "|---|---|---|---|"]
    for k, d in out["paired_delta"].items():
        L.append(f"| {k} | {d['delta']:+.3f} {ci(d['ci95'])} | {d['p_two_sided']:.3f} | {d['delta_pk0']:+.3f} {ci(d['ci95_pk0'])} (n={d['n_pk0']}) |")
    L += ["", "Training:", ""]
    for ML, d in out["training"].items():
        t = d["truncation"]
        L.append(f"- max_length {ML}: best epoch {d['best_epoch']}, val AUROC {f(d['val_auroc'])}, threshold {d['threshold']:.2f} (val), test AUROC {f(d['test_auroc'])}, "
                 f"test balanced acc at thr {f(d['test_balanced_acc_at_thr'])}. Truncation (train): {100*t['train']['frac_truncated']:.1f}% of examples cut, "
                 f"mean {t['train']['mean_tokens_cut_among_truncated']:.0f} tokens cut among those, mean {100*t['train']['mean_frac_context_cut']:.1f}% of context tokens; mean total length {t['train']['mean_len']:.0f}.")
    (OUT / "summary.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["transfer", "train", "final"]); ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--micro", type=int, default=16); ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    cmd = {"transfer": cmd_transfer, "train": cmd_train, "final": cmd_final}[a.mode]
    cmd(a, torch.device(a.device))
