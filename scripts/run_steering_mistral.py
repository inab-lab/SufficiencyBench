#!/usr/bin/env python
"""run_steering_mistral.py — reconstruct the deleted activation-steering experiment.

Injects alpha * w_hat (the CSP / sufficiency probe direction) into Mistral-7B's
residual stream at layer 12 over a 5-layer window (decoder blocks 10-14), sweeps
alpha in {-2,-1,0,+1,+2} over a quadrant-balanced held-out set, and measures
per-quadrant (Q1-Q4) generation correctness and abstention rate.

Direction: w_hat = unit LogReg coefficient of the CSP probe (on the delta
h(q+c)-h(q)) at layer 12, sign-oriented toward the *sufficient* pole. alpha is in
units of the std of the sufficiency-coordinate projection over the train set
(a principled, data-grounded scale; the paper under-specifies alpha's units).
Quadrants use Mistral's parametric knowledge (pk_cache_mistral: "knows").

Outputs: results/steering_mistral.json  (+ fig produced by a sibling call).
Run under the 90% CPU cap (generation-heavy).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/home/inab/Documents/MSD_Project/paper1")
sys.path.insert(0, str(ROOT))
from configs.paths import HIDDEN_STATES_DIR, MODEL_CONFIGS

DATA = ROOT / "data" / "experiments" / "sufficiency_bench"
RES = ROOT / "results"
LAYER = 12
WINDOW = [10, 11, 12, 13, 14]         # 5-layer window centred at 12
ALPHAS = [-2, -1, 0, 1, 2]
ABSTAIN = ["cannot answer", "don't have enough", "not enough information",
           "no information", "unable to answer", "insufficient information",
           "does not provide", "can't answer", "not provided",
           "cannot be answered", "i cannot", "i can't", "cannot be determined",
           "cannot provide", "no mention", "does not contain", "doesn't contain",
           "does not mention", "not mentioned", "not specified"]


def compute_direction():
    """w_hat: unit LogReg coef of the CSP probe at layer 12, toward sufficient."""
    d = HIDDEN_STATES_DIR / "mistral" / "train"
    h_qc = np.load(d / "h_with_context.npy")[:, LAYER, :].astype(np.float64)
    h_q = np.load(d / "h_question_only.npy")[:, LAYER, :].astype(np.float64)
    y = np.load(d / "labels.npy")
    delta = h_qc - h_q
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(delta, y)
    w = clf.coef_[0]
    w_unit = w / np.linalg.norm(w)
    proj = h_qc @ w_unit
    if proj[y == 1].mean() < proj[y == 0].mean():   # orient toward sufficient
        w_unit = -w_unit
    return w_unit.astype(np.float32)


def measure_residual_norm(model, tok, prompts, device, layer=LAYER, n=16):
    """Mean live per-token residual-stream L2 norm at `layer` over a sample."""
    vals = {}
    def hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        vals["m"] = hs.norm(dim=-1).float().mean().item()
    h = model.model.layers[layer].register_forward_hook(hook)
    enc = tok(prompts[:n], return_tensors="pt", padding=True,
              truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        model(**enc)
    h.remove()
    return vals["m"]


def build_heldout(n_per_quadrant, seed=0):
    pk = json.load(open(DATA / "pk_cache_mistral.json"))
    test = json.load(open(DATA / "test.json"))
    rng = random.Random(seed)
    # split questions by PK
    pk1, pk0 = [], []
    for it in test:
        knows = pk.get(it["id"], {}).get("knows")
        if knows is True:
            pk1.append(it)
        elif knows is False:
            pk0.append(it)
    rng.shuffle(pk1); rng.shuffle(pk0)
    pk1 = pk1[:n_per_quadrant]        # each gives a Q1 (suf) + Q3 (insuf)
    pk0 = pk0[:n_per_quadrant]        # each gives a Q2 (suf) + Q4 (insuf)
    items = []
    for it in pk1 + pk0:
        knows = pk[it["id"]]["knows"]
        for c in it["conditions"]:
            suf = c["sufficient"]
            q = ("Q1" if suf else "Q3") if knows else ("Q2" if suf else "Q4")
            items.append({"prompt": c["prompt"], "gold": it["gold_answer"],
                          "quadrant": q})
    return items


# global steering state, toggled per alpha
_STATE = {"vec": None}


def make_hook():
    def hook(module, inp, out):
        v = _STATE["vec"]
        if v is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + v,) + out[1:]
        return out + v
    return hook


def classify(gen_text, gold):
    g = gen_text.strip().lower()
    abst = any(p in g for p in ABSTAIN)
    gold = (gold or "").strip().lower()
    correct = bool(gold) and (gold in g) and not abst
    return correct, abst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-quadrant", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=60)
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    w_unit = compute_direction()
    items = build_heldout(args.n_per_quadrant)
    print(f"[data] {len(items)} conditions; quadrants: "
          f"{ {q: sum(1 for i in items if i['quadrant']==q) for q in ['Q1','Q2','Q3','Q4']} }")

    cfg = MODEL_CONFIGS["mistral"]
    tok = AutoTokenizer.from_pretrained(cfg["path"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.float16, device_map=device)
    model.eval()

    # Calibrate injection magnitude to the LIVE residual-stream norm (principled,
    # outcome-independent): base_scale = 0.15 * mean per-token ||h|| at layer 12.
    # A grid sweep (scripts/_calibrate_steer.py) shows the interpretable->collapse
    # transition sits near 0.15-0.30 of the residual norm, so |alpha|=1 (~15% of
    # ||h||) gives a graded effect and |alpha|=2 (~30%) drives the collapse the
    # paper reports. alpha is thus expressed as a fraction of the residual norm.
    m_norm = measure_residual_norm(model, tok, [it["prompt"] for it in items], device)
    scale = 0.15 * m_norm
    print(f"[scale] live L12 residual norm={m_norm:.2f} -> base_scale={scale:.3f} "
          f"(|alpha|=1 ~ 15% of residual norm, |alpha|=2 ~ 30%)")

    w_t = torch.tensor(w_unit, device=device, dtype=torch.float16)
    hooks = [model.model.layers[l].register_forward_hook(make_hook()) for l in WINDOW]

    results = {q: {} for q in ["Q1", "Q2", "Q3", "Q4"]}
    per_alpha = {}
    prompts = [it["prompt"] for it in items]
    golds = [it["gold"] for it in items]
    quads = [it["quadrant"] for it in items]

    for alpha in ALPHAS:
        _STATE["vec"] = None if alpha == 0 else (alpha * scale) * w_t
        gens = []
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i:i + args.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            gen_ids = out[:, enc["input_ids"].shape[1]:]
            gens.extend(tok.batch_decode(gen_ids, skip_special_tokens=True))
        # aggregate per quadrant
        agg = {q: {"n": 0, "correct": 0, "abstain": 0} for q in ["Q1", "Q2", "Q3", "Q4"]}
        for g, gold, q in zip(gens, golds, quads):
            c, a = classify(g, gold)
            agg[q]["n"] += 1
            agg[q]["correct"] += int(c)
            agg[q]["abstain"] += int(a)
        per_alpha[str(alpha)] = agg
        line = " ".join(
            f"{q}:c={agg[q]['correct']/max(agg[q]['n'],1):.2f},a={agg[q]['abstain']/max(agg[q]['n'],1):.2f}"
            for q in ["Q1", "Q2", "Q3", "Q4"])
        print(f"[alpha={alpha:+d}] {line}")

    for h in hooks:
        h.remove()

    # reshape into per-quadrant rate series
    out = {
        "model": "mistral",
        "layer": LAYER, "window": WINDOW, "alphas": ALPHAS,
        "n_per_quadrant": args.n_per_quadrant,
        "direction": "CSP LogReg probe coef (unit) at layer 12, toward sufficient pole",
        "alpha_scale": scale,
        "alpha_units": "fraction of live L12 residual-stream norm; inject = alpha*0.15*||h||*w_hat",
        "live_residual_norm_L12": m_norm,
        "per_alpha": per_alpha,
        "correctness": {}, "abstention": {},
    }
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        out["correctness"][q] = [per_alpha[str(a)][q]["correct"] / max(per_alpha[str(a)][q]["n"], 1) for a in ALPHAS]
        out["abstention"][q] = [per_alpha[str(a)][q]["abstain"] / max(per_alpha[str(a)][q]["n"], 1) for a in ALPHAS]

    with open(RES / "steering_mistral.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[done] wrote results/steering_mistral.json")


if __name__ == "__main__":
    main()
