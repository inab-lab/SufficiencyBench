#!/usr/bin/env python
"""run_steering_random_control.py -- random-direction control for run_steering_mistral.py.
Usage: cd ROOT; PYTHONPATH=ROOT:ROOT/src HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
  python scripts/run_steering_random_control.py [--n-random 5] [--seed-base 0] [--include-probe-direction]
Same protocol as the original (data seed 0, L13, window 11-15, alphas -2..2, 0.15*mean residual norm,
same hook/prompts/batch/metrics) but w_hat -> K seeded Gaussian unit directions (same norm = 1).
Runtime ~ (4K+1)/5 x original (alpha=0 is direction-independent, run once): ~4.2x for K=5; +0.8x with probe.
Output: results/steering_mistral_random_control.json (per-direction per_alpha/correctness/abstention + summary).
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPTS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("run_steering_mistral", SCRIPTS / "run_steering_mistral.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)   # original keeps main() under __main__ guard -> safe to import

QUADS = ["Q1", "Q2", "Q3", "Q4"]
HEADLINE = "abstention_shift_pos2_minus_neg2"   # abstention rate(alpha=+2) - abstention rate(alpha=-2)


def random_direction(dim, seed):
    v = np.random.default_rng(seed).standard_normal(dim)
    return (v / np.linalg.norm(v)).astype(np.float32)


def generate(model, tok, prompts, device, batch_size, max_new):
    gens = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gens.extend(tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return gens


def aggregate(gens, golds, quads):
    agg = {q: {"n": 0, "correct": 0, "abstain": 0} for q in QUADS}
    for g, gold, q in zip(gens, golds, quads):
        c, a = S.classify(g, gold)
        agg[q]["n"] += 1
        agg[q]["correct"] += int(c)
        agg[q]["abstain"] += int(a)
    return agg


def series(per_alpha, key):
    return {q: [per_alpha[str(a)][q][key] / max(per_alpha[str(a)][q]["n"], 1) for a in S.ALPHAS]
            for q in QUADS}


def shift(per_alpha):
    """abstention(+2) - abstention(-2): per quadrant and pooled over all conditions."""
    def rate(a, qs):
        n = sum(per_alpha[str(a)][q]["n"] for q in qs)
        return sum(per_alpha[str(a)][q]["abstain"] for q in qs) / max(n, 1)
    d = {q: rate(2, [q]) - rate(-2, [q]) for q in QUADS}
    d["pooled"] = rate(2, QUADS) - rate(-2, QUADS)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-quadrant", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--out", default=str(S.RES / "steering_mistral_random_control.json"))
    ap.add_argument("--include-probe-direction", action="store_true",
                    help="also run the real CSP probe direction (paired comparison)")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    items = S.build_heldout(args.n_per_quadrant)   # default seed=0, same as original
    prompts = [it["prompt"] for it in items]
    golds = [it["gold"] for it in items]
    quads = [it["quadrant"] for it in items]
    print(f"[data] {len(items)} conditions")

    cfg = S.MODEL_CONFIGS["mistral"]
    tok = AutoTokenizer.from_pretrained(cfg["path"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(cfg["path"], torch_dtype=torch.float16, device_map=device)
    model.eval()

    m_norm = S.measure_residual_norm(model, tok, prompts, device)
    scale = 0.15 * m_norm
    print(f"[scale] residual norm={m_norm:.2f} base_scale={scale:.3f}")

    dirs = {}
    if args.include_probe_direction:
        dirs["probe"] = S.compute_direction()
    seeds = [args.seed_base + k for k in range(args.n_random)]
    dim = model.config.hidden_size
    for s in seeds:
        dirs[f"random_{s}"] = random_direction(dim, s)

    hooks = [model.model.layers[l].register_forward_hook(S.make_hook()) for l in S.WINDOW]

    # alpha=0 applies no steering, so greedy generation is direction-independent: run once, share.
    S._STATE["vec"] = None
    agg0 = aggregate(generate(model, tok, prompts, device, args.batch_size, args.max_new), golds, quads)

    per_dir = {}
    for name, w in dirs.items():
        w_t = torch.tensor(w, device=device, dtype=torch.float16)
        per_alpha = {}
        for alpha in S.ALPHAS:
            if alpha == 0:
                per_alpha["0"] = json.loads(json.dumps(agg0))
                continue
            S._STATE["vec"] = (alpha * scale) * w_t
            per_alpha[str(alpha)] = aggregate(
                generate(model, tok, prompts, device, args.batch_size, args.max_new), golds, quads)
            print(f"[{name} alpha={alpha:+d}] " + " ".join(
                f"{q}:a={per_alpha[str(alpha)][q]['abstain']/max(per_alpha[str(alpha)][q]['n'],1):.2f}"
                for q in QUADS))
        per_dir[name] = {"per_alpha": per_alpha,
                         "correctness": series(per_alpha, "correct"),
                         "abstention": series(per_alpha, "abstain"),
                         HEADLINE: shift(per_alpha)}
    S._STATE["vec"] = None
    for h in hooks:
        h.remove()

    rnd = [per_dir[n][HEADLINE] for n in per_dir if n.startswith("random_")]
    summary = {"headline_quantity": HEADLINE,
               "definition": "abstention rate at alpha=+2 minus at alpha=-2 (per quadrant and pooled)",
               "n_random": len(rnd),
               "random_mean": {k: float(np.mean([r[k] for r in rnd])) for k in rnd[0]} if rnd else None,
               "random_std": {k: float(np.std([r[k] for r in rnd], ddof=1)) if len(rnd) > 1 else None
                              for k in rnd[0]} if rnd else None}
    if "probe" in per_dir:
        summary["probe"] = per_dir["probe"][HEADLINE]

    out = {"model": "mistral", "layer": S.LAYER, "window": S.WINDOW, "alphas": S.ALPHAS,
           "n_per_quadrant": args.n_per_quadrant, "alpha_scale": scale,
           "live_residual_norm": m_norm, "random_seeds": seeds,
           "direction_note": "random: N(0,I) in R^hidden, L2-normalised (unit, same norm as probe w_hat)",
           "directions": per_dir, "summary": summary}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
