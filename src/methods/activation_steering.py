#!/usr/bin/env python3
"""
activation_steering.py — Sufficiency-Guided Activation Steering

Uses the DECO probe's weight vector as a steering vector added to hidden
states during generation.  The sufficiency direction identified by
neuroscience analysis (RSA, CCGP, d-prime) is used to *causally intervene*
on model behavior.

Three steering directions:
  1. deco_probe  — LogReg on h(q+c) − h(q)  (supervised, DECO space)
  2. std_probe   — LogReg on h(q+c)          (supervised, standard)
  3. meandiff_raw — mean(h_suf) − mean(h_insuf)  on h(q+c)  (unsupervised)

Design for robustness:
  • Direction extraction and model generation are strictly separate phases.
    Directions are saved as small .npy files; all hidden-state arrays are
    freed before any model is loaded.
  • Every example is checkpointed to a JSONL file so a crash loses at most
    one generation.
  • Alpha sweep is kept small: [−2, −1, 1, 2] × 2 direction types + baseline
    = 9 configs  (~1,800 generations, not 5,000).

Usage:
  # Phase A: directions only (no GPU, <1 min)
  python src/methods/activation_steering.py --step directions --model mistral

  # Phase B: full steering (GPU, ~2–4 h per model)
  python src/methods/activation_steering.py --step steer --model mistral

  # Phase C: figures from completed results
  python src/methods/activation_steering.py --step figures --model mistral
"""

import json, gc, os, sys, signal, argparse, time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.paths import (
    BENCH_DIR, HIDDEN_STATES_DIR, RESULTS_DIR, FIGURES_DIR,
    MODEL_CONFIGS, CRAG_DIR,
)

# ── Guard against silent SIGTERM/SIGXCPU (OOM-killer sends SIGKILL which
#    can't be caught, but SIGTERM from cgroups can) ──
def _signal_handler(signum, frame):
    print(f"\n[WARN] Caught signal {signum} — saving checkpoint and exiting")
    sys.exit(1)

for _sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2):
    try:
        signal.signal(_sig, _signal_handler)
    except (OSError, ValueError):
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

ABSTAIN_PHRASES = [
    "cannot answer", "don't have enough", "not enough information",
    "i cannot", "i don't know", "unable to answer",
    "the context does not", "no information",
    "cannot be determined", "not provided", "not mentioned",
    "insufficient information", "does not provide",
    "not specified", "no relevant information",
]

DEFAULT_ALPHAS = [-2.0, -1.0, 1.0, 2.0]
DEFAULT_DIRECTIONS = ["deco_probe", "meandiff_raw"]

# ──────────────────────────────────────────────────────────────────────────────
# Phase A: Extract and save steering directions  (no GPU, no model)
# ──────────────────────────────────────────────────────────────────────────────

def extract_and_save_directions(model_key: str) -> dict:
    """
    Extract candidate steering directions from saved hidden states.
    Saves each direction as a small .npy file under results/<model>/directions/.
    Returns a summary dict (also saved as JSON).
    """
    print(f"\n{'='*70}")
    print(f"DIRECTION EXTRACTION — {model_key}")
    print(f"{'='*70}")

    cfg = MODEL_CONFIGS[model_key]
    n_layers = cfg["n_layers"]
    hs_dir = HIDDEN_STATES_DIR / model_key / "train"

    # Memory-map only — no materialisation yet
    h_qc = np.load(str(hs_dir / "h_with_context.npy"), mmap_mode="r")
    h_q  = np.load(str(hs_dir / "h_question_only.npy"), mmap_mode="r")
    labels = np.load(str(hs_dir / "labels.npy"))

    # Choose target layers from DECO results
    with open(RESULTS_DIR / model_key / "deco_results.json") as f:
        deco_res = json.load(f)
    best_deco_layer = deco_res["best_layers"]["DECO"]
    best_std_layer  = deco_res["best_layers"]["standard"]

    target_layers = sorted(set([
        best_deco_layer, best_std_layer,
        n_layers // 2, 3 * n_layers // 4,
        n_layers - 4, n_layers - 1,
    ]))
    target_layers = [l for l in target_layers if 0 <= l < n_layers]

    out_dir = RESULTS_DIR / model_key / "directions"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"model": model_key, "target_layers": target_layers, "per_layer": {}}

    for layer in target_layers:
        print(f"\n  Layer {layer}:")

        # --- Materialise just this layer (float32) ---
        X_deco_raw = np.array(h_qc[:, layer, :] - h_q[:, layer, :], dtype=np.float32)
        X_std_raw  = np.array(h_qc[:, layer, :], dtype=np.float32)

        # 1. DECO probe
        sc_d = StandardScaler(); X_d = sc_d.fit_transform(X_deco_raw)
        pr_d = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        pr_d.fit(X_d, labels)
        dir_deco = (pr_d.coef_[0] / sc_d.scale_).astype(np.float32)
        dir_deco /= np.linalg.norm(dir_deco)
        auroc_deco = roc_auc_score(labels, pr_d.predict_proba(X_d)[:, 1])
        print(f"    DECO probe AUROC: {auroc_deco:.4f}")

        # 2. Standard probe
        sc_s = StandardScaler(); X_s = sc_s.fit_transform(X_std_raw)
        pr_s = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        pr_s.fit(X_s, labels)
        dir_std = (pr_s.coef_[0] / sc_s.scale_).astype(np.float32)
        dir_std /= np.linalg.norm(dir_std)

        # 3. Mean-difference on h(q+c)
        suf = labels == 1; insuf = labels == 0
        dir_md = X_std_raw[suf].mean(0) - X_std_raw[insuf].mean(0)
        dir_md = (dir_md / np.linalg.norm(dir_md)).astype(np.float32)

        # Cosines
        cos = {
            "deco_std":     float(np.dot(dir_deco, dir_std)),
            "deco_meandiff": float(np.dot(dir_deco, dir_md)),
            "std_meandiff":  float(np.dot(dir_std, dir_md)),
        }
        for k, v in cos.items():
            print(f"    cos({k}) = {v:.4f}")

        # Save direction vectors
        np.save(str(out_dir / f"deco_probe_L{layer}.npy"), dir_deco)
        np.save(str(out_dir / f"std_probe_L{layer}.npy"),  dir_std)
        np.save(str(out_dir / f"meandiff_raw_L{layer}.npy"), dir_md)

        report["per_layer"][str(layer)] = {"auroc_deco": auroc_deco, "cosines": cos}

        del X_deco_raw, X_std_raw, X_d, X_s
        gc.collect()

    # CCD comparison (if available)
    ccd_path = CRAG_DIR / model_key
    ccd_files = list(ccd_path.glob("ccd_L*.npy")) if ccd_path.exists() else []
    for cf in ccd_files:
        lnum = int(cf.stem.split("L")[1])
        ccd = np.load(str(cf)).astype(np.float32)
        ccd /= np.linalg.norm(ccd)
        deco_f = out_dir / f"deco_probe_L{lnum}.npy"
        if deco_f.exists():
            d = np.load(str(deco_f))
            c = float(np.dot(d, ccd))
            report["per_layer"].setdefault(str(lnum), {})["cos_deco_ccd"] = c
            print(f"\n  CCD at L{lnum}: cos(deco,ccd) = {c:.4f}")

    # Free everything
    del h_qc, h_q, labels
    gc.collect()

    with open(RESULTS_DIR / model_key / "steering_directions.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDirections saved to {out_dir}/")
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Phase B helpers
# ──────────────────────────────────────────────────────────────────────────────

def check_abstain(text):
    t = text.lower()
    return any(p in t for p in ABSTAIN_PHRASES)


def format_prompt(question, context):
    return (
        f"Based on the following context, answer the question. "
        f"If the context does not contain enough information, "
        f"say 'I cannot answer this from the provided context.'\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def generate_with_steering(
    model, tokenizer, prompt, device,
    steering_direction=None, target_layers=None,
    alpha=0.0, max_new_tokens=256,
):
    """Generate with optional activation steering via forward hooks."""
    hooks = []

    if steering_direction is not None and alpha != 0.0 and target_layers:
        dir_t = torch.tensor(steering_direction, dtype=torch.float32, device=device)

        def _make_hook(layer_idx):
            def _fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                s = alpha * dir_t.to(h.dtype)
                h[:, :, :] += s.unsqueeze(0).unsqueeze(0)
                return (h,) + out[1:] if isinstance(out, tuple) else h
            return _fn

        for li in target_layers:
            handle = model.model.layers[li].register_forward_hook(_make_hook(li))
            hooks.append(handle)

    try:
        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=2048,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=0.0, do_sample=False,
            )
        text = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
    finally:
        for h in hooks:
            h.remove()
        del inputs
        if 'out' in dir():
            del out
        torch.cuda.empty_cache()

    return text


# ──────────────────────────────────────────────────────────────────────────────
# Phase B: Steering experiment with checkpointing
# ──────────────────────────────────────────────────────────────────────────────

def load_directions_from_disk(model_key, layer):
    """Load pre-saved direction vectors (tiny files)."""
    d_dir = RESULTS_DIR / model_key / "directions"
    dirs = {}
    for name in ["deco_probe", "std_probe", "meandiff_raw"]:
        p = d_dir / f"{name}_L{layer}.npy"
        if p.exists():
            dirs[name] = np.load(str(p))
    return dirs


def build_example_list(model_key, n_samples):
    """Load test data and sample balanced across quadrants."""
    with open(BENCH_DIR / "test.json") as f:
        test_data = json.load(f)

    examples = []
    for ex in test_data:
        for cond in ex["conditions"]:
            examples.append({
                "question": ex["question"],
                "gold_answer": ex["gold_answer"],
                "model_knows": ex["model_knows_answer"],
                "context": cond["context"],
                "prompt": cond["prompt"],
                "sufficient": cond["sufficient"],
                "quadrant": cond["quadrant"],
                "question_type": ex["question_type"],
            })

    np.random.seed(42)
    by_quad = {}
    for e in examples:
        by_quad.setdefault(e["quadrant"], []).append(e)

    selected = []
    per_quad = max(n_samples // 4, 10)
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        pool = by_quad.get(q, [])
        n = min(per_quad, len(pool))
        idx = np.random.choice(len(pool), n, replace=False)
        selected.extend([pool[i] for i in idx])

    print(f"  Selected {len(selected)} examples:")
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        print(f"    {q}: {sum(1 for e in selected if e['quadrant'] == q)}")
    return selected


def run_steering_experiment(
    model_key: str = "mistral",
    device: str = "cuda:0",
    n_samples: int = 200,
    alphas=None,
    direction_types=None,
    max_new_tokens: int = 256,
):
    if alphas is None:
        alphas = DEFAULT_ALPHAS
    if direction_types is None:
        direction_types = DEFAULT_DIRECTIONS

    print(f"\n{'='*70}")
    print(f"ACTIVATION STEERING EXPERIMENT — {model_key}")
    print(f"{'='*70}")

    # ── 1. Ensure directions exist (no GPU needed) ──
    dir_json = RESULTS_DIR / model_key / "steering_directions.json"
    if not dir_json.exists():
        print("  Directions not found — extracting first...")
        extract_and_save_directions(model_key)
        gc.collect()

    # ── 2. Choose steering layer and window ──
    with open(RESULTS_DIR / model_key / "deco_results.json") as f:
        deco_res = json.load(f)
    steer_layer = deco_res["best_layers"]["DECO"]
    n_layers = MODEL_CONFIGS[model_key]["n_layers"]
    steer_window = sorted(set(
        [steer_layer] +
        [steer_layer + off for off in [-4, -2, 2, 4]
         if 0 <= steer_layer + off < n_layers]
    ))

    # ── 3. Load directions from small .npy files ──
    dirs = load_directions_from_disk(model_key, steer_layer)
    if not dirs:
        raise FileNotFoundError(
            f"No direction files at results/{model_key}/directions/ for layer {steer_layer}"
        )

    # Build config list: (name, direction_type, alpha)
    configs = [("baseline", None, 0.0)]
    for dt in direction_types:
        if dt not in dirs:
            print(f"  [WARN] direction '{dt}' not found, skipping")
            continue
        for alpha in alphas:
            configs.append((f"{dt}_a{alpha}", dt, alpha))

    print(f"\n  Steering from layer {steer_layer}, applied at {steer_window}")
    print(f"  Configs: {len(configs)}  (baseline + "
          f"{len(configs)-1} steering)")

    # ── 4. Build example list ──
    selected = build_example_list(model_key, n_samples)

    # ── 5. Load checkpoint (crash recovery) ──
    ckpt_path = RESULTS_DIR / model_key / "steering_checkpoint.jsonl"
    done_indices = set()
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            for line in f:
                rec = json.loads(line)
                done_indices.add(rec["example_idx"])
        print(f"  Resuming: {len(done_indices)} examples already done")

    # ── 6. Load model (AFTER directions are loaded and hidden states freed) ──
    cfg = MODEL_CONFIGS[model_key]
    print(f"\n  Loading {model_key} in 8-bit on {device}...")
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        quantization_config=bnb_config,
        device_map=device,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    print("  Model loaded.")

    # ── 7. Generation loop with per-example checkpointing ──
    total_gens = (len(selected) - len(done_indices)) * len(configs)
    print(f"\n  Remaining: {len(selected) - len(done_indices)} examples × "
          f"{len(configs)} configs = {total_gens} generations")

    t0 = time.time()
    ckpt_f = open(ckpt_path, "a")

    try:
        for ex_idx, ex in enumerate(tqdm(selected, desc="Examples")):
            if ex_idx in done_indices:
                continue

            result = {
                "example_idx": ex_idx,
                "quadrant": ex["quadrant"],
                "sufficient": ex["sufficient"],
                "model_knows": ex["model_knows"],
                "gold_answer": ex["gold_answer"],
                "question_type": ex["question_type"],
                "generations": {},
            }

            for config_name, dt, alpha in configs:
                direction = dirs.get(dt) if dt else None
                text = generate_with_steering(
                    model, tokenizer, ex["prompt"], device,
                    steering_direction=direction,
                    target_layers=steer_window if direction is not None else None,
                    alpha=alpha,
                    max_new_tokens=max_new_tokens,
                )
                abstains = check_abstain(text)
                correct = (
                    ex["gold_answer"].lower() in text.lower() and not abstains
                )
                result["generations"][config_name] = {
                    "text": text[:500],
                    "abstains": abstains,
                    "correct": correct,
                }

            # Checkpoint immediately
            ckpt_f.write(json.dumps(result) + "\n")
            ckpt_f.flush()

            if (ex_idx + 1) % 20 == 0:
                elapsed = time.time() - t0
                done_now = ex_idx + 1 - len(done_indices)
                rate = elapsed / max(done_now, 1)
                remaining = (len(selected) - ex_idx - 1) * rate
                print(f"    [{ex_idx+1}/{len(selected)}] "
                      f"{rate:.1f}s/ex, ~{remaining/60:.0f}min left")
                torch.cuda.empty_cache()

    finally:
        ckpt_f.close()

    # ── 8. Consolidate checkpoint → final results ──
    all_results = []
    with open(ckpt_path) as f:
        for line in f:
            all_results.append(json.loads(line))

    save_data = {
        "model": model_key,
        "steer_layer": steer_layer,
        "steer_window": steer_window,
        "n_examples": len(all_results),
        "configs": [(n, d, a) for n, d, a in configs],
        "examples": all_results,
    }
    with open(RESULTS_DIR / model_key / "steering_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Final results saved ({len(all_results)} examples)")

    # ── 9. Print analysis ──
    _print_steering_analysis(model_key, all_results, configs)

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return all_results


def _print_steering_analysis(model_key, results, configs):
    """Per-quadrant analysis for each configuration."""
    print(f"\n{'='*70}")
    print(f"STEERING RESULTS — {model_key}")
    print(f"{'='*70}")

    header = f"{'Config':<30s}"
    for q in ["Q1(s+c)", "Q2(s+u)", "Q3(i+c)", "Q4(i+u)"]:
        header += f" {q:<12s}"
    header += f" {'Composite':<10s}"
    print(f"\n{header}")
    print("-" * 90)

    best_composite = -1
    best_name = ""

    for config_name, dt, alpha in configs:
        per_quad = {}
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            qr = [r for r in results if r["quadrant"] == q]
            if not qr:
                per_quad[q] = "N/A"
                continue
            correct = sum(1 for r in qr if r["generations"][config_name]["correct"])
            abstain = sum(1 for r in qr if r["generations"][config_name]["abstains"])
            per_quad[q] = f"{correct}/{len(qr)}c {abstain}/{len(qr)}a"

        suf   = [r for r in results if r["sufficient"]]
        insuf = [r for r in results if not r["sufficient"]]
        correct_rate = sum(
            1 for r in suf if r["generations"][config_name]["correct"]
        ) / max(len(suf), 1)
        abstain_rate = sum(
            1 for r in insuf if r["generations"][config_name]["abstains"]
        ) / max(len(insuf), 1)
        over_refuse = sum(
            1 for r in suf if r["generations"][config_name]["abstains"]
        ) / max(len(suf), 1)
        composite = (correct_rate + abstain_rate - over_refuse) / 2

        if composite > best_composite:
            best_composite = composite
            best_name = config_name

        row = f"{config_name:<30s}"
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            row += f" {per_quad[q]:<12s}"
        row += f" {composite:<10.3f}"
        print(row)

    print(f"\n  >>> Best config: {best_name} (composite={best_composite:.3f})")

    # Q3 detail
    print(f"\n{'='*70}")
    print("Q3 (Insufficient + Confident) — target for abstention improvement")
    print(f"{'='*70}")
    q3 = [r for r in results if r["quadrant"] == "Q3"]
    if q3:
        for cn, dt, alpha in configs:
            ab = sum(1 for r in q3 if r["generations"][cn]["abstains"])
            co = sum(1 for r in q3 if r["generations"][cn]["correct"])
            print(f"  {cn:<30s}: abstain={ab}/{len(q3)} ({ab/len(q3):.0%}), "
                  f"correct={co}/{len(q3)} ({co/len(q3):.0%})")

    # Q1 detail
    print(f"\n{'='*70}")
    print("Q1 (Sufficient + Confident) — must preserve correctness")
    print(f"{'='*70}")
    q1 = [r for r in results if r["quadrant"] == "Q1"]
    if q1:
        for cn, dt, alpha in configs:
            ab = sum(1 for r in q1 if r["generations"][cn]["abstains"])
            co = sum(1 for r in q1 if r["generations"][cn]["correct"])
            print(f"  {cn:<30s}: correct={co}/{len(q1)} ({co/len(q1):.0%}), "
                  f"abstain={ab}/{len(q1)} ({ab/len(q1):.0%})")


# ──────────────────────────────────────────────────────────────────────────────
# Phase C: Figures
# ──────────────────────────────────────────────────────────────────────────────

def generate_steering_figures(model_key: str):
    """Paper-ready figures from steering results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.labelsize": 12, "axes.titlesize": 13,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    MODEL_LABELS = {
        "mistral": "Mistral 7B", "qwen": "Qwen 2.5 7B", "llama": "Llama 3.1 8B",
    }

    with open(RESULTS_DIR / model_key / "steering_results.json") as f:
        data = json.load(f)

    results = data["examples"]
    configs = data["configs"]

    # Identify direction types and alphas
    dtypes, alphas = set(), set()
    for name, dt, alpha in configs:
        if dt:
            dtypes.add(dt)
            alphas.add(alpha)
    dtypes = sorted(dtypes)
    alphas = sorted(alphas)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {
        "deco_probe": "#D32F2F", "std_probe": "#1976D2", "meandiff_raw": "#4CAF50",
    }

    for dt in dtypes:
        composites, abstain_q3 = [], []
        for alpha in alphas:
            cn = f"{dt}_a{alpha}"
            suf   = [r for r in results if r["sufficient"]]
            insuf = [r for r in results if not r["sufficient"]]
            cr = sum(1 for r in suf if r["generations"].get(cn, {}).get("correct")) / max(len(suf), 1)
            ar = sum(1 for r in insuf if r["generations"].get(cn, {}).get("abstains")) / max(len(insuf), 1)
            orf = sum(1 for r in suf if r["generations"].get(cn, {}).get("abstains")) / max(len(suf), 1)
            composites.append((cr + ar - orf) / 2)

            q3 = [r for r in results if r["quadrant"] == "Q3"]
            aq3 = sum(1 for r in q3 if r["generations"].get(cn, {}).get("abstains")) / max(len(q3), 1)
            abstain_q3.append(aq3)

        label = dt.replace("_", " ").title()
        c = colors.get(dt, "C0")
        axes[0].plot(alphas, composites, "o-", color=c, label=label, lw=2, ms=5)
        axes[1].plot(alphas, abstain_q3, "o-", color=c, label=label, lw=2, ms=5)

    # Baseline reference
    suf   = [r for r in results if r["sufficient"]]
    insuf = [r for r in results if not r["sufficient"]]
    bc = sum(1 for r in suf if r["generations"]["baseline"]["correct"]) / max(len(suf), 1)
    ba = sum(1 for r in insuf if r["generations"]["baseline"]["abstains"]) / max(len(insuf), 1)
    bo = sum(1 for r in suf if r["generations"]["baseline"]["abstains"]) / max(len(suf), 1)
    base_comp = (bc + ba - bo) / 2
    axes[0].axhline(y=base_comp, color="gray", ls="--", alpha=.7, label="Baseline")

    q3b = [r for r in results if r["quadrant"] == "Q3"]
    bq3 = sum(1 for r in q3b if r["generations"]["baseline"]["abstains"]) / max(len(q3b), 1)
    axes[1].axhline(y=bq3, color="gray", ls="--", alpha=.7, label="Baseline")

    axes[0].set(xlabel="Steering Strength (α)", ylabel="Composite Faithfulness",
                title="Faithfulness vs. Steering Strength")
    axes[0].legend()
    axes[1].set(xlabel="Steering Strength (α)", ylabel="Abstention Rate",
                title="Q3 (Insuf+Conf) Abstention vs. Steering")
    axes[1].legend()

    fig.suptitle(f"{MODEL_LABELS.get(model_key, model_key)}: Activation Steering",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig_steering_{model_key}.pdf")
    fig.savefig(FIGURES_DIR / f"fig_steering_{model_key}.png")
    plt.close(fig)
    print(f"  Saved fig_steering_{model_key}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sufficiency-guided activation steering experiments"
    )
    parser.add_argument("--step", required=True,
                        choices=["directions", "steer", "figures"])
    parser.add_argument("--model", default="mistral",
                        choices=list(MODEL_CONFIGS.keys()) + ["all"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--alphas", type=float, nargs="+", default=None,
                        help="Alpha values (default: -2 -1 1 2)")
    parser.add_argument("--directions", nargs="+", default=None,
                        help="Direction types (default: deco_probe meandiff_raw)")
    args = parser.parse_args()

    models = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]

    for mk in models:
        try:
            if args.step == "directions":
                extract_and_save_directions(mk)

            elif args.step == "steer":
                run_steering_experiment(
                    model_key=mk, device=args.device,
                    n_samples=args.n_samples,
                    alphas=args.alphas, direction_types=args.directions,
                )

            elif args.step == "figures":
                generate_steering_figures(mk)

        except Exception as e:
            print(f"\nERROR on {mk}: {e}")
            import traceback
            traceback.print_exc()
