"""
extract_longcontext.py — last-token hidden states at long context.

Same readout as extraction/extract_states.py (last token, every decoder layer,
float32 on CPU), but at 4k-100k tokens instead of <=2048.  Two things make that
possible on a 32GB card:

  1. NO ``output_hidden_states=True``.  That would materialise
     (n_layers+1) x seq_len x hidden -- 33 x 100k x 4096 x 2B = 27 GB at 100k.
     Instead a forward hook on every decoder layer keeps only ``out[0][:, -1, :]``.
  2. The forward runs through ``model.model`` (the bare decoder stack), not the
     CausalLM wrapper, so ``lm_head`` is never applied.  A full-sequence logit
     tensor at 100k tokens would be 100k x 128256 x 2B = 26 GB on its own.

Layer indexing matches the short-context caches exactly: row i of the saved
array is the output of decoder layer i+1, i.e. ``hidden_states[1:]`` (the
embedding output is skipped).  ``--verify-hooks`` proves that on a short prompt.

Outputs, per length:
  data/experiments/longcontext/<model>/<length>/
      h_with_context.npy   (N_rows, n_layers, hidden)  float32
      labels.npy           (N_rows,)  1 = sufficient
      metadata.json        one dict per row
      .done                written last
h(q) is context-independent and therefore cached once per model:
      data/experiments/longcontext/<model>/h_question_only.npy  (n_questions, ...)
      data/experiments/longcontext/<model>/hq_index.json        question_id -> row

Usage:
  python src/longcontext/extract_longcontext.py --model llama --device cuda:1
  python src/longcontext/extract_longcontext.py --model llama --device cuda:1 \
      --limit 2 --lengths base 4000
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longcontext.common import (  # noqa: E402
    LENGTHS, MODEL_CONFIGS, QWEN_YARN_MAX_POSITIONS, QWEN_YARN_ROPE,
    RUN_LOGS_DIR, atomic_write_json, build_prompt, format_question_only,
    get_logger, hq_index_path, hq_path, length_dir, load_manifest,
    manifest_entries, max_positions_for, needs_yarn, realised_max_tokens,
    supported_lengths,
)

CHECKPOINT_EVERY = 20
CONDITIONS = ("sufficient", "insufficient")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(model_key: str, device: str, yarn: bool = False):
    """8-bit (bf16 compute) load, mirroring extract_states.load_model, plus
    sdpa attention and -- for Qwen above 32k -- YaRN rope scaling."""
    cfg = MODEL_CONFIGS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    kwargs = dict(
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map=device,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if yarn:
        kwargs["rope_scaling"] = dict(QWEN_YARN_ROPE)
        kwargs["max_position_embeddings"] = QWEN_YARN_MAX_POSITIONS

    model = AutoModelForCausalLM.from_pretrained(cfg["path"], **kwargs)
    model.eval()
    return model, tokenizer


def decoder_layers(model):
    """The ModuleList of decoder blocks (Llama / Mistral / Qwen2 all use
    ``model.model.layers``)."""
    return model.model.layers


class LastTokenHooks:
    """Forward hooks that keep ONLY the last-token activation of each decoder
    layer.  Memory cost is n_layers x hidden floats, independent of seq_len.

    Row i reproduces ``outputs.hidden_states[1:][i]`` exactly, which is what the
    short-context caches store.  That tuple is assembled by LlamaModel/Qwen2Model
    /MistralModel as::

        (embeddings, out(L0), ..., out(L_{n-2}), final_norm(out(L_{n-1})))

    i.e. the LAST element has passed through ``model.model.norm`` while the
    others have not.  So we hook layers 0..n-2 plus the final norm, NOT all n
    layers -- hooking layer n-1 instead would give a pre-norm vector that the
    transferred probe has never seen.
    """

    def __init__(self, model):
        self.layers = decoder_layers(model)
        self.norm = model.model.norm
        self.n_layers = len(self.layers)
        self.buf = [None] * self.n_layers
        self._handles = []

    def _make(self, i):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self.buf[i] = h[:, -1, :].detach().to(torch.float32).cpu()
        return hook

    def __enter__(self):
        for i, layer in enumerate(self.layers[:-1]):
            self._handles.append(layer.register_forward_hook(self._make(i)))
        self._handles.append(
            self.norm.register_forward_hook(self._make(self.n_layers - 1)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []

    def stack(self) -> np.ndarray:
        """(n_layers, hidden) float32 — rows are decoder layers 1..N."""
        if any(b is None for b in self.buf):
            raise RuntimeError("a decoder layer hook did not fire")
        return np.concatenate([b.numpy() for b in self.buf], axis=0)


def forward_last_token(model, tokenizer, text: str, device, hooks) -> tuple:
    """Run one prompt through the decoder stack; return (states, n_tokens).

    No truncation, no logits, no KV cache.
    """
    inputs = tokenizer(text, return_tensors="pt")
    n_tokens = int(inputs["input_ids"].shape[1])
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        model.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            use_cache=False,
        )
    return hooks.stack(), n_tokens


# ---------------------------------------------------------------------------
# Hook-vs-output_hidden_states parity check
# ---------------------------------------------------------------------------

def verify_hooks(model, tokenizer, device, logger):
    """Assert the hook readout equals ``outputs.hidden_states[1:][:, -1, :]``
    on a short prompt (the representation the short-context caches hold)."""
    text = format_question_only("What is the capital of France?")
    with LastTokenHooks(model) as hooks:
        hooked, n = forward_last_token(model, tokenizer, text, device, hooks)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    ref = np.stack([hs[0, -1, :].float().cpu().numpy()
                    for hs in out.hidden_states[1:]])
    max_diff = float(np.max(np.abs(hooked - ref)))
    logger.info(f"[verify-hooks] shape={hooked.shape} (ref {ref.shape}) "
                f"n_tokens={n} max|diff|={max_diff:.3e}")
    assert hooked.shape == ref.shape, "hook layer count != hidden_states[1:]"
    assert max_diff < 1e-4, f"hook readout differs from hidden_states ({max_diff})"
    del out
    torch.cuda.empty_cache()
    return max_diff


# ---------------------------------------------------------------------------
# h(q) cache
# ---------------------------------------------------------------------------

def ensure_hq(model, tokenizer, device, model_key, entries, logger):
    """Compute (or load) h(q) once per question — the DECO subtrahend.

    h(q) does not depend on the context, so it is shared by every length.
    """
    qids, seen = [], set()
    for e in entries:
        if e["question_id"] not in seen:
            seen.add(e["question_id"])
            qids.append({"question_id": e["question_id"], "question": e["question"]})

    ipath, hpath = hq_index_path(model_key), hq_path(model_key)
    if ipath.exists() and hpath.exists():
        index = json.load(open(ipath))
        if all(q["question_id"] in index for q in qids):
            logger.info(f"[hq] cache hit ({len(index)} questions)")
            return np.load(hpath), index
        logger.info("[hq] cache incomplete — recomputing")

    hpath.parent.mkdir(parents=True, exist_ok=True)
    rows, index = [], {}
    t0 = time.time()
    with LastTokenHooks(model) as hooks:
        for q in qids:
            h, _ = forward_last_token(
                model, tokenizer, format_question_only(q["question"]), device, hooks)
            index[q["question_id"]] = len(rows)
            rows.append(h)
    arr = np.stack(rows).astype(np.float32)
    np.save(hpath, arr)
    atomic_write_json(ipath, index)
    logger.info(f"[hq] computed {arr.shape} in {time.time() - t0:.1f}s -> {hpath}")
    return arr, index


# ---------------------------------------------------------------------------
# Per-length extraction
# ---------------------------------------------------------------------------

def _plan(entries):
    """Flat, deterministic (entry, condition) work list for one length."""
    return [(e, c) for e in entries for c in CONDITIONS]


def _load_checkpoint(out_dir):
    ckpt = out_dir / "partial.npz"
    meta = out_dir / "partial_meta.json"
    if not (ckpt.exists() and meta.exists()):
        return [], [], []
    try:
        z = np.load(ckpt)
        m = json.load(open(meta))
        h = list(z["h"])
        labels = list(z["labels"])
        if len(h) == len(labels) == len(m):
            return h, labels, m
    except Exception as exc:  # noqa: BLE001 — a corrupt checkpoint just restarts
        print(f"  [warn] unreadable checkpoint in {out_dir}: {exc}")
    return [], [], []


def _save_checkpoint(out_dir, h, labels, meta):
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "partial.tmp.npz"   # np.savez appends .npz unless the name already ends with it
    np.savez(tmp, h=np.stack(h).astype(np.float32),
             labels=np.array(labels, dtype=np.int64))
    os.replace(tmp, out_dir / "partial.npz")
    atomic_write_json(out_dir / "partial_meta.json", meta)


def extract_length(model, tokenizer, device, model_key, length, entries, logger):
    out_dir = length_dir(model_key, length)
    if (out_dir / ".done").exists():
        logger.info(f"[{length}] already done — skipping")
        return

    plan = _plan(entries)
    h, labels, meta = _load_checkpoint(out_dir)
    if meta:
        logger.info(f"[{length}] resuming from checkpoint at {len(meta)}/{len(plan)}")

    limit = max_positions_for(model_key, length)
    torch.cuda.reset_peak_memory_stats(device)
    t_length = time.time()

    with LastTokenHooks(model) as hooks:
        for i in range(len(meta), len(plan)):
            entry, cond = plan[i]
            cid = entry["condition_ids"][cond]
            prompt = build_prompt(entry, cond)

            t0 = time.time()
            try:
                n_est = entry.get("n_prompt_tokens", {}).get(cond)
                if n_est is not None and n_est > limit:
                    logger.warning(
                        f"[{length}] SKIP {cid}: {n_est} tokens > max_positions {limit}")
                    meta.append({"condition_id": cid, "skipped": "too_long",
                                 "n_tokens": n_est})
                    continue
                states, n_tokens = forward_last_token(
                    model, tokenizer, prompt, device, hooks)
                if n_tokens > limit:
                    logger.warning(
                        f"[{length}] SKIP {cid}: {n_tokens} tokens > {limit}")
                    meta.append({"condition_id": cid, "skipped": "too_long",
                                 "n_tokens": n_tokens})
                    continue
            except torch.cuda.OutOfMemoryError as exc:
                logger.error(f"[{length}] OOM on {cid}: {exc}")
                torch.cuda.empty_cache()
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "extract_oom.txt").write_text(f"{cid}: {exc}\n")
                if h:
                    _save_checkpoint(out_dir, h, labels, meta)
                return
            dt = time.time() - t0

            h.append(states)
            labels.append(1 if cond == "sufficient" else 0)
            peak = torch.cuda.max_memory_allocated(device) / 1e9
            meta.append({
                "condition_id": cid,
                "question_id": entry["question_id"],
                "question_type": entry["question_type"],
                "position": entry["position"],
                "position_label": entry["position_label"],
                "sufficient": cond == "sufficient",
                "length": length,
                "n_tokens": n_tokens,
                "seconds": round(dt, 3),
            })
            logger.info(f"[{length}] {i + 1}/{len(plan)} {cid} "
                        f"tok={n_tokens} {dt:.2f}s peak={peak:.2f}GB")

            if (i + 1) % CHECKPOINT_EVERY == 0 and h:
                _save_checkpoint(out_dir, h, labels, meta)

    if not h:
        logger.error(f"[{length}] no rows extracted — not writing outputs")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "h_with_context.npy", np.stack(h).astype(np.float32))
    np.save(out_dir / "labels.npy", np.array(labels, dtype=np.int64))
    kept = [m for m in meta if "skipped" not in m]
    atomic_write_json(out_dir / "metadata.json", kept)
    for stale in ("partial.npz", "partial_meta.json"):
        (out_dir / stale).unlink(missing_ok=True)
    (out_dir / ".done").write_text(
        json.dumps({"n_rows": len(h), "n_skipped": len(meta) - len(kept),
                    "seconds": round(time.time() - t_length, 1)}) + "\n")

    peak = torch.cuda.max_memory_allocated(device) / 1e9
    toks = [m["n_tokens"] for m in kept]
    secs = [m["seconds"] for m in kept]
    logger.info(
        f"[{length}] DONE rows={len(h)} skipped={len(meta) - len(kept)} "
        f"tok_mean={np.mean(toks):.0f} s/example={np.mean(secs):.2f} "
        f"peak={peak:.2f}GB total={time.time() - t_length:.0f}s")


# ---------------------------------------------------------------------------

def run(model_key, device, lengths, limit, verify, manifest_path=None):
    logger = get_logger(f"lc-extract-{model_key}",
                        RUN_LOGS_DIR / f"longcontext_{model_key}.log")
    manifest = load_manifest(manifest_path)
    requested = [str(x) for x in lengths]
    lengths = supported_lengths(model_key, requested, manifest)
    dropped = [l for l in requested if l not in lengths]
    logger.info(f"===== extract {model_key} on {device} lengths={lengths} "
                f"limit={limit} =====")
    for l in dropped:
        logger.warning(f"DROP length {l}: realised "
                       f"{realised_max_tokens(model_key, l, manifest)} tokens "
                       f"> window {max_positions_for(model_key, l, manifest)}")

    # Group by rope configuration so the model is loaded at most twice.
    groups = [(False, [l for l in lengths if not needs_yarn(model_key, l, manifest)]),
              (True, [l for l in lengths if needs_yarn(model_key, l, manifest)])]

    for yarn, group in groups:
        group = [l for l in group if not (length_dir(model_key, l) / ".done").exists()]
        if not group:
            continue
        logger.info(f"loading {model_key} (yarn={yarn}) ...")
        model, tokenizer = load_model(model_key, device, yarn=yarn)
        try:
            if verify:
                verify_hooks(model, tokenizer, device, logger)
            all_entries = manifest_entries(manifest, lengths=group, limit=limit)
            ensure_hq(model, tokenizer, device, model_key, all_entries, logger)
            for length in group:
                entries = manifest_entries(manifest, lengths=[length], limit=limit)
                extract_length(model, tokenizer, device, model_key, length,
                               entries, logger)
                torch.cuda.empty_cache()
                gc.collect()
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    logger.info(f"===== extract {model_key} complete =====")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--lengths", nargs="+", default=LENGTHS)
    ap.add_argument("--limit", type=int, default=None,
                    help="debug: only the first N questions of the manifest")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--subsample", type=int, default=None,
                    help="keep only N questions (nested seed-0 subset); applies to all lengths in this run")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore .done markers / checkpoints (they are still written)")
    ap.add_argument("--verify-hooks", action="store_true",
                    help="assert hook readout == output_hidden_states[1:] first")
    args = ap.parse_args()
    if args.subsample is not None:
        import longcontext.common as _lc_common
        _lc_common.SUBSAMPLE = args.subsample

    if args.no_resume:
        for length in args.lengths:
            (length_dir(args.model, length) / ".done").unlink(missing_ok=True)
    run(args.model, args.device, args.lengths, args.limit, args.verify_hooks,
        args.manifest)


if __name__ == "__main__":
    main()
