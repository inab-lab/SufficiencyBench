"""
vc_longcontext.py — verbalized confidence at long context.

The behavioural baseline that the probes are compared against: append the
methods/baselines.py sufficiency-rating suffix to the (long) RAG prompt, greedy-
decode 5 tokens, take the first integer /100.  Exactly the same prompt, decoding
and parsing as the short-context run -- only the context length changes.

Differences from methods/baselines.py, forced by the context length:
  * batch size 1 (a 100k-token batch of 16 is not representable), so no padding;
  * NO truncation (baselines.py truncates at 2048 -- that would defeat the whole
    experiment);
  * 8-bit weights by default instead of fp16: at 100k tokens the KV cache alone
    is ~13 GB, which does not fit alongside 16 GB of fp16 weights on a 32 GB
    card.  ``--precision fp16`` restores the baselines.py loader.

Output: data/experiments/longcontext/<model>/<length>/vc_scores.json
        {condition_id: {"score": float, "raw": str, "n_tokens": int, ...}}
A length that OOMs is abandoned (vc_oom.txt is written) and the run continues
with the next length; that length is NOT retried on a re-run -- delete its
vc_oom.txt to try it again (e.g. with a smaller --prefill-chunk).

Usage:
  python src/longcontext/vc_longcontext.py --model llama --device cuda:1
  python src/longcontext/vc_longcontext.py --model llama --device cuda:1 \
      --limit 2 --lengths base 4000
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longcontext.common import (  # noqa: E402
    LENGTHS, MODEL_CONFIGS, RUN_LOGS_DIR, VC_SUFFIX, atomic_write_json,
    build_prompt, get_logger, length_dir, load_manifest, manifest_entries,
    max_positions_for, needs_yarn, parse_vc_score, supported_lengths,
)
from longcontext.extract_longcontext import load_model  # noqa: E402

CHECKPOINT_EVERY = 20
CONDITIONS = ("sufficient", "insufficient")
MAX_NEW_TOKENS = 5

#: Above this prompt length, prefill is CHUNKED (see prefill_chunked).  A
#: single-shot 100k prefill needs the ~18 GB of transient activations that the
#: hidden-state extraction peaks at PLUS the ~13 GB KV cache that generation
#: must retain, which does not fit in 32 GB -- measured: llama OOMs at 100k.
#: Feeding the prompt in chunks caps the transient term at one chunk's worth and
#: is mathematically identical (causal attention + RoPE are position-absolute).
CHUNK_THRESHOLD = 32768
PREFILL_CHUNK = 8192


def load_model_fp16(model_key, device):
    """The methods/baselines.py loader, kept as an opt-in alternative."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    cfg = MODEL_CONFIGS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.float16, device_map=device,
        attn_implementation="sdpa",
    )
    model.eval()
    return model, tokenizer


def _greedy_chunked(model, tokenizer, input_ids, max_new_tokens, chunk):
    """Greedy decode with a CHUNKED prefill, for prompts too long to prefill in
    one shot.  Equivalent to model.generate(do_sample=False): the KV cache is
    built chunk by chunk (causal attention means chunk k only ever attends to
    chunks <= k, which the cache already holds), then tokens are taken by argmax
    one at a time.  ``--validate-chunking`` checks it against generate().
    """
    from transformers import DynamicCache

    past = DynamicCache()
    n = input_ids.shape[1]
    out = None
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        out = model(
            input_ids=input_ids[:, start:end],
            attention_mask=torch.ones((1, end), dtype=torch.long,
                                      device=input_ids.device),
            past_key_values=past,
            use_cache=True,
            num_logits_to_keep=1,
        )
        past = out.past_key_values

    generated = []
    logits = out.logits[:, -1, :]
    for _ in range(max_new_tokens):
        nxt = int(logits.argmax(dim=-1).item())
        if nxt == tokenizer.eos_token_id:
            break
        generated.append(nxt)
        n += 1
        out = model(
            input_ids=torch.tensor([[nxt]], device=input_ids.device),
            attention_mask=torch.ones((1, n), dtype=torch.long,
                                      device=input_ids.device),
            past_key_values=past,
            use_cache=True,
            num_logits_to_keep=1,
        )
        past = out.past_key_values
        logits = out.logits[:, -1, :]
    del past, out
    return generated


def score_one(model, tokenizer, prompt, device, chunk_threshold=CHUNK_THRESHOLD,
              chunk=PREFILL_CHUNK):
    """Greedy 5-token completion of prompt+VC_SUFFIX -> (score, raw, n_tokens)."""
    meta_prompt = prompt + VC_SUFFIX
    inputs = tokenizer(meta_prompt, return_tensors="pt")
    n_tokens = int(inputs["input_ids"].shape[1])
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        if chunk_threshold and n_tokens > chunk_threshold:
            gen = _greedy_chunked(model, tokenizer, inputs["input_ids"],
                                  MAX_NEW_TOKENS, chunk)
        else:
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen = out[0, inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return parse_vc_score(raw), raw, n_tokens


def validate_chunking(model, tokenizer, device, entries, logger, chunk=PREFILL_CHUNK):
    """Prove the chunked path matches generate() where BOTH fit in memory."""
    n_ok = 0
    for entry in entries[:4]:
        for cond in CONDITIONS:
            ref = score_one(model, tokenizer, build_prompt(entry, cond), device,
                            chunk_threshold=0)
            chk = score_one(model, tokenizer, build_prompt(entry, cond), device,
                            chunk_threshold=1, chunk=chunk)
            ok = ref[0] == chk[0]              # the parsed 0-100 score must match
            same_text = ref[1] == chk[1]       # trailing filler tokens may differ under int8 matmul noise
            n_ok += ok
            logger.info(f"[validate-chunking] {entry['condition_ids'][cond]} "
                        f"tok={ref[2]} generate={ref[1]!r} chunked={chk[1]!r} "
                        f"[{'OK' if same_text else ('SCORE-OK/text-differs' if ok else 'MISMATCH')}]")
            assert ok, "chunked prefill changed the parsed VC score vs generate()"
    logger.info(f"[validate-chunking] PASS on {n_ok} prompts (chunk={chunk})")


def score_length(model, tokenizer, device, model_key, length, entries, logger,
                 chunk_threshold=CHUNK_THRESHOLD, chunk=PREFILL_CHUNK):
    out_dir = length_dir(model_key, length)
    out_path = out_dir / _vc_filename()
    scores = {}
    if out_path.exists():
        try:
            scores = json.load(open(out_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{length}] unreadable vc_scores.json ({exc}) — restarting")

    plan = [(e, c) for e in entries for c in CONDITIONS]
    todo = [(e, c) for e, c in plan if e["condition_ids"][c] not in scores]
    if not todo:
        logger.info(f"[{length}] VC already complete ({len(scores)} scores)")
        return
    logger.info(f"[{length}] VC: {len(todo)}/{len(plan)} to score")

    limit = max_positions_for(model_key, length)
    torch.cuda.reset_peak_memory_stats(device)
    t_length = time.time()

    for i, (entry, cond) in enumerate(todo):
        cid = entry["condition_ids"][cond]
        t0 = time.time()
        try:
            score, raw, n_tokens = score_one(
                model, tokenizer, build_prompt(entry, cond), device,
                chunk_threshold, chunk)
        except torch.cuda.OutOfMemoryError as exc:
            logger.error(f"[{length}] VC OOM on {cid} — abandoning this length")
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "vc_oom.txt").write_text(
                f"{cid}\n{exc}\npeak_gb={torch.cuda.max_memory_allocated(device) / 1e9:.2f}\n")
            if scores:
                atomic_write_json(out_path, scores)
            gc.collect()
            torch.cuda.empty_cache()
            return
        dt = time.time() - t0

        if n_tokens > limit:
            logger.warning(f"[{length}] {cid}: {n_tokens} tokens > max_positions "
                           f"{limit} — score may be unreliable")
        scores[cid] = {
            "score": score,
            "raw": raw,
            "question_id": entry["question_id"],
            "question_type": entry["question_type"],
            "position": entry["position"],
            "sufficient": cond == "sufficient",
            "n_tokens": n_tokens,
            "seconds": round(dt, 3),
        }
        logger.info(f"[{length}] VC {i + 1}/{len(todo)} {cid} tok={n_tokens} "
                    f"score={score:.2f} raw={raw!r} {dt:.2f}s "
                    f"peak={torch.cuda.max_memory_allocated(device) / 1e9:.2f}GB")

        if (i + 1) % CHECKPOINT_EVERY == 0:
            atomic_write_json(out_path, scores)

    atomic_write_json(out_path, scores)
    logger.info(f"[{length}] VC DONE n={len(scores)} "
                f"peak={torch.cuda.max_memory_allocated(device) / 1e9:.2f}GB "
                f"total={time.time() - t_length:.0f}s")


def run(model_key, device, lengths, limit, precision, manifest_path=None,
        chunk_threshold=CHUNK_THRESHOLD, chunk=PREFILL_CHUNK, validate=False):
    logger = get_logger(f"lc-vc-{model_key}",
                        RUN_LOGS_DIR / f"longcontext_{model_key}.log")
    manifest = load_manifest(manifest_path)
    lengths = supported_lengths(model_key, [str(x) for x in lengths], manifest)
    logger.info(f"===== VC {model_key} on {device} lengths={lengths} "
                f"limit={limit} precision={precision} "
                f"chunk={chunk}@>{chunk_threshold} =====")

    groups = [(False, [l for l in lengths if not needs_yarn(model_key, l, manifest)]),
              (True, [l for l in lengths if needs_yarn(model_key, l, manifest)])]

    for yarn, group in groups:
        group = [l for l in group
                 if not (length_dir(model_key, l) / _vc_filename()).exists()
                 or _incomplete(model_key, l, manifest, limit)]
        if not group:
            continue
        logger.info(f"loading {model_key} (yarn={yarn}, {precision}) ...")
        if precision == "fp16":
            if yarn:
                logger.warning("fp16 loader does not apply YaRN — skipping >32k")
                continue
            model, tokenizer = load_model_fp16(model_key, device)
        else:
            model, tokenizer = load_model(model_key, device, yarn=yarn)
        try:
            if validate:
                validate_chunking(
                    model, tokenizer, device,
                    manifest_entries(manifest, lengths=["4000"], limit=2),
                    logger, chunk)
            for length in group:
                entries = manifest_entries(manifest, lengths=[length], limit=limit)
                score_length(model, tokenizer, device, model_key, length,
                             entries, logger, chunk_threshold, chunk)
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    logger.info(f"===== VC {model_key} complete =====")


#: optional suffix for the VC output file (used by the chunked-vs-generate agreement study)
OUT_TAG = None


def _vc_filename():
    return "vc_scores.json" if not OUT_TAG else f"vc_scores_{OUT_TAG}.json"


def _incomplete(model_key, length, manifest, limit):
    path = length_dir(model_key, length) / _vc_filename()
    if not path.exists():
        return True
    if (length_dir(model_key, length) / "vc_oom.txt").exists():
        return False  # this length already OOMed; do not retry automatically
    try:
        scores = json.load(open(path))
    except Exception:  # noqa: BLE001
        return True
    entries = manifest_entries(manifest, lengths=[length], limit=limit)
    return any(e["condition_ids"][c] not in scores
               for e in entries for c in CONDITIONS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--lengths", nargs="+", default=LENGTHS)
    ap.add_argument("--limit", type=int, default=None,
                    help="debug: only the first N questions of the manifest")
    ap.add_argument("--precision", default="8bit", choices=["8bit", "fp16"])
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--tag", default=None, help="write vc_scores_<tag>.json instead of vc_scores.json")
    ap.add_argument("--subsample", type=int, default=None,
                    help="keep only N questions (nested seed-0 subset); applies to all lengths in this run")
    ap.add_argument("--chunk-threshold", type=int, default=CHUNK_THRESHOLD,
                    help="prompts longer than this are prefilled in chunks "
                         "(0 = never chunk, i.e. plain generate())")
    ap.add_argument("--prefill-chunk", type=int, default=PREFILL_CHUNK)
    ap.add_argument("--validate-chunking", action="store_true",
                    help="assert chunked prefill == generate() on short prompts")
    args = ap.parse_args()
    if args.tag:
        global OUT_TAG
        OUT_TAG = args.tag
    if args.subsample is not None:
        import longcontext.common as _lc_common
        _lc_common.SUBSAMPLE = args.subsample
    run(args.model, args.device, args.lengths, args.limit, args.precision,
        args.manifest, args.chunk_threshold, args.prefill_chunk,
        args.validate_chunking)


if __name__ == "__main__":
    main()
