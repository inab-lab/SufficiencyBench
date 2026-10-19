"""
common.py — shared pieces for the SufficiencyBench long-context extension.

The long-context experiment asks: does a sufficiency probe that works on the
short (~200-token) SufficiencyBench contexts still work when the *same* target
passage is buried inside 4k / 16k / 32k / 64k / 100k tokens of distractors?

Nothing here touches the GPU.  It owns:
  * the length / position grid,
  * the prompt templates (identical to extraction/extract_states.py and to the
    verbalized-confidence suffix in methods/baselines.py),
  * the benchmark index (condition_id -> context) and
  * ``assemble_context`` / ``build_prompt``, which rebuild a full long context
    deterministically from a manifest entry -- so the manifest only has to store
    condition ids, never the (huge) strings themselves.
"""

import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))          # project root (configs/)
sys.path.insert(0, str(_HERE.parents[1]))          # src/ (extraction/, methods/)

from configs.paths import BENCH_DIR, EXPERIMENTS_DIR, RESULTS_DIR, MODEL_CONFIGS  # noqa: E402

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

# "base" = the original short context alone (the reference point: this is what
# every number in the main paper was measured on).
LENGTHS = ["base", "4000", "16000", "32000", "64000", "100000"]

#: target passage placement inside the distractor sequence
POSITIONS = [0.0, 0.5, 1.0]
POSITION_LABELS = {0.0: "first", 0.5: "middle", 1.0: "last"}

#: passages (distractors + target) are glued with a blank line
PASSAGE_SEP = "\n\n"

#: relative tolerance on the realised context length vs. its nominal target
LENGTH_TOLERANCE = 0.05

N_QUESTIONS = 300
BUILD_SEED = 0

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

LONGCONTEXT_DIR = EXPERIMENTS_DIR / "longcontext"
MANIFEST_PATH = LONGCONTEXT_DIR / "manifest.json"
LC_RESULTS_DIR = RESULTS_DIR / "longcontext"
RUN_LOGS_DIR = _HERE.parents[2] / "run_logs"

#: tokenizer used to *construct* the sets (one common yardstick for all models)
BUILD_TOKENIZER = "llama"


def length_dir(model_key: str, length: str) -> Path:
    return LONGCONTEXT_DIR / model_key / str(length)


def hq_path(model_key: str) -> Path:
    """h(q) is context-independent, so it is cached once per model."""
    return LONGCONTEXT_DIR / model_key / "h_question_only.npy"


def hq_index_path(model_key: str) -> Path:
    return LONGCONTEXT_DIR / model_key / "hq_index.json"


# ---------------------------------------------------------------------------
# Context-window limits
# ---------------------------------------------------------------------------

#: native max_position_embeddings per backbone
MAX_POSITIONS = {"llama": 131072, "mistral": 32768, "qwen": 32768}

#: Qwen2.5 ships with rope_scaling=None and a 32k window.  For the >32k lengths
#: we re-load it with YaRN (factor 4 -> 131072).  Applied *only* to those
#: lengths so the <=32k numbers stay on the stock configuration.
QWEN_YARN_ROPE = {
    "type": "yarn",
    "rope_type": "yarn",          # transformers >=4.45 spelling; both accepted
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
}
QWEN_YARN_MAX_POSITIONS = 131072


#: cached per-model realised prompt lengths, see realised_max_tokens()
TOKEN_BUDGET_PATH = LONGCONTEXT_DIR / "token_budget.json"

#: how many of the longest (by llama tokens) entries to measure per length
_BUDGET_PROBE_K = 5
#: safety factor on the measured maximum
_BUDGET_MARGIN = 1.01


def realised_max_tokens(model_key: str, length: str, manifest=None) -> int:
    """Upper bound on the prompt length THIS model's tokenizer produces at a
    nominal (llama-token) length.

    The sets are built once with the llama tokenizer, but the other backbones do
    not agree with it: Mistral v0.3 emits ~15% more tokens for the same English
    text and Qwen2.5 ~4% more.  So a "32000" prompt is ~36.7k Mistral tokens and
    ~33.3k Qwen tokens -- both past their 32768 window.  Window decisions
    therefore have to be made on the realised count, not the nominal label.

    Measured on the ``_BUDGET_PROBE_K`` longest entries of that length (the
    spread within a length is ~100 tokens) and cached on disk; the per-example
    guard in extraction is still exact.
    """
    key = f"{model_key}/{length}"
    cache = {}
    if TOKEN_BUDGET_PATH.exists():
        try:
            cache = json.load(open(TOKEN_BUDGET_PATH))
        except Exception:  # noqa: BLE001
            cache = {}
    if key in cache:
        return int(cache[key])

    if manifest is None:
        manifest = load_manifest()
    entries = [e for e in manifest["entries"] if e["length"] == str(length)]
    entries.sort(key=lambda e: -max(e["n_prompt_tokens"].values()))
    tok = load_tokenizer(model_key)
    worst = 0
    for entry in entries[:_BUDGET_PROBE_K]:
        for cond in ("sufficient", "insufficient"):
            worst = max(worst, len(tok(build_prompt(entry, cond))["input_ids"]))
    worst = int(worst * _BUDGET_MARGIN)

    cache[key] = worst
    atomic_write_json(TOKEN_BUDGET_PATH, cache)
    return worst


def needs_yarn(model_key: str, length: str, manifest=None) -> bool:
    """True when this (model, length) pair must be run with YaRN rope scaling.

    Only Qwen2.5 has a YaRN path here (Mistral v0.3 exposes no rope scaling).
    """
    if model_key != "qwen" or str(length) == "base":
        return False
    return realised_max_tokens(model_key, length, manifest) > MAX_POSITIONS["qwen"]


def max_positions_for(model_key: str, length: str, manifest=None) -> int:
    if needs_yarn(model_key, length, manifest):
        return QWEN_YARN_MAX_POSITIONS
    return MAX_POSITIONS[model_key]


def supported_lengths(model_key: str, lengths=None, manifest=None):
    """Drop lengths whose realised prompt does not fit the backbone's window.

    With the current sets that removes 32000/64000/100000 for Mistral (no rope
    scaling, and 32000 nominal is already ~36.7k Mistral tokens) and nothing for
    Llama or, with YaRN, Qwen.
    """
    out = []
    for ln in (lengths if lengths is not None else LENGTHS):
        ln = str(ln)
        if ln == "base":
            out.append(ln)
            continue
        if realised_max_tokens(model_key, ln, manifest) > \
                max_positions_for(model_key, ln, manifest):
            continue
        out.append(ln)
    return out


# ---------------------------------------------------------------------------
# Prompts (verbatim from extraction/extract_states.py + methods/baselines.py)
# ---------------------------------------------------------------------------

def format_with_context(context: str, question: str) -> str:
    """The SufficiencyBench RAG prompt -- byte-identical to the cached
    ``condition["prompt"]`` fields and to extract_states.py."""
    return (
        f"Based on the following context, answer the question. "
        f"If the context does not contain enough information, "
        f"say 'I cannot answer this from the provided context.'\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )


# h(q) baseline prompt; imported so it can never drift from the short-context run.
from extraction.extract_states import format_question_only  # noqa: E402,F401

#: verbalized-confidence suffix, verbatim from methods/baselines.py
VC_SUFFIX = (
    "\n\nBefore answering, rate from 0 to 100: "
    "does the context contain enough information to answer? "
    "Reply with ONLY the number.\nScore:"
)


def parse_vc_score(response: str) -> float:
    """First integer in the completion, /100, clamped; 0.5 if none found.
    Mirrors methods/baselines.verbalized_confidence_batch."""
    try:
        numbers = re.findall(r"\d+", response)
        if numbers:
            return min(max(float(numbers[0]) / 100, 0.0), 1.0)
        return 0.5
    except (ValueError, IndexError):
        return 0.5


# ---------------------------------------------------------------------------
# Benchmark index
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_split(split: str):
    with open(BENCH_DIR / f"{split}.json") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def condition_index():
    """condition_id -> {context, question, question_type, gold_answer,
    sufficient, question_id, split} over train+val+test."""
    idx = {}
    for split in ("train", "val", "test"):
        path = BENCH_DIR / f"{split}.json"
        if not path.exists():
            continue
        for ex in load_split(split):
            for cond in ex["conditions"]:
                idx[cond["condition_id"]] = {
                    "context": cond["context"],
                    "sufficient": bool(cond["sufficient"]),
                    "question": ex["question"],
                    "question_id": ex["id"],
                    "question_type": ex["question_type"],
                    "gold_answer": ex.get("gold_answer", ""),
                    "split": split,
                }
    return idx


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def load_manifest(path: Path = None) -> dict:
    with open(path or MANIFEST_PATH) as f:
        return json.load(f)


#: When set (int), manifest_entries() keeps only this many questions, chosen by a
#: seed-0 shuffle of the sorted question ids, so subsets of different sizes are nested.
SUBSAMPLE = None


def subsample_question_ids(manifest: dict, n: int, seed: int = 0) -> set:
    import random
    qids = sorted({e["question_id"] for e in manifest["entries"]})
    random.Random(seed).shuffle(qids)
    return set(qids[:n])


def manifest_entries(manifest: dict, lengths=None, limit: int = None):
    """Entries for the requested lengths, in manifest (question) order.

    ``limit`` caps the number of *questions* (not entries), so a debug run keeps
    whole questions and all their lengths.
    """
    wanted = None if lengths is None else {str(x) for x in lengths}
    keep_q = None
    if SUBSAMPLE is not None:
        keep_q = subsample_question_ids(manifest, SUBSAMPLE)
    if limit is not None:
        seen = []
        for e in manifest["entries"]:
            if e["question_id"] not in seen:
                seen.append(e["question_id"])
            if len(seen) >= limit:
                break
        keep_q = set(seen)
    out = []
    for e in manifest["entries"]:
        if wanted is not None and e["length"] not in wanted:
            continue
        if keep_q is not None and e["question_id"] not in keep_q:
            continue
        out.append(e)
    return out


def resolve_condition(entry: dict, condition) -> str:
    """Accept 'sufficient'/'insufficient', True/False, or a raw condition_id."""
    if isinstance(condition, bool):
        condition = "sufficient" if condition else "insufficient"
    if condition in entry["condition_ids"]:
        return entry["condition_ids"][condition]
    if condition in entry["condition_ids"].values():
        return condition
    raise KeyError(f"unknown condition {condition!r} for {entry['question_id']}")


def assemble_context(manifest_entry: dict, condition) -> str:
    """Rebuild the full (possibly 100k-token) context for one manifest entry.

    The distractor sequence is identical for the sufficient and insufficient
    conditions of a question at a given length -- only the target passage that
    is spliced in at ``insert_index`` changes.
    """
    idx = condition_index()
    target_id = resolve_condition(manifest_entry, condition)
    passages = [idx[cid]["context"] for cid in manifest_entry["distractors"]]
    passages.insert(manifest_entry["insert_index"], idx[target_id]["context"])
    return PASSAGE_SEP.join(passages)


def build_prompt(manifest_entry: dict, condition) -> str:
    """assemble_context + the SufficiencyBench prompt template."""
    ctx = assemble_context(manifest_entry, condition)
    return format_with_context(ctx, manifest_entry["question"])


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def load_tokenizer(model_key: str = BUILD_TOKENIZER):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_CONFIGS[model_key]["path"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def get_logger(name: str, logfile: Path):
    """stdout + append-to-file logger (one file per model, as the run scripts
    expect in run_logs/)."""
    import logging
    logfile.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(logfile)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)
