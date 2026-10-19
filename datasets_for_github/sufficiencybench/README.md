# SufficiencyBench, version 1.1

Paired benchmark for predicting whether a retrieved context suffices to answer a question. It is the version used in the paper "Will the Retrieved Context Suffice? Confound-Controlled Evaluation of Sufficiency Predictors for Retrieval-Augmented Generation" (under review at ACM Transactions on Information Systems).

## Files

| File | Content |
|---|---|
| `train.json`, `val.json`, `test.json` | Question records (a JSON list each), split over questions |
| `metadata.json` | Version, split sizes, quadrant and question-type counts, length-ratio statistics, changelog |
| `v1.1_build_report.json` | Ids of the questions dropped when moving from v1.0 to v1.1, and the resulting counts |
| `pk_cache_{llama,mistral,qwen,qwen3,qwen14,deepseek32}.json` | Closed-book (parametric knowledge, PK) generations and labels, one file per evaluated model |

## Splits

2,685 questions, each with a sufficient and an insufficient context (5,370 examples).

| Split | Questions |
|---|---|
| train | 1,352 |
| val | 264 |
| test | 1,069 |

By source: 705 SQuAD v2 (`squad_*`), 1,480 HotpotQA (`hotpot_*`), 500 ELI5 (`eli5_*`). By question type: 705 factual, 1,202 multi-hop, 278 comparative, 500 subjective. Both conditions of a question are always in the same split.

## Record format

Each question is one object:

```json
{
  "id": "squad_133",
  "question": "Who was in charge of the papal army in the War of Barbastro?",
  "gold_answer": "William of Montreuil",
  "question_type": "factual",
  "model_knows_answer": false,
  "closed_book_confidence": 0.92,
  "closed_book_answer": "I don't know. ...",
  "conditions": [
    {
      "condition_id": "squad_133_suf",
      "context": "...",
      "sufficient": true,
      "model_confident": false,
      "quadrant": "Q2",
      "expected_behavior": "answer",
      "context_length": 442,
      "prompt": "Based on the following context, answer the question. ..."
    },
    {
      "condition_id": "squad_133_insuf",
      "context": "...",
      "sufficient": false,
      "model_confident": false,
      "quadrant": "Q4",
      "expected_behavior": "abstain",
      "context_length": 442,
      "prompt": "..."
    }
  ]
}
```

(`context` and `prompt` are shortened here; `closed_book_*` and `model_knows_answer` / `model_confident` in the question files are from the Llama 3.1 8B build model, see the PK note below.)

## Quadrants

Sufficiency is model-agnostic. Parametric knowledge (PK) is per model, from a closed-book run without context.

| Quadrant | Context | PK |
|---|---|---|
| Q1 | sufficient | PK = 1 (the model knows the answer) |
| Q2 | sufficient | PK = 0 |
| Q3 | insufficient | PK = 1 |
| Q4 | insufficient | PK = 0 |

## Construction

Each insufficient context is a minimally edited copy of its sufficient counterpart. The answer-bearing sentence of the gold passage (found from SQuAD v2 gold spans or HotpotQA supporting facts) is removed and replaced by a topically similar, non-answering sentence of matched length retrieved with FAISS and MiniLM-L6 from a different document. Quality filters require that the replacement sentence does not contain the gold answer, a length ratio in [0.5, 2.0] and a cosine similarity of at least 0.5. For questions whose gold answer does not occur literally in a sentence (mostly ELI5), the central sentence of the passage is removed instead. The pairs are therefore matched in length, entity density and topic, and differ only in whether the answer-bearing sentence is present. See Section 3.1 of the paper for details and Section 3.2 for the validity checks.

## Whitespace normalisation (v1.0 to v1.1)

In v1.0 the sufficient contexts kept the double spaces of the source passages while the edited insufficient contexts were re-joined with single spaces, so the number of double spaces was a label cue. In v1.1 every run of whitespace in every context is collapsed to a single space in both conditions and the prompts are rebuilt. 95 questions whose two contexts were identical once whitespace was removed were dropped (the ids are in `v1.1_build_report.json`). Questions, gold answers, ids, splits and PK labels are otherwise unchanged. Version 1.0 is in `../sufficiencybench_v1.0/`.

## Notes on PK labels

- `metadata.json` uses the older wording "Confident" / "Uncertain" in `quadrant_definitions` (Q1 = Sufficient + Confident, Q2 = Sufficient + Uncertain, and so on). "Confident" means PK = 1 and "Uncertain" means PK = 0.
- The `quadrant`, `model_knows_answer` and `model_confident` fields and the `pk_cache_*` files were produced with the rule used while the benchmark was built: the normalised gold answer occurs anywhere in the 60-token closed-book generation and no abstention phrase occurs anywhere in it.
- The PK rates reported in the paper use a first-sentence rule instead (the gold answer occurs in the first sentence of the generation and that sentence contains no abstention phrase), implemented in `scripts/recompute_pk_v2.py` in the code repository. Both label sets, with PK-stratified AUROCs, are in `results/pk_v2/`.
- `metadata.json` field `model_used_for_pk_test` is `llama-3.1-8b-instruct`.
- The `pk_cache_*` files cover different sets of questions (for example `pk_cache_llama.json` includes questions that were not kept in the final benchmark, and `pk_cache_deepseek32.json` covers the test split only).
