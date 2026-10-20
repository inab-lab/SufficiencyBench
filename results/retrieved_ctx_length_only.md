# Length-only baseline on retrieved contexts (Llama-side labels; CPU only)

Label = support-sufficiency (`sufficient` in data/experiments/retrieved_ctx/<tag>/test.json). Bootstrap: 2,000 question resamples, seed 0, percentile 95% CI. "chars"/"words" = retrieved context text inside the generator prompt (BM25 prompt includes "Title: " prefixes). "prompt tokens" = Llama tokenizer length of the entire prompt (template+context+question) from test_scores.json.


## bm25_k5: n_test=1500, sufficient=44.5%, train n=3000 (42.8% sufficient)

| Predictor | AUROC [95% CI] | direction | mean suff. | mean insuff. |
|---|---|---|---|---|
| chars | 0.577 [0.549, 0.607] | longer context => LESS sufficient (AUROC reported as 1-AUC) | 2731.0 | 2959.1 |
| words | 0.584 [0.557, 0.614] | longer context => LESS sufficient (AUROC reported as 1-AUC) | 442.6 | 483.3 |
| prompt_tokens(llama, whole prompt) | 0.573 [0.546, 0.603] | longer context => LESS sufficient (AUROC reported as 1-AUC) | 688.1 | 736.3 |
| LogReg on log(chars), fit on train | 0.577 [0.549, 0.607] | coef=-0.955 | | |

## e5_k5: n_test=1500, sufficient=66.2%, train n=3000 (66.5% sufficient)

| Predictor | AUROC [95% CI] | direction | mean suff. | mean insuff. |
|---|---|---|---|---|
| chars | 0.575 [0.545, 0.604] | longer context => LESS sufficient (AUROC reported as 1-AUC) | 2387.0 | 2575.6 |
| words | 0.583 [0.552, 0.611] | longer context => LESS sufficient (AUROC reported as 1-AUC) | 386.1 | 420.0 |
| prompt_tokens(llama, whole prompt) | 0.577 [0.547, 0.606] | longer context => LESS sufficient (AUROC reported as 1-AUC) | 610.8 | 651.8 |
| LogReg on log(chars), fit on train | 0.575 [0.545, 0.604] | coef=-0.749 | | |
