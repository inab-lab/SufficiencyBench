# Retrieved-context, E5 dense retriever (e5_k5), support label

Sufficiency rate (all gold titles in top-5), test: llama 0.662 (N=1500); BM25 reference 0.445. Retrieval-level rate by k (test, E5): k=1 0.000, k=3 0.563, k=5 0.662, k=10 0.765 (BM25: 0.000, 0.330, 0.445, 0.587).

AUROC [95% CI], support label.

| Predictor | llama |
|---|---|
| Retriever-score logreg (5 feats) | 0.751 [0.725, 0.776] |
| Retriever top-1 score | 0.558 [0.526, 0.590] |
| E5 cosine(q, ctx) | 0.652 [0.623, 0.681] |
| Verbalized confidence | 0.689 [0.665, 0.713] |
| Model answers (no abstention) | 0.696 [0.672, 0.721] |
| Standard probe (transfer) | 0.801 [0.777, 0.826] |
| CSP probe (transfer) | 0.798 [0.773, 0.823] |
| Standard probe (retrained) | 0.922 [0.906, 0.935] |
| CSP probe (retrained) | 0.924 [0.909, 0.938] |

Paired delta AUROC of transferred probes (95% CI; paired bootstrap, B=2000).

| Comparison | llama |
|---|---|
| Standard (transfer) - VC | +0.113 [+0.086, +0.140], p=0.000 |
| CSP (transfer) - VC | +0.109 [+0.082, +0.136], p=0.000 |
| Standard (transfer) - E5 cosine | +0.149 [+0.111, +0.188], Holm p=0.0040 |
| CSP (transfer) - E5 cosine | +0.145 [+0.107, +0.185], Holm p=0.0040 |
