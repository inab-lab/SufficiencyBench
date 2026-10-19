# Retrieved-context k-sensitivity (llama, BM25 top-k, test split, label = support)

Probes are transferred from the SufficiencyBench-trained probes (no retraining on retrieved data); AUROC with 95% bootstrap CI (B=2000, resampling questions; truncated prompts dropped).

- bm25_k3: n = 1500, positive rate = 0.330
- bm25_k5: n = 1500, positive rate = 0.445

| Predictor | k3 | k5 |
|---|---|---|
| Standard probe (transfer) | 0.849 [0.830, 0.869] | 0.807 [0.785, 0.829] |
| CSP probe (transfer) | 0.833 [0.812, 0.853] | 0.795 [0.772, 0.817] |
| Verbalized confidence | 0.729 [0.708, 0.750] | 0.707 [0.684, 0.729] |
| Model answers (no abstention) | 0.733 [0.711, 0.755] | 0.725 [0.702, 0.747] |
| e5 cosine(q, ctx) | 0.767 [0.742, 0.792] | 0.712 [0.687, 0.738] |
