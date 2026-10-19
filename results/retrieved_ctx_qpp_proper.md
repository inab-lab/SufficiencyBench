Retrieved-context bm25_k5, support label. Test N=1500, train N=3000. 95% bootstrap CI over questions (B=2000, seed 0).

| Predictor | k=5 | k=10 | k=20 |
|---|---|---|---|
| WIG | 0.565 [0.536, 0.593] | 0.545 [0.515, 0.573] | 0.531 [0.501, 0.560] |
| NQC | 0.586 [0.558, 0.615] | 0.599 [0.571, 0.629] | 0.587 [0.558, 0.616] |
| SMV | 0.590 [0.562, 0.620] | 0.614 [0.584, 0.643] | 0.592 [0.563, 0.620] |
| Clarity | 0.475 [0.445, 0.504] | 0.485 [0.455, 0.515] | 0.493 [0.463, 0.523] |
| sigma_max | 0.575 [0.546, 0.604] | 0.580 [0.549, 0.608] | 0.567 [0.536, 0.596] |
| raw top-k score std (reference) | 0.584 [0.555, 0.612] | 0.602 [0.572, 0.631] | 0.594 [0.564, 0.623] |
| Logreg over WIG, NQC, SMV, Clarity, sigma_max (fit on train) | 0.593 [0.564, 0.622] | 0.655 [0.626, 0.683] | 0.594 [0.563, 0.624] |

Best k for the logreg by 5-fold train CV AUROC (k=5: 0.606, k=10: 0.666, k=20: 0.613): k=10. Sanity: raw std@5 = 0.5837 (paper 0.584); paper 5-feature raw logreg = 0.6711 (paper 0.671).

### Paired Delta AUROC: transferred Standard probe minus QPP predictor, llama (N=1500, probe AUROC 0.807)

| Comparison | Delta | 95% CI | p | Holm p |
|---|---|---|---|---|
| Standard (transfer) vs QPP logreg k=5 | +0.214 | [+0.181, +0.249] | 0.0005 | 0.0090 |
| Standard (transfer) vs QPP logreg k=10 | +0.152 | [+0.120, +0.186] | 0.0005 | 0.0090 |
| Standard (transfer) vs WIG@5 | +0.242 | [+0.207, +0.276] | 0.0005 | 0.0090 |
| Standard (transfer) vs WIG@10 | +0.262 | [+0.227, +0.297] | 0.0005 | 0.0090 |
| Standard (transfer) vs WIG@20 | +0.276 | [+0.241, +0.311] | 0.0005 | 0.0090 |
| Standard (transfer) vs NQC@5 | +0.221 | [+0.185, +0.256] | 0.0005 | 0.0090 |
| Standard (transfer) vs NQC@10 | +0.208 | [+0.173, +0.243] | 0.0005 | 0.0090 |
| Standard (transfer) vs NQC@20 | +0.220 | [+0.185, +0.254] | 0.0005 | 0.0090 |
| Standard (transfer) vs SMV@5 | +0.217 | [+0.181, +0.253] | 0.0005 | 0.0090 |
| Standard (transfer) vs SMV@10 | +0.194 | [+0.160, +0.229] | 0.0005 | 0.0090 |
| Standard (transfer) vs SMV@20 | +0.215 | [+0.180, +0.249] | 0.0005 | 0.0090 |
| Standard (transfer) vs Clarity@5 | +0.333 | [+0.296, +0.369] | 0.0005 | 0.0090 |
| Standard (transfer) vs Clarity@10 | +0.322 | [+0.286, +0.358] | 0.0005 | 0.0090 |
| Standard (transfer) vs Clarity@20 | +0.314 | [+0.278, +0.350] | 0.0005 | 0.0090 |
| Standard (transfer) vs sigma_max@5 | +0.232 | [+0.198, +0.268] | 0.0005 | 0.0090 |
| Standard (transfer) vs sigma_max@10 | +0.228 | [+0.194, +0.263] | 0.0005 | 0.0090 |
| Standard (transfer) vs sigma_max@20 | +0.240 | [+0.206, +0.275] | 0.0005 | 0.0090 |
| Standard (transfer) vs raw std@5 | +0.223 | [+0.190, +0.259] | 0.0005 | 0.0090 |

### Paired Delta AUROC: transferred Standard probe minus QPP predictor, mistral (N=1500, probe AUROC 0.773)

| Comparison | Delta | 95% CI | p | Holm p |
|---|---|---|---|---|
| Standard (transfer) vs QPP logreg k=5 | +0.180 | [+0.145, +0.217] | 0.0005 | 0.0090 |
| Standard (transfer) vs QPP logreg k=10 | +0.118 | [+0.084, +0.153] | 0.0005 | 0.0090 |
| Standard (transfer) vs WIG@5 | +0.207 | [+0.173, +0.243] | 0.0005 | 0.0090 |
| Standard (transfer) vs WIG@10 | +0.228 | [+0.193, +0.263] | 0.0005 | 0.0090 |
| Standard (transfer) vs WIG@20 | +0.241 | [+0.206, +0.276] | 0.0005 | 0.0090 |
| Standard (transfer) vs NQC@5 | +0.186 | [+0.149, +0.223] | 0.0005 | 0.0090 |
| Standard (transfer) vs NQC@10 | +0.173 | [+0.137, +0.210] | 0.0005 | 0.0090 |
| Standard (transfer) vs NQC@20 | +0.185 | [+0.149, +0.222] | 0.0005 | 0.0090 |
| Standard (transfer) vs SMV@5 | +0.182 | [+0.146, +0.219] | 0.0005 | 0.0090 |
| Standard (transfer) vs SMV@10 | +0.159 | [+0.122, +0.196] | 0.0005 | 0.0090 |
| Standard (transfer) vs SMV@20 | +0.181 | [+0.145, +0.217] | 0.0005 | 0.0090 |
| Standard (transfer) vs Clarity@5 | +0.298 | [+0.259, +0.337] | 0.0005 | 0.0090 |
| Standard (transfer) vs Clarity@10 | +0.287 | [+0.247, +0.326] | 0.0005 | 0.0090 |
| Standard (transfer) vs Clarity@20 | +0.279 | [+0.239, +0.318] | 0.0005 | 0.0090 |
| Standard (transfer) vs sigma_max@5 | +0.197 | [+0.161, +0.238] | 0.0005 | 0.0090 |
| Standard (transfer) vs sigma_max@10 | +0.193 | [+0.157, +0.233] | 0.0005 | 0.0090 |
| Standard (transfer) vs sigma_max@20 | +0.206 | [+0.169, +0.246] | 0.0005 | 0.0090 |
| Standard (transfer) vs raw std@5 | +0.189 | [+0.152, +0.229] | 0.0005 | 0.0090 |

### Paired Delta AUROC: transferred Standard probe minus QPP predictor, qwen (N=1500, probe AUROC 0.810)

| Comparison | Delta | 95% CI | p | Holm p |
|---|---|---|---|---|
| Standard (transfer) vs QPP logreg k=5 | +0.218 | [+0.183, +0.251] | 0.0005 | 0.0090 |
| Standard (transfer) vs QPP logreg k=10 | +0.156 | [+0.121, +0.190] | 0.0005 | 0.0090 |
| Standard (transfer) vs WIG@5 | +0.245 | [+0.211, +0.279] | 0.0005 | 0.0090 |
| Standard (transfer) vs WIG@10 | +0.266 | [+0.231, +0.301] | 0.0005 | 0.0090 |
| Standard (transfer) vs WIG@20 | +0.279 | [+0.244, +0.315] | 0.0005 | 0.0090 |
| Standard (transfer) vs NQC@5 | +0.224 | [+0.189, +0.260] | 0.0005 | 0.0090 |
| Standard (transfer) vs NQC@10 | +0.211 | [+0.175, +0.245] | 0.0005 | 0.0090 |
| Standard (transfer) vs NQC@20 | +0.223 | [+0.187, +0.258] | 0.0005 | 0.0090 |
| Standard (transfer) vs SMV@5 | +0.220 | [+0.185, +0.255] | 0.0005 | 0.0090 |
| Standard (transfer) vs SMV@10 | +0.197 | [+0.161, +0.231] | 0.0005 | 0.0090 |
| Standard (transfer) vs SMV@20 | +0.218 | [+0.182, +0.252] | 0.0005 | 0.0090 |
| Standard (transfer) vs Clarity@5 | +0.336 | [+0.298, +0.375] | 0.0005 | 0.0090 |
| Standard (transfer) vs Clarity@10 | +0.325 | [+0.287, +0.364] | 0.0005 | 0.0090 |
| Standard (transfer) vs Clarity@20 | +0.317 | [+0.279, +0.356] | 0.0005 | 0.0090 |
| Standard (transfer) vs sigma_max@5 | +0.235 | [+0.201, +0.270] | 0.0005 | 0.0090 |
| Standard (transfer) vs sigma_max@10 | +0.231 | [+0.197, +0.265] | 0.0005 | 0.0090 |
| Standard (transfer) vs sigma_max@20 | +0.244 | [+0.209, +0.278] | 0.0005 | 0.0090 |
| Standard (transfer) vs raw std@5 | +0.227 | [+0.193, +0.262] | 0.0005 | 0.0090 |

p = 2*frac(delta<=0) (floor 1/B), Holm over all comparisons within a model.
