Paired bootstrap over questions (B=2000, seed 0). p = 2*frac(delta<=0), floor 1/B. Holm within the 8 probe-vs-baseline comparisons per model; baseline-vs-VC rows use p = 2*min(frac<=0, frac>=0) (direction not assumed, as in analysis.md) and are not Holm-adjusted.

### llama (bm25_k5, support label, N=1500, B=2000)

AUROC: Standard (transfer) 0.807, Standard (retrained) 0.936, CSP (transfer) 0.795, CSP (retrained) 0.924, QPP logreg 0.671, E5 cosine 0.712, VC 0.707

| Comparison | Delta AUROC | 95% CI | p | Holm p |
|---|---|---|---|---|
| Standard (transfer) vs E5 cosine | +0.096 | [+0.063, +0.127] | 0.0005 | 0.0040 |
| CSP (transfer) vs E5 cosine | +0.083 | [+0.050, +0.114] | 0.0005 | 0.0040 |
| Standard (retrained) vs E5 cosine | +0.224 | [+0.197, +0.248] | 0.0005 | 0.0040 |
| CSP (retrained) vs E5 cosine | +0.213 | [+0.185, +0.238] | 0.0005 | 0.0040 |
| Standard (transfer) vs QPP logreg | +0.136 | [+0.104, +0.171] | 0.0005 | 0.0040 |
| CSP (transfer) vs QPP logreg | +0.124 | [+0.089, +0.158] | 0.0005 | 0.0040 |
| Standard (retrained) vs QPP logreg | +0.265 | [+0.236, +0.293] | 0.0005 | 0.0040 |
| CSP (retrained) vs QPP logreg | +0.253 | [+0.224, +0.282] | 0.0005 | 0.0040 |
| E5 cosine vs VC | +0.005 | [-0.026, +0.037] | 0.7430 | -- |
| QPP logreg vs VC | -0.036 | [-0.069, -0.001] | 0.0410 | -- |

### mistral (bm25_k5, support label, N=1500, B=2000)

AUROC: Standard (transfer) 0.773, Standard (retrained) 0.916, CSP (transfer) 0.747, CSP (retrained) 0.906, QPP logreg 0.671, E5 cosine 0.712, VC 0.617

| Comparison | Delta AUROC | 95% CI | p | Holm p |
|---|---|---|---|---|
| Standard (transfer) vs E5 cosine | +0.061 | [+0.028, +0.092] | 0.0005 | 0.0040 |
| CSP (transfer) vs E5 cosine | +0.036 | [+0.001, +0.070] | 0.0430 | 0.0430 |
| Standard (retrained) vs E5 cosine | +0.204 | [+0.177, +0.231] | 0.0005 | 0.0040 |
| CSP (retrained) vs E5 cosine | +0.194 | [+0.167, +0.221] | 0.0005 | 0.0040 |
| Standard (transfer) vs QPP logreg | +0.102 | [+0.068, +0.138] | 0.0005 | 0.0040 |
| CSP (transfer) vs QPP logreg | +0.076 | [+0.041, +0.113] | 0.0005 | 0.0040 |
| Standard (retrained) vs QPP logreg | +0.245 | [+0.216, +0.275] | 0.0005 | 0.0040 |
| CSP (retrained) vs QPP logreg | +0.235 | [+0.204, +0.264] | 0.0005 | 0.0040 |
| E5 cosine vs VC | +0.095 | [+0.061, +0.128] | 0.0005 | -- |
| QPP logreg vs VC | +0.054 | [+0.019, +0.088] | 0.0005 | -- |

### qwen (bm25_k5, support label, N=1500, B=2000)

AUROC: Standard (transfer) 0.811, Standard (retrained) 0.936, CSP (transfer) 0.818, CSP (retrained) 0.935, QPP logreg 0.671, E5 cosine 0.712, VC 0.700

| Comparison | Delta AUROC | 95% CI | p | Holm p |
|---|---|---|---|---|
| Standard (transfer) vs E5 cosine | +0.099 | [+0.068, +0.129] | 0.0005 | 0.0040 |
| CSP (transfer) vs E5 cosine | +0.106 | [+0.076, +0.135] | 0.0005 | 0.0040 |
| Standard (retrained) vs E5 cosine | +0.224 | [+0.198, +0.249] | 0.0005 | 0.0040 |
| CSP (retrained) vs E5 cosine | +0.223 | [+0.197, +0.248] | 0.0005 | 0.0040 |
| Standard (transfer) vs QPP logreg | +0.140 | [+0.105, +0.173] | 0.0005 | 0.0040 |
| CSP (transfer) vs QPP logreg | +0.147 | [+0.115, +0.181] | 0.0005 | 0.0040 |
| Standard (retrained) vs QPP logreg | +0.265 | [+0.237, +0.293] | 0.0005 | 0.0040 |
| CSP (retrained) vs QPP logreg | +0.264 | [+0.236, +0.291] | 0.0005 | 0.0040 |
| E5 cosine vs VC | +0.012 | [-0.023, +0.046] | 0.5140 | -- |
| QPP logreg vs VC | -0.029 | [-0.064, +0.008] | 0.1170 | -- |
