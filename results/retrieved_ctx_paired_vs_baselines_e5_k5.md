Paired bootstrap over questions (B=2000, seed 0). p = 2*frac(delta<=0), floor 1/B. Holm within the 8 probe-vs-baseline comparisons per model; baseline-vs-VC rows use p = 2*min(frac<=0, frac>=0) (direction not assumed, as in analysis.md) and are not Holm-adjusted.

### llama (e5_k5, support label, N=1500, B=2000)

AUROC: Standard (transfer) 0.801, Standard (retrained) 0.922, CSP (transfer) 0.798, CSP (retrained) 0.924, QPP logreg 0.751, E5 cosine 0.652, VC 0.689

| Comparison | Delta AUROC | 95% CI | p | Holm p |
|---|---|---|---|---|
| Standard (transfer) vs E5 cosine | +0.149 | [+0.111, +0.188] | 0.0005 | 0.0040 |
| CSP (transfer) vs E5 cosine | +0.145 | [+0.107, +0.185] | 0.0005 | 0.0040 |
| Standard (retrained) vs E5 cosine | +0.269 | [+0.239, +0.300] | 0.0005 | 0.0040 |
| CSP (retrained) vs E5 cosine | +0.272 | [+0.242, +0.303] | 0.0005 | 0.0040 |
| Standard (transfer) vs QPP logreg | +0.050 | [+0.017, +0.086] | 0.0010 | 0.0040 |
| CSP (transfer) vs QPP logreg | +0.046 | [+0.012, +0.081] | 0.0070 | 0.0070 |
| Standard (retrained) vs QPP logreg | +0.170 | [+0.144, +0.196] | 0.0005 | 0.0040 |
| CSP (retrained) vs QPP logreg | +0.173 | [+0.147, +0.200] | 0.0005 | 0.0040 |
| E5 cosine vs VC | -0.036 | [-0.073, +0.001] | 0.0560 | -- |
| QPP logreg vs VC | +0.063 | [+0.028, +0.096] | 0.0005 | -- |
