# Splice controls on SufficiencyBench v1.1 (test split)

95% CIs: question-clustered bootstrap, B=1000, seed 0.

## Control A: context-only probe, test AUROC (probe on h(c); no question in prompt)

| Model | Context-only @ Standard layer | Context-only @ own best layer (val) | Standard probe h(q+c) | CSP probe |
|---|---|---|---|---|
| llama | 0.889 [0.876, 0.902] (L14) | 0.889 [0.876, 0.901] (L13) | 0.907 [0.896, 0.918] (L14) | 0.904 [0.894, 0.915] (L14) |
| mistral | 0.884 [0.872, 0.896] (L13) | 0.876 [0.863, 0.889] (L16) | 0.910 [0.900, 0.920] (L13) | 0.902 [0.891, 0.913] (L13) |
| qwen | 0.870 [0.856, 0.885] (L19) | 0.870 [0.856, 0.885] (L19) | 0.904 [0.892, 0.915] (L19) | 0.914 [0.903, 0.924] (L18) |

## Control B: spliced-but-sufficient (one non-answer sentence replaced; label stays sufficient)

llama: 1038 spliced of 1069 test questions; 31 skipped (no replaceable non-answer sentence under the filters).
mistral: 1038 spliced of 1069 test questions; 31 skipped (no replaceable non-answer sentence under the filters).
qwen: 1038 spliced of 1069 test questions; 31 skipped (no replaceable non-answer sentence under the filters).

| Model | Probe | mean p orig-suf | mean p spliced-suf | mean p insuf | AUROC orig-suf vs spliced-suf | AUROC spliced-suf vs insuf | (ref) AUROC orig-suf vs insuf | frac spliced < val bal-acc thr | (ref) frac orig-suf < thr |
|---|---|---|---|---|---|---|---|---|---|
| llama | standard | 0.866 [0.846, 0.885] | 0.570 [0.543, 0.598] | 0.205 [0.182, 0.227] | 0.726 [0.711, 0.741] | 0.765 [0.748, 0.783] | 0.910 [0.898, 0.921] | 0.381 [0.353, 0.409] | 0.103 [0.085, 0.122] |
| llama | csp | 0.849 [0.830, 0.868] | 0.557 [0.527, 0.585] | 0.203 [0.182, 0.225] | 0.713 [0.699, 0.728] | 0.759 [0.740, 0.777] | 0.905 [0.894, 0.916] | 0.393 [0.363, 0.425] | 0.106 [0.087, 0.125] |
| mistral | standard | 0.828 [0.807, 0.847] | 0.474 [0.446, 0.501] | 0.181 [0.163, 0.203] | 0.759 [0.745, 0.772] | 0.738 [0.719, 0.756] | 0.911 [0.901, 0.921] | 0.411 [0.381, 0.442] | 0.090 [0.073, 0.108] |
| mistral | csp | 0.841 [0.820, 0.860] | 0.495 [0.465, 0.521] | 0.198 [0.176, 0.220] | 0.748 [0.735, 0.762] | 0.733 [0.715, 0.750] | 0.904 [0.893, 0.914] | 0.461 [0.433, 0.492] | 0.114 [0.095, 0.135] |
| qwen | standard | 0.837 [0.817, 0.856] | 0.532 [0.503, 0.562] | 0.213 [0.191, 0.236] | 0.712 [0.699, 0.727] | 0.749 [0.729, 0.768] | 0.902 [0.891, 0.913] | 0.499 [0.465, 0.532] | 0.181 [0.158, 0.204] |
| qwen | csp | 0.848 [0.828, 0.867] | 0.495 [0.465, 0.522] | 0.185 [0.164, 0.207] | 0.745 [0.732, 0.761] | 0.747 [0.727, 0.766] | 0.915 [0.904, 0.925] | 0.439 [0.409, 0.470] | 0.100 [0.083, 0.119] |

AUROC orig-suf vs spliced-suf uses original as the positive class (0.5 = probe ignores the splice).
AUROC spliced-suf vs insuf uses spliced-sufficient as the positive class (stays near the paper's value if the probe reads sufficiency).

Context-only prompt: `Based on the following context, answer the question. If the context does not contain enough information, say 'I cannot answer this from the provided context.'\n\nContext: {context}\n\nAnswer:`
