# Document-swap control (n_questions=1069, B=1000, question-clustered bootstrap, 95% CI)

AUROC [CI]. Matched = original SufficiencyBench test; Swap = insufficient context replaced by a BM25-nearest other-question sufficient context. Delta = swap - matched (paired bootstrap).

| Model | Predictor | Stratum | Matched | Swap | Delta (swap - matched) |
|---|---|---|---|---|---|
| llama | length_only | overall (n=1069q) | 0.503 [0.502, 0.504] | 0.636 [0.615, 0.657] | +0.134 [+0.113, +0.154] |
| llama | length_only | pk0 (n=1046q) | 0.503 [0.502, 0.504] | 0.641 [0.619, 0.661] | +0.138 [+0.117, +0.159] |
| llama | embedding_cosine | overall (n=1069q) | 0.634 [0.623, 0.647] | 0.941 [0.933, 0.949] | +0.307 [+0.294, +0.319] |
| llama | embedding_cosine | pk0 (n=1046q) | 0.634 [0.624, 0.646] | 0.943 [0.934, 0.952] | +0.308 [+0.296, +0.320] |
| llama | verbalized_confidence | overall (n=1069q) | 0.717 [0.702, 0.732] | 0.914 [0.902, 0.925] | +0.197 [+0.182, +0.211] |
| llama | verbalized_confidence | pk0 (n=1046q) | 0.716 [0.701, 0.732] | 0.913 [0.902, 0.924] | +0.197 [+0.183, +0.212] |
| llama | standard_probe | overall (n=1069q) | 0.907 [0.895, 0.918] | 0.980 [0.975, 0.984] | +0.073 [+0.063, +0.084] |
| llama | standard_probe | pk0 (n=1046q) | 0.906 [0.894, 0.917] | 0.980 [0.974, 0.984] | +0.074 [+0.063, +0.085] |
| llama | csp_probe | overall (n=1069q) | 0.904 [0.893, 0.915] | 0.980 [0.974, 0.984] | +0.076 [+0.065, +0.086] |
| llama | csp_probe | pk0 (n=1046q) | 0.904 [0.894, 0.915] | 0.980 [0.974, 0.985] | +0.075 [+0.065, +0.086] |
| mistral | length_only | overall (n=1069q) | 0.503 [0.502, 0.504] | 0.636 [0.615, 0.658] | +0.134 [+0.112, +0.155] |
| mistral | length_only | pk0 (n=883q) | 0.503 [0.502, 0.504] | 0.646 [0.624, 0.668] | +0.143 [+0.122, +0.166] |
| mistral | embedding_cosine | overall (n=1069q) | 0.634 [0.622, 0.645] | 0.941 [0.932, 0.950] | +0.307 [+0.294, +0.320] |
| mistral | embedding_cosine | pk0 (n=883q) | 0.629 [0.616, 0.642] | 0.939 [0.929, 0.948] | +0.310 [+0.296, +0.324] |
| mistral | verbalized_confidence | overall (n=1069q) | 0.700 [0.685, 0.716] | 0.841 [0.826, 0.855] | +0.142 [+0.126, +0.156] |
| mistral | verbalized_confidence | pk0 (n=883q) | 0.695 [0.678, 0.711] | 0.844 [0.828, 0.858] | +0.149 [+0.132, +0.164] |
| mistral | standard_probe | overall (n=1069q) | 0.910 [0.900, 0.920] | 0.975 [0.969, 0.980] | +0.065 [+0.055, +0.075] |
| mistral | standard_probe | pk0 (n=883q) | 0.903 [0.892, 0.914] | 0.971 [0.965, 0.978] | +0.068 [+0.058, +0.078] |
| mistral | csp_probe | overall (n=1069q) | 0.902 [0.890, 0.913] | 0.969 [0.962, 0.975] | +0.067 [+0.057, +0.078] |
| mistral | csp_probe | pk0 (n=883q) | 0.897 [0.885, 0.909] | 0.964 [0.957, 0.972] | +0.067 [+0.057, +0.078] |
| qwen | length_only | overall (n=1069q) | 0.503 [0.502, 0.504] | 0.636 [0.617, 0.657] | +0.134 [+0.114, +0.154] |
| qwen | length_only | pk0 (n=1027q) | 0.503 [0.502, 0.504] | 0.641 [0.621, 0.662] | +0.138 [+0.118, +0.159] |
| qwen | embedding_cosine | overall (n=1069q) | 0.634 [0.622, 0.646] | 0.941 [0.933, 0.950] | +0.307 [+0.295, +0.319] |
| qwen | embedding_cosine | pk0 (n=1027q) | 0.633 [0.621, 0.645] | 0.941 [0.932, 0.950] | +0.308 [+0.295, +0.321] |
| qwen | verbalized_confidence | overall (n=1069q) | 0.780 [0.766, 0.796] | 0.932 [0.921, 0.942] | +0.152 [+0.138, +0.167] |
| qwen | verbalized_confidence | pk0 (n=1027q) | 0.781 [0.766, 0.799] | 0.933 [0.922, 0.943] | +0.151 [+0.136, +0.165] |
| qwen | standard_probe | overall (n=1069q) | 0.904 [0.892, 0.914] | 0.979 [0.974, 0.984] | +0.076 [+0.065, +0.086] |
| qwen | standard_probe | pk0 (n=1027q) | 0.904 [0.891, 0.915] | 0.978 [0.973, 0.984] | +0.075 [+0.064, +0.086] |
| qwen | csp_probe | overall (n=1069q) | 0.914 [0.903, 0.925] | 0.984 [0.979, 0.989] | +0.070 [+0.060, +0.082] |
| qwen | csp_probe | pk0 (n=1027q) | 0.913 [0.901, 0.923] | 0.983 [0.978, 0.988] | +0.071 [+0.061, +0.082] |

## Construction

```
{
 "n": 1069,
 "top1_rejected_answer": 110,
 "top1_rejected_source": 0,
 "top1_rejected_any": 110,
 "rejected_answer_total": 222,
 "rejected_source_total": 0,
 "no_candidate": 0,
 "mean_rank_chosen": 0.2076707202993452,
 "frac_rank0": 0.8971000935453695,
 "mean_len_swap_ctx": 909.2095416276894,
 "mean_len_matched_insuf_ctx": 739.9466791393826,
 "mean_len_sufficient_ctx": 743.4527595884003,
 "n_distinct_swap_sources": 562
}
```

Mean chars (report subset): {'swap_insufficient': 909.2095416276894, 'matched_insufficient': 739.9466791393826, 'sufficient': 743.4527595884003}
