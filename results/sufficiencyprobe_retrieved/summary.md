# SufficiencyProbe on retrieved context (bm25_k5, label = support)

Test N=1500 (sufficient rate 0.445); bootstrap over questions, B=1000. Text-only ModernBERT-large, no LLM.

| System | AUROC [95% CI] | PK=0 llama | PK=0 mistral | PK=0 qwen |
|---|---|---|---|---|
| transferred_v2 | 0.786 [0.762, 0.809] | 0.790 [0.766, 0.815] (n=1403) | 0.787 [0.759, 0.813] (n=1118) | 0.791 [0.765, 0.814] (n=1329) |
| transferred_v2_crossdataset | 0.770 [0.745, 0.794] | 0.772 [0.748, 0.796] (n=1403) | 0.760 [0.731, 0.788] (n=1118) | 0.769 [0.743, 0.792] (n=1329) |
| transferred_v2_crosssource | 0.649 [0.618, 0.676] | 0.653 [0.625, 0.681] (n=1403) | 0.655 [0.621, 0.686] (n=1118) | 0.649 [0.617, 0.678] (n=1329) |
| retrained_len512 | 0.921 [0.907, 0.934] | 0.917 [0.902, 0.932] (n=1403) | 0.916 [0.899, 0.933] (n=1118) | 0.919 [0.902, 0.934] (n=1329) |
| retrained_len1024 | 0.953 [0.942, 0.962] | 0.953 [0.942, 0.963] (n=1403) | 0.949 [0.937, 0.961] (n=1118) | 0.950 [0.939, 0.961] (n=1329) |

LLM retrained Standard probes (recomputed; analysis.md value in brackets):

- llama: 0.936 [0.924, 0.947] [0.936], within 0.002: True, n=1500
- mistral: 0.916 [0.902, 0.928] [0.916], within 0.002: True, n=1500
- qwen: 0.936 [0.924, 0.947] [0.936], within 0.002: True, n=1500

Paired delta AUROC (system minus LLM retrained Standard probe), bootstrap over questions:

| Contrast | Δ [95% CI] | p | Δ PK=0 [95% CI] (PK of that LLM) |
|---|---|---|---|
| transferred_v2__minus__llama_retrained_standard | -0.150 [-0.174, -0.126] | 0.000 | -0.146 [-0.172, -0.122] (n=1403) |
| transferred_v2_crossdataset__minus__llama_retrained_standard | -0.166 [-0.192, -0.141] | 0.000 | -0.163 [-0.189, -0.139] (n=1403) |
| transferred_v2_crosssource__minus__llama_retrained_standard | -0.287 [-0.316, -0.258] | 0.000 | -0.283 [-0.312, -0.252] (n=1403) |
| retrained_len512__minus__llama_retrained_standard | -0.015 [-0.031, -0.000] | 0.048 | -0.019 [-0.034, -0.002] (n=1403) |
| retrained_len1024__minus__llama_retrained_standard | +0.017 [0.005, 0.028] | 0.012 | +0.017 [0.005, 0.029] (n=1403) |
| transferred_v2__minus__mistral_retrained_standard | -0.130 [-0.153, -0.106] | 0.000 | -0.119 [-0.149, -0.092] (n=1118) |
| transferred_v2_crossdataset__minus__mistral_retrained_standard | -0.146 [-0.169, -0.121] | 0.000 | -0.147 [-0.178, -0.118] (n=1118) |
| transferred_v2_crosssource__minus__mistral_retrained_standard | -0.267 [-0.297, -0.239] | 0.000 | -0.252 [-0.285, -0.218] (n=1118) |
| retrained_len512__minus__mistral_retrained_standard | +0.005 [-0.010, 0.020] | 0.540 | +0.010 [-0.009, 0.030] (n=1118) |
| retrained_len1024__minus__mistral_retrained_standard | +0.037 [0.024, 0.049] | 0.000 | +0.043 [0.027, 0.059] (n=1118) |
| transferred_v2__minus__qwen_retrained_standard | -0.150 [-0.171, -0.128] | 0.000 | -0.144 [-0.168, -0.119] (n=1329) |
| transferred_v2_crossdataset__minus__qwen_retrained_standard | -0.166 [-0.190, -0.141] | 0.000 | -0.165 [-0.194, -0.140] (n=1329) |
| transferred_v2_crosssource__minus__qwen_retrained_standard | -0.287 [-0.314, -0.257] | 0.000 | -0.285 [-0.314, -0.256] (n=1329) |
| retrained_len512__minus__qwen_retrained_standard | -0.015 [-0.029, -0.001] | 0.042 | -0.015 [-0.032, 0.001] (n=1329) |
| retrained_len1024__minus__qwen_retrained_standard | +0.017 [0.005, 0.029] | 0.004 | +0.016 [0.003, 0.028] (n=1329) |

Training:

- max_length 512: best epoch 2, val AUROC 0.905, threshold 0.18 (val), test AUROC 0.921, test balanced acc at thr 0.847. Truncation (train): 80.6% of examples cut, mean 226 tokens cut among those, mean 22.9% of context tokens; mean total length 679.
- max_length 1024: best epoch 2, val AUROC 0.944, threshold 0.29 (val), test AUROC 0.953, test balanced acc at thr 0.878. Truncation (train): 4.8% of examples cut, mean 144 tokens cut among those, mean 0.5% of context tokens; mean total length 679.
