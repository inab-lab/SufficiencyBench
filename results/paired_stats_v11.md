# Paired statistics on SufficiencyBench v1.1 test (1,069 questions, 2,138 conditions)

Question-clustered bootstrap: 2000 resamples of questions (both conditions move together), seed 0. Delta = AUROC(method) - AUROC(reference); p = two-sided bootstrap p (2*min tail fraction, floor 1/B). Holm is across methods within a model, for the comparison against the Standard probe. DeLong p (in JSON) treats examples as independent and ignores pairing by question.


## llama


| Method | AUROC | Clustered 95% CI | Δ vs Standard [CI], Holm p | Δ vs VC [CI], p |
|---|---|---|---|---|
| Standard Probe | 0.907 | [0.895, 0.917] | - | +0.189 [+0.174, +0.205], 0.0005 |
| CSP Probe | 0.904 | [0.893, 0.915] | -0.002 [-0.009, +0.005], 0.558 | +0.187 [+0.171, +0.203], 0.0005 |
| Verbalized Confidence | 0.717 | [0.702, 0.733] | -0.189 [-0.205, -0.174], 0.0025 | - |
| Response Delta | 0.520 | [0.508, 0.532] | -0.387 [-0.402, -0.370], 0.0025 | -0.197 [-0.215, -0.179], 0.0005 |
| Token Prob Delta | 0.754 | [0.740, 0.768] | -0.152 [-0.168, -0.136], 0.0025 | +0.037 [+0.018, +0.057], 0.0005 |
| Verbalized Comparison | 0.363 | [0.347, 0.381] | -0.543 [-0.565, -0.521], 0.0025 | -0.354 [-0.379, -0.329], 0.0005 |

PK strata AUROC: Standard Probe: PK0 0.906, PK1 0.924; CSP Probe: PK0 0.904, PK1 0.900; Verbalized Confidence: PK0 0.716, PK1 0.761; Response Delta: PK0 0.518, PK1 0.590; Token Prob Delta: PK0 0.756, PK1 0.750; Verbalized Comparison: PK0 0.365, PK1 0.304

## mistral


| Method | AUROC | Clustered 95% CI | Δ vs Standard [CI], Holm p | Δ vs VC [CI], p |
|---|---|---|---|---|
| Standard Probe | 0.910 | [0.899, 0.920] | - | +0.210 [+0.194, +0.226], 0.0005 |
| CSP Probe | 0.902 | [0.890, 0.913] | -0.008 [-0.016, -0.001], 0.032 | +0.202 [+0.187, +0.218], 0.0005 |
| Verbalized Confidence | 0.700 | [0.685, 0.714] | -0.210 [-0.226, -0.194], 0.0025 | - |
| Response Delta | 0.419 | [0.404, 0.433] | -0.491 [-0.510, -0.473], 0.0025 | -0.281 [-0.303, -0.259], 0.0005 |
| Token Prob Delta | 0.780 | [0.765, 0.795] | -0.130 [-0.147, -0.113], 0.0025 | +0.080 [+0.060, +0.100], 0.0005 |
| Verbalized Comparison | 0.348 | [0.334, 0.363] | -0.561 [-0.581, -0.541], 0.0025 | -0.351 [-0.374, -0.327], 0.0005 |

PK strata AUROC: Standard Probe: PK0 0.903, PK1 0.941; CSP Probe: PK0 0.897, PK1 0.923; Verbalized Confidence: PK0 0.695, PK1 0.721; Response Delta: PK0 0.429, PK1 0.372; Token Prob Delta: PK0 0.795, PK1 0.749; Verbalized Comparison: PK0 0.357, PK1 0.309

## qwen


| Method | AUROC | Clustered 95% CI | Δ vs Standard [CI], Holm p | Δ vs VC [CI], p |
|---|---|---|---|---|
| Standard Probe | 0.904 | [0.892, 0.914] | - | +0.123 [+0.110, +0.138], 0.0005 |
| CSP Probe | 0.914 | [0.903, 0.924] | +0.010 [+0.001, +0.019], 0.019 | +0.134 [+0.120, +0.148], 0.0005 |
| Verbalized Confidence | 0.780 | [0.764, 0.795] | -0.123 [-0.138, -0.110], 0.0025 | - |
| Response Delta | 0.460 | [0.442, 0.477] | -0.444 [-0.465, -0.422], 0.0025 | -0.320 [-0.344, -0.296], 0.0005 |
| Token Prob Delta | 0.737 | [0.723, 0.751] | -0.167 [-0.183, -0.151], 0.0025 | -0.043 [-0.063, -0.024], 0.0005 |
| Verbalized Comparison | 0.362 | [0.343, 0.381] | -0.541 [-0.566, -0.517], 0.0025 | -0.418 [-0.445, -0.389], 0.0005 |

PK strata AUROC: Standard Probe: PK0 0.904, PK1 0.897; CSP Probe: PK0 0.913, PK1 0.939; Verbalized Confidence: PK0 0.781, PK1 0.775; Response Delta: PK0 0.459, PK1 0.469; Token Prob Delta: PK0 0.740, PK1 0.714; Verbalized Comparison: PK0 0.360, PK1 0.415

## qwen3


| Method | AUROC | Clustered 95% CI | Δ vs Standard [CI], Holm p | Δ vs VC [CI], p |
|---|---|---|---|---|
| VC (chat-nothink) | 0.781 | [0.765, 0.796] | - | - |
| VC (chat-think) | 0.771 | [0.754, 0.787] | - | -0.010 [-0.027, +0.007], 0.237 |
| VC (cot-nothink) | 0.802 | [0.787, 0.817] | - | +0.021 [+0.007, +0.035], 0.003 |
| VC (raw) | 0.729 | [0.714, 0.743] | - | -0.052 [-0.066, -0.039], 0.0005 |

PK strata AUROC: VC (chat-nothink): PK0 0.781, PK1 0.766; VC (chat-think): PK0 0.770, PK1 0.803; VC (cot-nothink): PK0 0.801, PK1 0.825; VC (raw): PK0 0.728, PK1 0.762

## qwen14


| Method | AUROC | Clustered 95% CI | Δ vs Standard [CI], Holm p | Δ vs VC [CI], p |
|---|---|---|---|---|
| Standard Probe | 0.940 | [0.930, 0.948] | - | +0.121 [+0.107, +0.135], 0.0005 |
| CSP Probe | 0.949 | [0.940, 0.957] | +0.010 [+0.003, +0.016], 0.007 | +0.131 [+0.118, +0.144], 0.0005 |
| Verbalized Confidence | 0.818 | [0.803, 0.833] | -0.121 [-0.135, -0.107], 0.001 | - |

PK strata AUROC: Standard Probe: PK0 0.939, PK1 0.943; CSP Probe: PK0 0.949, PK1 0.936; Verbalized Confidence: PK0 0.818, PK1 0.846

## deepseek32


| Method | AUROC | Clustered 95% CI | Δ vs Standard [CI], Holm p | Δ vs VC [CI], p |
|---|---|---|---|---|
| Standard Probe | 0.949 | [0.941, 0.956] | - | +0.146 [+0.132, +0.160], 0.0005 |
| CSP Probe | 0.946 | [0.937, 0.954] | -0.002 [-0.008, +0.003], 0.38 | +0.143 [+0.129, +0.157], 0.0005 |
| Verbalized Confidence | 0.803 | [0.788, 0.818] | -0.146 [-0.160, -0.132], 0.001 | - |

PK strata AUROC: Standard Probe: PK0 0.948, PK1 0.953; CSP Probe: PK0 0.947, PK1 0.941; Verbalized Confidence: PK0 0.803, PK1 0.801

## Risk-coverage

Decision: predict sufficient if score >= t; coverage = fraction predicted sufficient; risk = fraction of those that are insufficient. AURC = trapezoid area over coverage.

| Model | Method | AURC | Coverage @ risk<=5% | Coverage @ risk<=10% |
|---|---|---|---|---|
| llama | Standard Probe | 0.2141 | 0.162 | 0.306 |
| llama | CSP Probe | 0.2155 | 0.151 | 0.333 |
| llama | Verbalized Confidence | 0.3636 | 0.000 | 0.000 |
| mistral | Standard Probe | 0.2088 | 0.202 | 0.315 |
| mistral | CSP Probe | 0.2162 | 0.066 | 0.308 |
| mistral | Verbalized Confidence | 0.3553 | 0.000 | 0.000 |
| qwen | Standard Probe | 0.2147 | 0.127 | 0.332 |
| qwen | CSP Probe | 0.2110 | 0.134 | 0.341 |
| qwen | Verbalized Confidence | 0.2911 | 0.000 | 0.000 |
| qwen14 | Standard Probe | 0.1944 | 0.273 | 0.426 |
| qwen14 | CSP Probe | 0.1854 | 0.309 | 0.465 |
| qwen14 | Verbalized Confidence | 0.2662 | 0.000 | 0.000 |
| deepseek32 | Standard Probe | 0.1853 | 0.347 | 0.478 |
| deepseek32 | CSP Probe | 0.1877 | 0.293 | 0.478 |
| deepseek32 | Verbalized Confidence | 0.2996 | 0.000 | 0.000 |
