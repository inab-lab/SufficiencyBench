"""
Central path configuration for the Confident-but-Wrong project.
All paths are relative to PROJECT_ROOT. Adjust BASE_DATA_DIR if you
need to store large files (models, datasets, experiments) elsewhere.
"""

from pathlib import Path

# Project root = parent of configs/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Large-file storage — change this if you want models/data on a different disk
BASE_DATA_DIR = PROJECT_ROOT / "data"

# Models
MODELS_DIR = BASE_DATA_DIR / "models"
LLAMA_PATH = MODELS_DIR / "llama-3.1-8b-instruct"
MISTRAL_PATH = MODELS_DIR / "mistral-7b-instruct"
QWEN_PATH = MODELS_DIR / "qwen2.5-7b-instruct"
MINILM_PATH = MODELS_DIR / "minilm"

# Datasets (raw downloads)
DATASETS_DIR = BASE_DATA_DIR / "datasets"

# Experiments (outputs)
EXPERIMENTS_DIR = BASE_DATA_DIR / "experiments"
BENCH_DIR = EXPERIMENTS_DIR / "sufficiency_bench"
HIDDEN_STATES_DIR = EXPERIMENTS_DIR / "hidden_states"
PROBES_DIR = EXPERIMENTS_DIR / "probes"
CRAG_DIR = EXPERIMENTS_DIR / "crag"

# Results (paper-ready outputs)
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Model configs used across scripts
MODEL_CONFIGS = {
    "llama": {
        "path": str(LLAMA_PATH),
        "name": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "n_layers": 32,
        "hidden_dim": 4096,
    },
    "mistral": {
        "path": str(MISTRAL_PATH),
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "n_layers": 32,
        "hidden_dim": 4096,
    },
    "qwen": {
        "path": str(QWEN_PATH),
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "n_layers": 28,
        "hidden_dim": 3584,
    },
}

# Ensure key directories exist
for d in [MODELS_DIR, DATASETS_DIR, BENCH_DIR, HIDDEN_STATES_DIR,
          PROBES_DIR, CRAG_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
