"""Download all required models to local data/models/ directory."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from huggingface_hub import snapshot_download
from configs.paths import MODELS_DIR, MODEL_CONFIGS, MINILM_PATH

def main():
    # Download LLMs
    for name, cfg in MODEL_CONFIGS.items():
        local_dir = MODELS_DIR / name.replace("/", "-")
        # Use the directory name from paths.py
        if name == "llama":
            local_dir = MODELS_DIR / "llama-3.1-8b-instruct"
        elif name == "mistral":
            local_dir = MODELS_DIR / "mistral-7b-instruct"
        elif name == "qwen":
            local_dir = MODELS_DIR / "qwen2.5-7b-instruct"

        if (local_dir / "config.json").exists():
            print(f"[SKIP] {cfg['name']} already downloaded at {local_dir}")
            continue

        print(f"[DOWNLOADING] {cfg['name']} -> {local_dir}")
        snapshot_download(
            repo_id=cfg["name"],
            local_dir=str(local_dir),
            ignore_patterns=["*.gguf", "*.bin", "original/*"],
        )
        print(f"[DONE] {cfg['name']}")

    # Download MiniLM for sentence embeddings
    if (MINILM_PATH / "config.json").exists():
        print(f"[SKIP] MiniLM already downloaded")
    else:
        print(f"[DOWNLOADING] sentence-transformers/all-MiniLM-L6-v2 -> {MINILM_PATH}")
        snapshot_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            local_dir=str(MINILM_PATH),
        )
        print(f"[DONE] MiniLM")

    print("\n=== All models downloaded ===")


if __name__ == "__main__":
    main()
