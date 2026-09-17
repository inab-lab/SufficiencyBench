"""Download models that don't require gated access."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from huggingface_hub import snapshot_download
from configs.paths import MODELS_DIR, MINILM_PATH

MODELS = [
    ("mistralai/Mistral-7B-Instruct-v0.3", "mistral-7b-instruct"),
    ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct"),
]

def main():
    for repo_id, dirname in MODELS:
        local_dir = MODELS_DIR / dirname
        if (local_dir / "config.json").exists():
            print(f"[SKIP] {repo_id}")
            continue
        print(f"[DOWNLOADING] {repo_id} -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            ignore_patterns=["*.gguf", "*.bin", "original/*"],
        )
        print(f"[DONE] {repo_id}")

    # MiniLM
    if (MINILM_PATH / "config.json").exists():
        print(f"[SKIP] MiniLM")
    else:
        print(f"[DOWNLOADING] MiniLM -> {MINILM_PATH}")
        snapshot_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            local_dir=str(MINILM_PATH),
        )
        print(f"[DONE] MiniLM")

    print("\n=== Done ===")

if __name__ == "__main__":
    main()
