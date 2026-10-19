"""Download all required datasets."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import load_dataset
from configs.paths import DATASETS_DIR


DATASET_SPECS = [
    ("squad_v2", "rajpurkar/squad_v2", None),
    ("hotpotqa", "hotpotqa/hotpot_qa", "fullwiki"),
    ("nq", "google-research-datasets/natural_questions", "default"),
    ("eli5", "eli5_category", None),
]


def main():
    for name, path, subset in DATASET_SPECS:
        save_dir = DATASETS_DIR / name
        if save_dir.exists() and any(save_dir.iterdir()):
            print(f"[SKIP] {name} already exists at {save_dir}")
            continue

        print(f"[DOWNLOADING] {name} from {path}" +
              (f" ({subset})" if subset else ""))
        try:
            if subset:
                ds = load_dataset(path, subset, trust_remote_code=True)
            else:
                ds = load_dataset(path, trust_remote_code=True)
            ds.save_to_disk(str(save_dir))
            print(f"[DONE] {name} -> {save_dir}")
            # Print first example keys for inspection
            split = list(ds.keys())[0]
            print(f"  Splits: {list(ds.keys())}")
            print(f"  Columns: {ds[split].column_names}")
            print(f"  Size ({split}): {len(ds[split])}")
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            # Try fallback for eli5
            if name == "eli5":
                print(f"  Trying fallback: load_dataset('eli5')...")
                try:
                    ds = load_dataset("eli5", trust_remote_code=True)
                    ds.save_to_disk(str(save_dir))
                    print(f"  [DONE] eli5 (fallback)")
                except Exception as e2:
                    print(f"  [ERROR] eli5 fallback also failed: {e2}")
                    print(f"  Will substitute with another dataset later.")

    print("\n=== Dataset download complete ===")


if __name__ == "__main__":
    main()
