#!/usr/bin/env python3
"""
05_reset_project.py
Safely clear dataset and models folders. This permanently deletes files.
"""

import shutil
from pathlib import Path

DATASET_DIR = Path("dataset")
MODELS_DIR = Path("models")

def remove_path(p: Path):
    if not p.exists():
        print(f"[skip] {p} does not exist.")
        return
    try:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink(missing_ok=True)
        print(f"[removed] {p}")
    except Exception as e:
        print(f"[error] Could not remove {p}: {e}")

def main(confirm: bool = True):
    if confirm:
        resp = input("Permanently delete dataset and models? Type 'YES' to confirm: ").strip()
        if resp != "YES":
            print("Aborted.")
            return
    remove_path(DATASET_DIR)
    remove_path(MODELS_DIR)
    # Recreate empty folders for convenience
    # DATASET_DIR.mkdir(parents=True, exist_ok=True)
    # MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("Reset complete.")

if __name__ == "__main__":
    main()