"""
Person 3 — Dataset & Evaluation

Initial dataset inspection utilities for:
1. Ramsey Lab wound-analysis resources
2. QTDU ulcer/tissue dataset

This script will be expanded once the raw datasets
are placed under data/raw/.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def show_directory_structure(path: Path) -> None:
    """Print files and directories available under a path."""
    print(f"\nDataset location: {path}")

    if not path.exists():
        print("Directory does not exist yet.")
        return

    for item in sorted(path.rglob("*")):
        relative = item.relative_to(path)
        print(f"{'[DIR] ' if item.is_dir() else '[FILE]'} {relative}")


if __name__ == "__main__":
    print("=" * 60)
    print("PERSON 3 — DATASET INSPECTION")
    print("=" * 60)

    show_directory_structure(RAW_DATA_DIR)