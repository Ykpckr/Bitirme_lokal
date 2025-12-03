"""Count YOLO-format instance totals per class and split."""

from __future__ import annotations

import argparse
import collections
import pathlib
from typing import Iterable, List


def read_classes(dataset_dir: pathlib.Path) -> List[str]:
    classes_file = dataset_dir / "labels" / "classes.txt"
    if classes_file.exists():
        return [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    data_yaml = dataset_dir / "data.yaml"
    if data_yaml.exists():
        import yaml  # Lazy import to keep dependency optional if classes.txt is present

        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        names = data.get("names")
        if isinstance(names, dict):
            return [names[key] for key in sorted(names.keys())]
        if isinstance(names, list):
            return names
    raise FileNotFoundError("classes.txt or names field in data.yaml required to list classes")


def count_labels(label_files: Iterable[pathlib.Path]) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    for lbl in label_files:
        for line in lbl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cls_id = int(float(parts[0]))
            except (ValueError, IndexError):
                continue
            counts[cls_id] += 1
    return counts


def summarize_split(dataset_dir: pathlib.Path, split: str) -> collections.Counter:
    labels_dir = dataset_dir / "labels" / split
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")
    label_files = sorted(labels_dir.glob("*.txt"))
    return count_labels(label_files)


def format_counts(counts: collections.Counter, class_names: List[str]) -> str:
    lines = []
    total = sum(counts.values())
    lines.append(f"Total instances: {total}")
    for idx, name in enumerate(class_names):
        lines.append(f"  [{idx}] {name}: {counts.get(idx, 0)}")
    missing = set(counts) - set(range(len(class_names)))
    if missing:
        lines.append("  Unknown class ids: " + ", ".join(str(i) for i in sorted(missing)))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count YOLO label instances per dataset split")
    parser.add_argument(
        "dataset",
        type=pathlib.Path,
        nargs="?",
        default=pathlib.Path("model-training/ball-detection/ballDataset"),
        help="Path to dataset root containing labels/ and data.yaml",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=("train", "test"),
        help="Label subfolders to scan (default: train test)",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset.resolve()
    class_names = read_classes(dataset_dir)

    for split in args.splits:
        counts = summarize_split(dataset_dir, split)
        print(f"Split: {split}")
        print(format_counts(counts, class_names))
        print()


if __name__ == "__main__":
    main()
