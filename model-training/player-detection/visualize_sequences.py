#!/usr/bin/env python3
"""Quick-and-dirty SNMOT sequence visualizer.

Given the SNMOT dataset root, this script overlays bounding boxes and class
names for a couple of sequences so you can sanity-check the labels (e.g.,
the unified ``player`` class or historical ``player_team_left`` vs
``player_team_right``.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import cv2

LOGGER = logging.getLogger("snmot.visualize")

COLOR_CYCLE = {
    # Legacy team-specific labels
    "player_team_left": (0, 255, 0),
    "player_team_right": (0, 0, 255),
    "goalkeeper_team_left": (0, 165, 255),
    "goalkeeper_team_right": (255, 165, 0),
    # Current unified labels
    "player": (0, 200, 100),
    "goalkeeper": (255, 140, 0),
    "referee": (255, 0, 255),
    "ball": (255, 255, 255),
    "other": (128, 128, 128),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay SNMOT annotations for a few sequences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--snmot-root",
        type=Path,
        required=True,
        help="Path to SNMOT dataset root (contains train/test folders)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (train/test/dev)",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        help=(
            "Specific sequence names to visualize (defaults to first N"
            " sequences in the split)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help=(
            "Number of sequences to visualize when --sequences is not provided"
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum number of frames per sequence to export",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Only draw every Nth frame to keep outputs light",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("snmot_viz"),
        help="Directory where annotated frames will be written",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="If set, opens a cv2 window while iterating frames",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Optional resize factor for the output frames",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=0.5,
        help="Font scale for class labels",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Bounding-box line thickness",
    )
    return parser.parse_args()


def find_sequences(split_dir: Path, limit: int) -> List[Path]:
    candidates = [p for p in sorted(split_dir.iterdir()) if p.is_dir()]
    return candidates[:limit]


def load_seqinfo(sequence_dir: Path) -> Dict[str, int]:
    seqinfo_path = sequence_dir / "seqinfo.ini"
    parser = configparser.ConfigParser()
    parser.read(seqinfo_path)
    if "Sequence" not in parser:
        raise FileNotFoundError(
            f"Missing [Sequence] section in {seqinfo_path}"
        )
    section = parser["Sequence"]
    return {
        "imDir": section.get("imDir", "img1"),
        "imExt": section.get("imExt", ".jpg"),
        "seqLength": section.getint("seqLength", fallback=0),
    }


def normalize_name(text: str) -> str:
    replacements = {
        "goalkeepers": "goalkeeper",
        "keepers": "goalkeeper",
        "keeper": "goalkeeper",
        "goalie": "goalkeeper",
        "referees": "referee",
        "ref": "referee",
    }
    tokens = (
        text.lower()
        .replace("-", " ")
        .replace("/", " ")
        .replace("_", " ")
        .replace(";", " ")
        .split()
    )
    cleaned: List[str] = []
    for token in tokens:
        token = replacements.get(token, token)
        if token.isdigit():
            continue
        cleaned.append(token)
    return "_".join(cleaned)


def parse_gameinfo(sequence_dir: Path) -> Dict[int, str]:
    gameinfo_path = sequence_dir / "gameinfo.ini"
    parser = configparser.ConfigParser()
    parser.read(gameinfo_path)
    mapping: Dict[int, str] = {}
    if "Sequence" not in parser:
        return mapping
    for key, value in parser["Sequence"].items():
        if not key.startswith("trackletid_"):
            continue
        try:
            track_id = int(key.split("_")[1])
        except (IndexError, ValueError):
            continue
        class_name = normalize_name(value)
        if class_name:
            mapping[track_id] = class_name
    return mapping


def load_annotations(
    sequence_dir: Path,
    class_map: Dict[int, str],
) -> Dict[int, List[Tuple[int, float, float, float, float, str]]]:
    annotations: Dict[
        int,
        List[Tuple[int, float, float, float, float, str]],
    ] = {}
    gt_path = sequence_dir / "gt" / "gt.txt"
    with open(gt_path, newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 6:
                continue
            frame_id = int(float(row[0]))
            track_id = int(float(row[1]))
            x = float(row[2])
            y = float(row[3])
            w = float(row[4])
            h = float(row[5])
            class_name = class_map.get(track_id, "other")
            annotations.setdefault(frame_id, []).append(
                (track_id, x, y, w, h, class_name)
            )
    return annotations


def draw_boxes(image, labels, font_scale: float, thickness: int):
    for track_id, x, y, w, h, class_name in labels:
        pt1 = (int(x), int(y))
        pt2 = (int(x + w), int(y + h))
        color = COLOR_CYCLE.get(class_name, (200, 200, 200))
        cv2.rectangle(image, pt1, pt2, color, thickness)
        caption = f"{class_name}:{track_id}"
        (tw, th), _ = cv2.getTextSize(
            caption,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            1,
        )
        cv2.rectangle(
            image,
            (pt1[0], max(0, pt1[1] - th - 6)),
            (pt1[0] + tw + 6, pt1[1]),
            color,
            -1,
        )
        cv2.putText(
            image,
            caption,
            (pt1[0] + 3, pt1[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def visualize_sequence(
    sequence_dir: Path,
    output_dir: Path,
    max_frames: int,
    stride: int,
    show: bool,
    scale: float,
    font_scale: float,
    thickness: int,
):
    info = load_seqinfo(sequence_dir)
    image_dir = sequence_dir / info["imDir"]
    image_ext = info["imExt"]
    class_map = parse_gameinfo(sequence_dir)
    annotations = load_annotations(sequence_dir, class_map)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = sorted(annotations.keys())[:max_frames]
    for idx, frame_id in enumerate(frame_ids):
        if idx % stride != 0:
            continue
        img_path = image_dir / f"{frame_id:06d}{image_ext}"
        if not img_path.exists():
            LOGGER.warning("Missing frame %s", img_path)
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            LOGGER.warning("Failed to load %s", img_path)
            continue
        draw_boxes(image, annotations.get(frame_id, []), font_scale, thickness)
        if scale != 1.0:
            image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
        frame_out = output_dir / f"{sequence_dir.name}_{frame_id:06d}.jpg"
        cv2.imwrite(str(frame_out), image)
        if show:
            cv2.imshow("SNMOT", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if show:
        cv2.destroyAllWindows()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    args = parse_args()
    split_dir = args.snmot_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    if args.sequences:
        sequence_dirs: Iterable[Path] = [
            split_dir / name for name in args.sequences
        ]
    else:
        sequence_dirs = find_sequences(split_dir, args.limit)

    for seq_dir in sequence_dirs:
        if not seq_dir.exists():
            LOGGER.warning("Sequence not found: %s", seq_dir)
            continue
        LOGGER.info("Visualizing %s", seq_dir.name)
        visualize_sequence(
            seq_dir,
            args.output_dir,
            args.max_frames,
            args.stride,
            args.show,
            args.scale,
            args.font_scale,
            args.thickness,
        )


if __name__ == "__main__":
    main()
