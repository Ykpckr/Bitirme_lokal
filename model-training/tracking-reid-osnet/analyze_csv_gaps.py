"""Analyze tracker CSV for fragmentation and whether long gaps look like out-of-view exits.

Heuristic:
- Infer frame width/height from max x2/y2 in the CSV.
- For each track, split into contiguous segments by frame_id.
- For each gap event (segment boundary), examine the last bbox before the gap and the first bbox after.
- If either bbox touches the image border within `border_px`, treat as likely "out_of_view".

Usage (PowerShell):
  python model-training/tracking/analyze_csv_gaps.py --csv outputs/botsort_team_reid.csv --gap 90

Notes:
- This does not require the video file.
- Results are heuristics; use to decide whether to tune association vs detector.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Row:
    frame_id: int
    track_id: int
    cls_id: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    team_id: int


def _touches_border(b: Tuple[float, float, float, float], w: float, h: float, border_px: float) -> bool:
    x1, y1, x2, y2 = b
    return (
        x1 <= border_px
        or y1 <= border_px
        or x2 >= (w - border_px)
        or y2 >= (h - border_px)
    )


def load_rows(csv_path: Path) -> List[Row]:
    rows: List[Row] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required = {"frame_id", "track_id", "cls_id", "conf", "x1", "y1", "x2", "y2", "team_id"}
        missing = required - set(r.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in CSV: {sorted(missing)}")
        for d in r:
            rows.append(
                Row(
                    frame_id=int(d["frame_id"]),
                    track_id=int(d["track_id"]),
                    cls_id=int(d["cls_id"]),
                    conf=float(d["conf"]),
                    x1=float(d["x1"]),
                    y1=float(d["y1"]),
                    x2=float(d["x2"]),
                    y2=float(d["y2"]),
                    team_id=int(float(d["team_id"])) if d["team_id"] != "" else -1,
                )
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--gap", type=int, default=90, help="Only analyze gaps >= this many frames")
    ap.add_argument(
        "--border_px",
        type=float,
        default=25.0,
        help="Pixels from border counted as out-of-view",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=15,
        help="How many worst gaps to print",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    rows = load_rows(csv_path)

    # Infer frame size from coordinates
    max_x2 = max(r.x2 for r in rows)
    max_y2 = max(r.y2 for r in rows)
    w = float(max_x2)
    h = float(max_y2)

    by_id: Dict[int, List[Row]] = {}
    for r in rows:
        by_id.setdefault(r.track_id, []).append(r)

    gap_events = []  # (gap, track_id, end_frame, start_frame, end_bbox, start_bbox)
    for tid, rs in by_id.items():
        rs = sorted(rs, key=lambda x: x.frame_id)
        for a, b in zip(rs, rs[1:]):
            if b.frame_id != a.frame_id + 1:
                g = b.frame_id - a.frame_id - 1
                gap_events.append(
                    (
                        g,
                        tid,
                        a.frame_id,
                        b.frame_id,
                        (a.x1, a.y1, a.x2, a.y2),
                        (b.x1, b.y1, b.x2, b.y2),
                    )
                )

    gap_events.sort(key=lambda x: x[0], reverse=True)
    big = [e for e in gap_events if e[0] >= int(args.gap)]

    out_of_view = 0
    likely_missed = 0
    for g, _tid, _f0, _f1, bb0, bb1 in big:
        if _touches_border(bb0, w, h, float(args.border_px)) or _touches_border(bb1, w, h, float(args.border_px)):
            out_of_view += 1
        else:
            likely_missed += 1

    print(f"csv: {csv_path}")
    print(f"rows: {len(rows)}  tracks: {len(by_id)}")
    print(f"inferred_size: w~{w:.0f}  h~{h:.0f}  border_px={float(args.border_px):.1f}")
    print(f"gap_events_total: {len(gap_events)}")
    print(f"gap_events>={int(args.gap)}: {len(big)}")
    if big:
        print(f">=gap: out_of_view={out_of_view} ({out_of_view/len(big):.1%})  likely_missed={likely_missed} ({likely_missed/len(big):.1%})")

    print("worst_gaps:")
    for g, tid, f0, f1, bb0, bb1 in big[: int(args.top)]:
        b0 = _touches_border(bb0, w, h, float(args.border_px))
        b1 = _touches_border(bb1, w, h, float(args.border_px))
        flag = "border" if (b0 or b1) else "center"
        print(f"  gap={g:4d}  id={tid:4d}  {f0}->{f1}  {flag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
