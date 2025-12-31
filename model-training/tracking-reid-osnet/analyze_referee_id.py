import csv
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    csv_path = here / "outputs" / "botsort_team_reid.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path}")

    # Assumption for this project: referee class defaults to 2 (see _infer_class_ids fallback)
    referee_cls_id = 2

    track_ids = Counter()
    first_last = defaultdict(lambda: [None, None])  # tid -> [first_frame, last_frame]

    total_ref_rows = 0
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls_id = int(row["cls_id"])
            if cls_id != referee_cls_id:
                continue
            total_ref_rows += 1
            tid = int(row["track_id"])
            frame = int(row["frame_id"])
            track_ids[tid] += 1
            if first_last[tid][0] is None or frame < first_last[tid][0]:
                first_last[tid][0] = frame
            if first_last[tid][1] is None or frame > first_last[tid][1]:
                first_last[tid][1] = frame

    uniq = sorted(track_ids.keys())
    print(f"CSV: {csv_path}")
    print(f"Referee rows (cls_id={referee_cls_id}): {total_ref_rows}")
    print(f"Unique referee track_ids: {len(uniq)}")
    if uniq:
        print("track_ids:", uniq)
        print("Top by rows:")
        for tid, cnt in track_ids.most_common(10):
            a, b = first_last[tid]
            print(f"  id={tid} rows={cnt} frames=[{a},{b}]")

    # Hard check: should be exactly one referee track id (when referee appears)
    if len(uniq) > 1:
        print("WARNING: multiple referee IDs detected.")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
