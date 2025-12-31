import csv
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LastState:
    frame_id: int
    team_id: int
    cls_id: int
    cx: float
    cy: float


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def main() -> int:
    here = Path(__file__).resolve().parent
    csv_path = here / "outputs" / "botsort_team_reid.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path}")

    total_rows = 0
    max_frame = -1
    max_x2 = 0.0
    max_y2 = 0.0

    relink_rows = 0
    relink_sims: list[float] = []
    relink_ages: list[int] = []
    relink_center_dist_norms: list[float] = []
    relink_team_mismatch = 0
    relink_cls_mismatch = 0
    relink_missing_source_state = 0

    relink_rows_by_cls = Counter()
    relink_rows_player = 0
    relink_team_mismatch_player = 0
    relink_sims_player: list[float] = []
    relink_ages_player: list[int] = []
    relink_sims_player_team_match: list[float] = []
    relink_sims_player_team_mismatch: list[float] = []

    unique_tracks: set[int] = set()
    relinked_track_ids: set[int] = set()

    # Additional counters
    relink_source_use = Counter()

    last_by_track: dict[int, LastState] = {}

    # For births: first frame seen per track
    first_seen: dict[int, int] = {}

    relink_examples: list[dict] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "frame_id",
            "track_id",
            "cls_id",
            "conf",
            "x1",
            "y1",
            "x2",
            "y2",
            "team_id",
            "relinked",
            "relink_source_id",
            "relink_sim",
            "relink_inactive_age",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        for row in reader:
            total_rows += 1
            frame_id = int(row["frame_id"])
            track_id = int(row["track_id"])
            cls_id = int(row["cls_id"])
            team_id = int(row["team_id"])

            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            max_frame = max(max_frame, frame_id)
            max_x2 = max(max_x2, x2)
            max_y2 = max(max_y2, y2)

            unique_tracks.add(track_id)
            if track_id not in first_seen:
                first_seen[track_id] = frame_id

            relinked = int(row["relinked"]) == 1
            if relinked:
                relink_rows += 1
                relinked_track_ids.add(track_id)

                relink_rows_by_cls[cls_id] += 1
                if cls_id == 0:
                    relink_rows_player += 1

                source_id = int(row["relink_source_id"])
                relink_source_use[source_id] += 1

                relink_sim = float(row["relink_sim"])
                relink_sims.append(relink_sim)

                if cls_id == 0:
                    relink_sims_player.append(relink_sim)

                age = int(row["relink_inactive_age"])
                relink_ages.append(age)

                if cls_id == 0:
                    relink_ages_player.append(age)

                # Compare to last known state of the source track
                src_state = last_by_track.get(source_id)
                if src_state is None:
                    relink_missing_source_state += 1
                    dist_norm = float("nan")
                else:
                    if src_state.team_id != -1 and team_id != -1 and src_state.team_id != team_id:
                        relink_team_mismatch += 1
                        if cls_id == 0:
                            relink_team_mismatch_player += 1
                            relink_sims_player_team_mismatch.append(relink_sim)
                    if src_state.cls_id != cls_id:
                        relink_cls_mismatch += 1

                    if cls_id == 0 and src_state.team_id != -1 and team_id != -1 and src_state.team_id == team_id:
                        relink_sims_player_team_match.append(relink_sim)
                    # center distance normalized by diagonal (derived from max coords seen so far)
                    # We'll compute proper normalization after we finish scanning, but store raw dist now.
                    dist = math.hypot(cx - src_state.cx, cy - src_state.cy)
                    relink_center_dist_norms.append(dist)  # temporary raw dist
                    dist_norm = dist  # overwritten later

                if len(relink_examples) < 15:
                    relink_examples.append(
                        {
                            "frame_id": frame_id,
                            "track_id": track_id,
                            "source_id": source_id,
                            "sim": relink_sim,
                            "age": age,
                            "team_id": team_id,
                            "cls_id": cls_id,
                            "had_source_state": src_state is not None,
                        }
                    )

            # Update last state for this track
            last_by_track[track_id] = LastState(
                frame_id=frame_id,
                team_id=team_id,
                cls_id=cls_id,
                cx=cx,
                cy=cy,
            )

    # Finalize center-distance normalization using observed max coords.
    frame_w = max_x2
    frame_h = max_y2
    diag = math.hypot(frame_w, frame_h) if frame_w > 0 and frame_h > 0 else 1.0

    # Convert the list of raw distances to normalized distances, keeping NaNs out.
    raw_dists = relink_center_dist_norms
    relink_center_dist_norms = [d / diag for d in raw_dists if isinstance(d, float) and math.isfinite(d)]

    # Basic summary
    total_tracks = len(unique_tracks)
    total_frames = max_frame + 1 if max_frame >= 0 else 0

    def fmt_pct(x: float) -> str:
        return f"{x * 100.0:.2f}%"

    print("=== Relink / Reacquire Summary ===")
    print(f"CSV: {csv_path}")
    print(f"Rows: {total_rows:,}")
    print(f"Frames: {total_frames:,} (max frame_id={max_frame})")
    print(f"Observed frame size (approx): {int(frame_w)}x{int(frame_h)}")
    print(f"Unique track_ids: {total_tracks:,}")
    print(f"Relink rows (relinked=1): {relink_rows:,} ({fmt_pct(relink_rows / total_rows if total_rows else 0.0)})")
    print(f"Unique tracks that were relinked: {len(relinked_track_ids):,} ({fmt_pct(len(relinked_track_ids) / total_tracks if total_tracks else 0.0)})")

    if relink_rows_by_cls:
        cls_parts = ", ".join([f"cls{cid}:{cnt}" for cid, cnt in sorted(relink_rows_by_cls.items())])
        print(f"Relink rows by cls_id: {cls_parts}")

    # Births proxy
    births = total_tracks
    print(f"Track births (proxy = unique track_ids): {births:,}")

    if relink_sims:
        sims_sorted = sorted(relink_sims)
        ages_sorted = sorted(relink_ages)
        print("\n--- Relink sim ---")
        print(f"mean={statistics.fmean(relink_sims):.3f}  median={statistics.median(relink_sims):.3f}  p10={_percentile(sims_sorted, 10):.3f}  p90={_percentile(sims_sorted, 90):.3f}")

        print("\n--- Inactive age (frames) ---")
        print(f"mean={statistics.fmean(relink_ages):.1f}  median={statistics.median(relink_ages):.1f}  p90={_percentile(ages_sorted, 90):.1f}  p99={_percentile(ages_sorted, 99):.1f}")

    if relink_center_dist_norms:
        d_sorted = sorted(relink_center_dist_norms)
        print("\n--- Center distance at relink (normalized by diagonal) ---")
        print(f"mean={statistics.fmean(relink_center_dist_norms):.3f}  median={statistics.median(relink_center_dist_norms):.3f}  p90={_percentile(d_sorted, 90):.3f}  p99={_percentile(d_sorted, 99):.3f}")
        for thr in (0.25, 0.35, 0.50, 0.60):
            frac = sum(1 for d in relink_center_dist_norms if d > thr) / len(relink_center_dist_norms)
            print(f"> {thr:.2f}: {fmt_pct(frac)}")

    if relink_rows:
        print("\n--- Consistency checks vs relink_source_id last state ---")
        print(f"Missing source last-state: {relink_missing_source_state:,} ({fmt_pct(relink_missing_source_state / relink_rows)})")
        denom = max(relink_rows - relink_missing_source_state, 1)
        print(f"Team mismatch: {relink_team_mismatch:,} ({fmt_pct(relink_team_mismatch / denom)})")
        print(f"Class mismatch: {relink_cls_mismatch:,} ({fmt_pct(relink_cls_mismatch / denom)})")

        if relink_rows_player:
            denom_p = max(relink_rows_player - relink_missing_source_state, 1)
            print("\n--- Player-only (cls_id=0) ---")
            print(f"Relink rows (players): {relink_rows_player:,} ({fmt_pct(relink_rows_player / relink_rows)})")
            if relink_sims_player:
                sims_p = sorted(relink_sims_player)
                ages_p = sorted(relink_ages_player)
                print(f"sim median={statistics.median(relink_sims_player):.3f}  p10={_percentile(sims_p, 10):.3f}  p90={_percentile(sims_p, 90):.3f}")
                print(f"age median={statistics.median(relink_ages_player):.1f}  p90={_percentile(ages_p, 90):.1f}")
            print(f"team mismatch (players): {relink_team_mismatch_player:,} ({fmt_pct(relink_team_mismatch_player / denom_p)})")

            if relink_sims_player_team_match:
                print(f"sim median (team match): {statistics.median(relink_sims_player_team_match):.3f}")
            if relink_sims_player_team_mismatch:
                print(f"sim median (team mismatch): {statistics.median(relink_sims_player_team_mismatch):.3f}")

    # Sources used multiple times (can indicate churn / bad relink)
    multi_used = [(sid, c) for sid, c in relink_source_use.items() if sid >= 0 and c > 1]
    multi_used.sort(key=lambda x: x[1], reverse=True)
    print("\n--- Relink source reuse ---")
    print(f"Sources reused >1 times: {len(multi_used):,}")
    if multi_used[:10]:
        print("Top 10:", ", ".join([f"{sid}:{c}" for sid, c in multi_used[:10]]))

    if relink_examples:
        print("\n--- Example relinks (first 15) ---")
        for ex in relink_examples:
            print(
                f"f={ex['frame_id']} track={ex['track_id']} src={ex['source_id']} sim={ex['sim']:.3f} age={ex['age']} team={ex['team_id']} cls={ex['cls_id']} src_state={ex['had_source_state']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
