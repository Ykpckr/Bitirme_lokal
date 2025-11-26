import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from modules.team_assigner import DetectionObservation, TeamAssigner


def _make_frame(color_bgr):
    frame = np.zeros((200, 100, 3), dtype=np.uint8)
    frame[:, :] = color_bgr
    return frame


def _make_detection(frame_idx, track_id):
    return DetectionObservation(
        frame_index=frame_idx,
        track_id=track_id,
        bbox=(10.0, 10.0, 90.0, 190.0),
        class_name="player",
        confidence=0.95,
    )


def test_team_assigner_clusters_two_colors():
    teams_cfg = {
        "home": {"name": "Red", "color_hint_rgb": [220, 40, 40]},
        "away": {"name": "Blue", "color_hint_rgb": [40, 80, 220]},
    }
    assign_cfg = {
        "warmup_min_observations": 6,
        "warmup_max_observations": 12,
        "upper_body_ratio": 1.0,
        "hist_bins": [8, 4, 2],
        "smoothing_window": 2,
        "min_track_votes": 1,
        "confidence_decay": 1.0,
    }

    assigner = TeamAssigner(teams_cfg, assign_cfg)
    red_frame = _make_frame((0, 0, 255))
    blue_frame = _make_frame((255, 0, 0))

    frame_idx = 0
    for _ in range(3):
        assigner.process_detection(red_frame, _make_detection(frame_idx, frame_idx))
        frame_idx += 1
        assigner.process_detection(blue_frame, _make_detection(frame_idx, frame_idx))
        frame_idx += 1

    assert assigner.ready, "KMeans should be ready after warmup"

    red_assignment = assigner.process_detection(red_frame, _make_detection(frame_idx, 999))
    assert red_assignment is not None
    assert red_assignment.team_key == "home"

    blue_assignment = assigner.process_detection(blue_frame, _make_detection(frame_idx + 1, 1000))
    assert blue_assignment is not None
    assert blue_assignment.team_key == "away"
