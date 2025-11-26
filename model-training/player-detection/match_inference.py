#!/usr/bin/env python3
"""Run post-training inference with ByteTrack + team-color clustering."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import yaml
from tqdm import tqdm
from ultralytics import YOLO

from modules.team_assigner import AssignedTeam, DetectionObservation, TeamAssigner

LOGGER = logging.getLogger("match_inference")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MKV inference with ByteTrack and team clustering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/match_inference.yaml"),
        help="Path to inference configuration file",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Override video path from the config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output directory",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Process at most N frames (useful for smoke tests)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    config = load_config(args.config)

    if args.video:
        config["video_path"] = str(args.video)
    if args.output_dir:
        config["output_dir"] = str(args.output_dir)
    if args.max_frames is not None:
        config.setdefault("processing", {})["max_frames"] = args.max_frames

    output_dir = Path(config.get("output_dir", "match_outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_copy(config, output_dir / "resolved_config.yaml")

    run_pipeline(config, output_dir)


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_config_copy(config: Dict[str, Any], destination: Path) -> None:
    with open(destination, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def run_pipeline(config: Dict[str, Any], output_dir: Path) -> None:
    video_path = Path(config["video_path"]).expanduser()
    model_cfg = config.get("model", {})
    tracker_cfg = config.get("tracker", {})
    team_cfg = config.get("teams", {})
    assign_cfg = config.get("team_assignment", {})
    classes_cfg = config.get("classes", {})
    processing_cfg = config.get("processing", {})
    outputs_cfg = config.get("outputs", {})
    render_cfg = config.get("render", {})

    class_filter = set(classes_cfg.get("team_assign", ["player", "goalkeeper"]))
    max_frames = processing_cfg.get("max_frames")

    model = YOLO(model_cfg.get("weights", "models/football_detector_optimized/weights/best.pt"))
    tracker_path = resolve_tracker_path(tracker_cfg.get("config", "bytetrack.yaml"), base_dir=output_dir.parent)

    track_kwargs = {
        "tracker": tracker_path,
        "stream": True,
        "conf": model_cfg.get("conf", 0.15),
        "iou": model_cfg.get("iou", 0.5),
        "device": model_cfg.get("device"),
        "imgsz": model_cfg.get("imgsz"),
        "vid_stride": model_cfg.get("vid_stride"),
        "classes": model_cfg.get("class_indices"),
        "half": model_cfg.get("half"),
        "verbose": False,
        "persist": True,
    }
    track_kwargs = {k: v for k, v in track_kwargs.items() if v is not None}
    if tracker_cfg.get("overrides"):
        track_kwargs.update(tracker_cfg["overrides"])

    total_frames, fps = probe_video_meta(video_path)
    if total_frames is None:
        total_frames = max_frames
    if fps is None:
        fps = render_cfg.get("fps", 25.0)

    writer: Optional[cv2.VideoWriter] = None
    render_enabled = bool(render_cfg.get("enabled", True))
    target_video_path = output_dir / render_cfg.get("filename", "annotated.mp4")

    team_assigner = TeamAssigner(team_cfg, assign_cfg)
    per_frame_file = None
    frame_records_enabled = bool(outputs_cfg.get("save_frame_records", False))
    if frame_records_enabled:
        per_frame_file = open(output_dir / "frame_assignments.jsonl", "w", encoding="utf-8")

    try:
        results = model.track(source=str(video_path), **track_kwargs)
        class_names = get_class_names(model)
        progress = tqdm(results, total=total_frames, desc="Tracking", unit="frame")

        for frame_idx, result in enumerate(progress):
            if max_frames and frame_idx >= max_frames:
                break

            frame = result.orig_img
            if frame is None:
                continue

            if render_enabled and writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(target_video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps or 25.0,
                    (w, h),
                )

            frame_summary = []
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = [None] * len(xyxy)

                for det_idx, bbox in enumerate(xyxy):
                    class_id = cls_ids[det_idx]
                    class_name = class_names.get(class_id, str(class_id))
                    confidence = float(confs[det_idx])
                    track_id = int(track_ids[det_idx]) if track_ids[det_idx] is not None else None

                    observation = DetectionObservation(
                        frame_index=frame_idx,
                        track_id=track_id,
                        bbox=tuple(map(float, bbox)),
                        class_name=class_name,
                        confidence=confidence,
                    )

                    assignment: Optional[AssignedTeam] = None
                    if class_name in class_filter:
                        assignment = team_assigner.process_detection(frame, observation)

                    annotation = build_annotation_dict(observation, assignment)
                    frame_summary.append(annotation)

                    if render_enabled and writer is not None:
                        draw_detection(frame, annotation, team_assigner)

            if frame_records_enabled and per_frame_file is not None:
                per_frame_file.write(json.dumps({"frame_index": frame_idx, "detections": frame_summary}) + "\n")

            if render_enabled and writer is not None:
                writer.write(frame)

        progress.close()
    finally:
        if writer is not None:
            writer.release()
        if per_frame_file is not None:
            per_frame_file.close()

    save_summary(output_dir, video_path, config, team_assigner)
    LOGGER.info("Inference finished. Outputs written to %s", output_dir)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def probe_video_meta(video_path: Path) -> Tuple[Optional[int], Optional[float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        LOGGER.warning("Could not open video %s for metadata probe", video_path)
        return None, None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    cap.release()
    return frame_count, fps


def resolve_tracker_path(tracker_cfg: str, base_dir: Path) -> str:
    tracker_path = Path(tracker_cfg)
    if tracker_path.exists():
        return str(tracker_path)
    candidate = base_dir / tracker_cfg
    if candidate.exists():
        return str(candidate)
    return tracker_cfg  # allow ultralytics built-in names


def get_class_names(model: YOLO) -> Dict[int, str]:
    if hasattr(model, "names") and isinstance(model.names, dict):
        return model.names
    if hasattr(model.model, "names"):
        return model.model.names  # type: ignore[attr-defined]
    raise AttributeError("Unable to resolve class names from YOLO model")


def build_annotation_dict(observation: DetectionObservation, assignment: Optional[AssignedTeam]) -> Dict[str, Any]:
    annotation = {
        "track_id": observation.track_id,
        "class_name": observation.class_name,
        "confidence": observation.confidence,
        "bbox": list(observation.bbox),
    }
    if assignment:
        annotation["team_key"] = assignment.team_key
        annotation["team_name"] = assignment.team_name
        annotation["team_confidence"] = assignment.confidence
    else:
        annotation["team_key"] = None
        annotation["team_name"] = None
        annotation["team_confidence"] = 0.0
    return annotation


def draw_detection(frame: Any, annotation: Dict[str, Any], team_assigner: TeamAssigner) -> None:
    x1, y1, x2, y2 = map(int, annotation["bbox"])
    color = (200, 200, 200)
    label = annotation["class_name"]
    if annotation.get("team_key"):
        color = team_assigner.get_team_color(annotation["team_key"])
        label = f"{annotation['team_name']} ({annotation['class_name']})"
    elif annotation["class_name"] not in {"player", "goalkeeper"}:
        color = (255, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    caption = label
    if annotation.get("track_id") is not None:
        caption += f" #{annotation['track_id']}"
    cv2.putText(
        frame,
        caption,
        (x1, max(0, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def save_summary(output_dir: Path, video_path: Path, config: Dict[str, Any], team_assigner: TeamAssigner) -> None:
    summary = {
        "video_path": str(video_path),
        "teams": config.get("teams", {}),
        "model": config.get("model", {}),
        "tracker": config.get("tracker", {}),
        "team_assignment_config": config.get("team_assignment", {}),
        "assignments": team_assigner.get_track_summaries(),
    }
    with open(output_dir / "team_assignments.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
