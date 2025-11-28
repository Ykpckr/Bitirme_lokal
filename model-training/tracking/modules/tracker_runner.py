from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import yaml
from rich.console import Console
from ultralytics import YOLO

from .datasets import SNMOTDataset, SNMOTSequence


@dataclass
class TrackingConfig:
    detection_weights: str
    tracker_yaml: str
    dataset_root: str
    sequence_names: Optional[Sequence[str]] = None
    split_dirs: Optional[Sequence[str]] = None
    detection_conf: float = 0.3
    detection_iou: float = 0.5
    imgsz: int = 1280
    device: str = "auto"
    save_visualizations: bool = False
    output_dir: str = "outputs/tracks"
    limit_frames: Optional[int] = None
    reid_weights: Optional[str] = None
    half: bool = False
    vid_stride: int = 1

    @classmethod
    def from_yaml(cls, path: Path | str) -> "TrackingConfig":
        data = yaml.safe_load(Path(path).read_text())
        detection = data.get("detection", {})
        tracking = data.get("tracking", {})
        datasets = data.get("datasets", {})
        return cls(
            detection_weights=detection["weights"],
            detection_conf=detection.get("conf", 0.3),
            detection_iou=detection.get("iou", 0.5),
            imgsz=detection.get("imgsz", 1280),
            tracker_yaml=tracking["tracker_yaml"],
            reid_weights=tracking.get("reid_weights"),
            device=tracking.get("device", "auto"),
            save_visualizations=tracking.get("save_visualizations", False),
            half=tracking.get("half", False),
            vid_stride=tracking.get("vid_stride", 1),
            dataset_root=datasets["root"],
            sequence_names=datasets.get("sequences"),
            split_dirs=datasets.get("splits"),
            output_dir=datasets.get("output_dir", "outputs/tracks"),
            limit_frames=datasets.get("limit_frames"),
        )


class TrackerRunner:
    def __init__(self, cfg: TrackingConfig) -> None:
        self.cfg = cfg
        self.console = Console()
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(cfg.detection_weights)
        self.dataset = SNMOTDataset(
            Path(cfg.dataset_root),
            cfg.sequence_names,
            cfg.split_dirs,
        )
        self.tracker_yaml = self._prepare_tracker_yaml()

    def _prepare_tracker_yaml(self) -> Path:
        tracker_path = Path(self.cfg.tracker_yaml)
        if not tracker_path.exists():
            raise FileNotFoundError(f"Tracker yaml not found: {tracker_path}")
        if not self.cfg.reid_weights:
            return tracker_path
        data = yaml.safe_load(tracker_path.read_text())
        data["reid_weights"] = str(self.cfg.reid_weights)
        runtime_yaml = self.output_dir / f"tracker_{tracker_path.name}"
        with runtime_yaml.open("w") as f:
            yaml.safe_dump(data, f)
        return runtime_yaml

    def run(self) -> None:
        self.console.print(
            f"[bold green]Tracking {len(self.dataset.sequences)} sequences using {self.cfg.detection_weights}[/bold green]"
        )
        for sequence in self.dataset:
            self._run_sequence(sequence)

    def _run_sequence(self, sequence: SNMOTSequence) -> None:
        self.console.print(f"\n[cyan]→ Sequence:[/cyan] {sequence.name}")
        mot_output = self.output_dir / f"{sequence.name}.txt"
        visual_project = self.output_dir / "visualizations"
        if self.cfg.save_visualizations:
            visual_project.mkdir(parents=True, exist_ok=True)

        generator = self.model.track(
            source=str(sequence.img_dir),
            tracker=str(self.tracker_yaml),
            conf=self.cfg.detection_conf,
            iou=self.cfg.detection_iou,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            half=self.cfg.half,
            stream=True,
            verbose=False,
            save=self.cfg.save_visualizations,
            project=str(visual_project) if self.cfg.save_visualizations else None,
            name=sequence.name,
            persist=True,
            vid_stride=self.cfg.vid_stride,
        )

        rows = []
        for frame_idx, result in enumerate(generator, start=1):
            if self.cfg.limit_frames and frame_idx > self.cfg.limit_frames:
                break
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.id is None:
                continue
            ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            for track_id, (x1, y1, x2, y2), conf in zip(ids, xyxy, confs):
                width = x2 - x1
                height = y2 - y1
                rows.append(
                    [
                        frame_idx,
                        track_id,
                        round(float(x1), 2),
                        round(float(y1), 2),
                        round(float(width), 2),
                        round(float(height), 2),
                        round(float(conf), 4),
                        -1,
                        -1,
                        -1,
                    ]
                )

        with mot_output.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        self.console.print(
            f"[green]✔ Saved[/green] {mot_output} ({len(rows)} detections)"
        )
