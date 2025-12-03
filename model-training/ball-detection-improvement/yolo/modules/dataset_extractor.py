"""Dataset extraction pipeline tailored for SoccerNet MOT sequences."""

from __future__ import annotations

import configparser
import logging
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class SequenceSpec:
    """Represents a SoccerNet split glob specification."""

    split: str
    pattern: Path


class SoccerNetDatasetBuilder:
    """Low-level helper that converts SoccerNet MOT sequences into YOLO format."""

    def __init__(self, builder_config: Dict[str, Any]):
        self.config = builder_config
        self.workspace_root = Path(builder_config["workspace_root"]).expanduser().resolve()
        self.run_extraction = bool(builder_config.get("run_extraction", True))

        dataset_cfg = builder_config["dataset"]
        self.dataset_root = Path(dataset_cfg["root"]).expanduser()
        self.image_subdir = dataset_cfg.get("image_subdir", "img1")
        self.annotation_file = dataset_cfg.get("annotation_file", "gt/gt.txt")
        self.tracklet_file = dataset_cfg.get("tracklet_file", "gameinfo.ini")

        splits_cfg = dataset_cfg.get("splits", {})
        self.sequence_specs: List[SequenceSpec] = []
        for split, pattern in splits_cfg.items():
            if not pattern:
                continue
            pattern_path = Path(pattern)
            if not pattern_path.is_absolute():
                pattern_path = self.dataset_root / pattern_path
            self.sequence_specs.append(SequenceSpec(split=split, pattern=pattern_path))

        if not self.sequence_specs:
            raise ValueError("No dataset splits configured for extraction")

        classes_cfg = builder_config["classes"]
        self.class_order: List[str] = classes_cfg.get(
            "order", ["Player", "Ball", "Referee"]
        )
        raw_keyword_map = classes_cfg.get("keyword_map", {})
        self.keyword_map = {k.lower(): v for k, v in raw_keyword_map.items()}

        filters_cfg = builder_config.get("filters", {})
        self.min_box = float(filters_cfg.get("min_box_size", 0))
        self.min_conf = filters_cfg.get("drop_confidence_below")
        self.min_visibility = filters_cfg.get("min_visibility")
        self.keep_empty_frames = bool(filters_cfg.get("keep_empty_frames", False))

        output_cfg = builder_config["output"]
        self.output_root = self._resolve_path(output_cfg.get("dataset_dir", "ballDataset"))
        yaml_rel = output_cfg.get("dataset_yaml", "ballDataset/dataset.yaml")
        self.dataset_yaml = self._resolve_path(yaml_rel)
        self.naming_prefix = output_cfg.get("naming_prefix", "soccernet")

        self.options = builder_config.get("options", {})
        self.shuffle_sequences = bool(self.options.get("shuffle_sequences", True))
        self.seed = int(self.options.get("seed", 42))
        self.overwrite = bool(self.options.get("overwrite_existing", False))

        self.sample_counters: Counter[str] = Counter()
        self.instance_counters: Counter[str] = Counter()
        self.summary: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def run(self) -> Path:
        """Execute dataset extraction if enabled."""

        if not self.run_extraction:
            logger.info("Dataset extraction disabled via configuration (run_extraction=false)")
            return self.output_root

        if self._dataset_exists() and not self.overwrite:
            logger.info("Existing dataset detected at %s; skipping extraction", self.output_root)
            self._ensure_dataset_yaml()
            self._build_summary(existing_only=True)
            return self.output_root

        if self.output_root.exists() and self.overwrite:
            logger.info("Overwriting existing dataset directory: %s", self.output_root)
            shutil.rmtree(self.output_root)

        self._prepare_output_root()
        sequences = self._collect_sequences()
        if not sequences:
            logger.warning("No SoccerNet sequences matched the provided glob patterns")
            return self.output_root

        logger.info("Processing %d SoccerNet sequences", len(sequences))
        for seq_path, split in sequences:
            self._process_sequence(seq_path, split)

        self._write_dataset_yaml(sequences)
        self._build_summary(existing_only=False)
        logger.info("✅ Dataset extraction complete -> %s", self.output_root)
        return self.output_root

    def ensure_dataset_yaml(self) -> Path:
        """Guarantee that dataset.yaml exists and return its path."""

        if self.dataset_yaml.exists():
            return self.dataset_yaml

        if not self.output_root.exists():
            logger.warning("Dataset root %s missing; triggering extraction", self.output_root)
            self.run()
        else:
            logger.info("Dataset root exists but YAML missing; writing default config")
            self._write_dataset_yaml([])

        return self.dataset_yaml

    def get_statistics(self) -> Dict[str, Any]:
        """Return cached extraction summary."""

        if not self.summary:
            self._build_summary(existing_only=not bool(self.sample_counters))
        return self.summary

    # ------------------------------------------------------------------
    def _resolve_path(self, relative: str) -> Path:
        rel_path = Path(relative)
        if rel_path.is_absolute():
            return rel_path
        return (self.workspace_root / rel_path).resolve()

    def _dataset_exists(self) -> bool:
        required_dirs = [
            self.output_root / "images" / split
            for split in {spec.split for spec in self.sequence_specs}
        ]
        required_dirs += [
            self.output_root / "labels" / split
            for split in {spec.split for spec in self.sequence_specs}
        ]
        if not self.dataset_yaml.exists():
            return False
        return all(dir_path.exists() and any(dir_path.glob("*")) for dir_path in required_dirs)

    def _prepare_output_root(self) -> None:
        splits = {spec.split for spec in self.sequence_specs}
        for split in splits:
            (self.output_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _collect_sequences(self) -> List[Tuple[Path, str]]:
        sequences: List[Tuple[Path, str]] = []
        for spec in self.sequence_specs:
            glob_root = spec.pattern
            matches = sorted(glob_root.parent.glob(glob_root.name))
            for seq_path in matches:
                if seq_path.is_dir():
                    sequences.append((seq_path, spec.split))

        if self.shuffle_sequences and sequences:
            from random import Random

            rnd = Random(self.seed)
            rnd.shuffle(sequences)

        return sequences

    def _process_sequence(self, seq_path: Path, split: str) -> None:
        gt_path = seq_path / self.annotation_file
        if not gt_path.exists():
            logger.warning("Skipping %s (missing %s)", seq_path.name, self.annotation_file)
            return

        tracklet_labels = self._load_tracklet_labels(seq_path / self.tracklet_file)
        annotations_by_frame: Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]] = defaultdict(list)

        with open(gt_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parsed = self._parse_gt_line(line)
                if not parsed:
                    continue
                frame, track_id, bbox, conf, visibility = parsed
                if self.min_conf is not None and conf is not None and conf < self.min_conf:
                    continue
                if self.min_visibility is not None and visibility is not None and visibility < self.min_visibility:
                    continue
                descriptor = tracklet_labels.get(track_id)
                target_class = self._map_descriptor(descriptor)
                if target_class is None:
                    continue
                if bbox[2] < self.min_box or bbox[3] < self.min_box:
                    continue
                annotations_by_frame[frame].append((target_class, bbox))

        if not annotations_by_frame and not self.keep_empty_frames:
            logger.debug("No annotations kept for %s", seq_path.name)
            return

        image_dir = seq_path / self.image_subdir
        frame_ids = sorted(annotations_by_frame.keys())
        iterator: Iterable[int] = frame_ids
        if frame_ids:
            iterator = tqdm(frame_ids, desc=f"{seq_path.name} → {split}", leave=False)

        for frame in iterator:
            labels = annotations_by_frame.get(frame, [])
            if not labels and not self.keep_empty_frames:
                continue
            image_name = Path(f"{frame:06d}.jpg")
            image_path = image_dir / image_name
            if not image_path.exists():
                logger.warning("Image missing: %s", image_path)
                continue
            self._write_sample(image_path, labels, split, seq_path.name, frame)

        if frame_ids:
            iterator.close()

    def _write_sample(
        self,
        image_path: Path,
        labels: List[Tuple[str, Tuple[float, float, float, float]]],
        split: str,
        sequence_name: str,
        frame: int,
    ) -> None:
        with Image.open(image_path) as img:
            width, height = img.size

        stem = f"{self.naming_prefix}_{sequence_name}_{frame:06d}"
        dest_img = self.output_root / "images" / split / f"{stem}{image_path.suffix.lower()}"
        dest_lbl = self.output_root / "labels" / split / f"{stem}.txt"

        lines: List[str] = []
        for target_class, bbox in labels:
            try:
                class_idx = self.class_order.index(target_class.capitalize())
            except ValueError:
                continue
            x, y, w, h = self._normalize_bbox(bbox, width, height)
            lines.append(f"{class_idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            self.instance_counters[target_class] += 1

        if not lines and not self.keep_empty_frames:
            return

        dest_img.parent.mkdir(parents=True, exist_ok=True)
        dest_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, dest_img)
        dest_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self.sample_counters[split] += 1

    def _write_dataset_yaml(self, sequences: List[Tuple[Path, str]]) -> None:
        has_test_split = any(split == "test" for _, split in sequences)
        dataset_cfg = {
            "path": str(self.output_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": self.class_order,
            "nc": len(self.class_order),
        }
        if has_test_split or (self.output_root / "images" / "test").exists():
            dataset_cfg["test"] = "images/test"

        self.dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
        with open(self.dataset_yaml, "w", encoding="utf-8") as handle:
            yaml.safe_dump(dataset_cfg, handle, sort_keys=False)
        logger.info("dataset.yaml written ➜ %s", self.dataset_yaml)

    def _ensure_dataset_yaml(self) -> None:
        if not self.dataset_yaml.exists():
            self._write_dataset_yaml([])

    def _build_summary(self, existing_only: bool) -> None:
        stats = {
            "output_root": str(self.output_root),
            "dataset_yaml": str(self.dataset_yaml),
            "classes": self.class_order,
            "sample_counts": dict(self.sample_counters),
            "instance_counts": dict(self.instance_counters),
            "existing_dataset": existing_only,
        }
        self.summary = stats

    def _load_tracklet_labels(self, gameinfo_path: Path) -> Dict[int, str]:
        if not gameinfo_path.exists():
            logger.warning("%s not found; falling back to keyword inference only", gameinfo_path)
            return {}

        parser = configparser.ConfigParser()
        parser.read(gameinfo_path)
        mapping: Dict[int, str] = {}
        for section in parser.sections():
            for key, value in parser.items(section):
                if key.lower().startswith("trackletid_"):
                    try:
                        track_id = int(key.split("_")[-1])
                    except ValueError:
                        continue
                    descriptor = value.split(";")[0].strip().lower()
                    mapping[track_id] = descriptor

        if mapping:
            return mapping

        # Fallback: raw parsing
        for line in gameinfo_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().lower()
            if key.startswith("trackletid_"):
                try:
                    track_id = int(key.split("_")[-1])
                except ValueError:
                    continue
                descriptor = value.split(";")[0].strip().lower()
                mapping[track_id] = descriptor
        return mapping

    def _map_descriptor(self, descriptor: Optional[str]) -> Optional[str]:
        if descriptor:
            for keyword, target in self.keyword_map.items():
                if keyword in descriptor:
                    return target.capitalize()
        return None

    @staticmethod
    def _parse_gt_line(
        line: str,
    ) -> Optional[Tuple[int, int, Tuple[float, float, float, float], Optional[float], Optional[float]]]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            return None
        try:
            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6]) if len(parts) >= 7 else None
            visibility = float(parts[8]) if len(parts) >= 9 else None
            return frame, track_id, (x, y, w, h), conf, visibility
        except ValueError:
            return None

    @staticmethod
    def _normalize_bbox(
        bbox: Tuple[float, float, float, float], width: int, height: int
    ) -> Tuple[float, float, float, float]:
        x, y, w, h = bbox
        w = max(0.0, min(w, width))
        h = max(0.0, min(h, height))
        x = max(0.0, min(x, width))
        y = max(0.0, min(y, height))
        x_center = (x + w / 2.0) / width
        y_center = (y + h / 2.0) / height
        return x_center, y_center, w / width, h / height


class DatasetExtractor:
    """Facade used by the training pipeline to orchestrate extraction."""

    def __init__(self, extraction_config: Dict[str, Any], paths_config: Dict[str, Any]):
        self.extraction_config = extraction_config or {}
        self.paths_config = paths_config or {}
        builder_config = self._compose_builder_config()
        self.builder = SoccerNetDatasetBuilder(builder_config)

    def extract_dataset(self) -> Path:
        logger.info("🚀 Starting SoccerNet dataset extraction ...")
        return self.builder.run()

    def ensure_dataset_yaml(self) -> Path:
        return self.builder.ensure_dataset_yaml()

    def get_dataset_stats(self) -> Dict[str, Any]:
        return self.builder.get_statistics()

    # ------------------------------------------------------------------
    def _compose_builder_config(self) -> Dict[str, Any]:
        if not self.paths_config.get("workspace_root"):
            raise ValueError("paths.workspace_root must be defined for extraction")

        workspace_root = Path(self.paths_config["workspace_root"]).expanduser()
        soccernet_cfg = self.extraction_config.get("soccernet", {})
        if not soccernet_cfg.get("root"):
            raise ValueError("extraction.soccernet.root must be provided")

        classes_cfg = self.extraction_config.get("classes")
        if classes_cfg is None:
            classes_cfg = self._build_classes_from_mapping()

        builder_config: Dict[str, Any] = {
            "workspace_root": str(workspace_root),
            "run_extraction": bool(self.extraction_config.get("run_extraction", True)),
            "dataset": {
                "root": soccernet_cfg["root"],
                "splits": soccernet_cfg.get("splits", {}),
                "image_subdir": soccernet_cfg.get("image_subdir", "img1"),
                "annotation_file": soccernet_cfg.get("annotation_file", "gt/gt.txt"),
                "tracklet_file": soccernet_cfg.get("tracklet_file", "gameinfo.ini"),
            },
            "classes": classes_cfg,
            "filters": self.extraction_config.get("filters", {}),
            "output": self.extraction_config.get("output", {}),
            "options": self.extraction_config.get("options", {}),
        }

        return builder_config

    def _build_classes_from_mapping(self) -> Dict[str, Any]:
        mapping_cfg = self.extraction_config.get("class_mapping")
        if not mapping_cfg:
            raise ValueError(
                "Either extraction.classes or extraction.class_mapping must be defined"
            )

        order = mapping_cfg.get(
            "order",
            ["Player", "Ball", "Referee"],
        )
        keyword_map: Dict[str, str] = {}
        for category, keywords in mapping_cfg.items():
            if not category.endswith("_keywords"):
                continue
            target_class = category.replace("_keywords", "").capitalize()
            for keyword in keywords or []:
                keyword_map[keyword.lower()] = target_class

        if not keyword_map:
            raise ValueError("class_mapping keyword lists cannot be empty")

        return {"order": order, "keyword_map": keyword_map}


def create_extractor(
    extraction_config: Dict[str, Any], paths_config: Dict[str, Any]
) -> DatasetExtractor:
    """Factory wrapper expected by the CLI pipeline."""

    return DatasetExtractor(extraction_config, paths_config)
