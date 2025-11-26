"""
Dataset Extraction Module

Handles extraction of frames and labels for supported datasets and outputs
data ready for YOLO training.
"""

import csv
import logging
import os
import shutil
import configparser
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set

import cv2
import numpy as np
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DatasetExtractor:
    """Extracts frames and YOLO-format labels for supported datasets."""
    
    def __init__(self, config: Dict[str, Any], paths_config: Dict[str, Any]):
        """
        Initialize dataset extractor.
        
        Args:
            config: Extraction configuration
            paths_config: Path configuration
        """
        self.config = config
        self.paths_config = paths_config
        self.dataset_type = (config.get('dataset_type') or 'soccernet').lower()

        # Track discovered class names in a deterministic order
        self._class_names: List[str] = []
        self._class_name_to_id: Dict[str, int] = {}

        initial_class_names = config.get('class_names') or []
        for class_name in initial_class_names:
            self._register_class_name(class_name)
        
        snmot_cfg = config.get('snmot', {})
        self._balance_config = snmot_cfg.get('balance_frames', {})
        self._instance_caps_cfg = snmot_cfg.get('instance_caps', {})
        self._instance_caps_state: Dict[str, Dict[str, int]]
        self._instance_caps_state = defaultdict(dict)
        self._label_transform_cfg = snmot_cfg.get('label_transform') or {}
        
    def extract_dataset(self) -> Path:
        """
        Extract the configured dataset into YOLO-ready format.
        
        Returns:
            Path to the output dataset root
        """
        if not self.config.get('run_extraction', True):
            logger.info("Dataset extraction skipped (run_extraction=False)")
            return self._get_output_root()

        logger.info("=" * 80)
        logger.info("DATASET EXTRACTION")
        logger.info("=" * 80)
        logger.info(f"Dataset type: {self.dataset_type.upper()}")

        if self.dataset_type == 'snmot':
            output_root = self._extract_snmot_dataset()
        else:
            output_root = self._extract_soccernet_dataset()

        self._create_dataset_yaml(output_root)
        logger.info("✅ Dataset extraction completed")
        return output_root

    def _extract_soccernet_dataset(self) -> Path:
        """Extract dataset from SoccerNet format."""
        soccernet_root = self._get_soccernet_root()
        output_root = self._get_output_root()

        logger.info(f"Source: {soccernet_root}")
        logger.info(f"Output: {output_root}")

        self._batch_extract_soccernet_dataset(soccernet_root, output_root)
        return output_root
    
    def _get_soccernet_root(self) -> Path:
        """Get SoccerNet root path."""
        workspace_root = Path(self.paths_config['workspace_root']).expanduser()
        return workspace_root / self.paths_config['soccernet_root']
    
    def _get_output_root(self) -> Path:
        """Get output dataset root path."""
        workspace_root = Path(self.paths_config['workspace_root']).expanduser()
        return workspace_root / self.paths_config['output_root']
    
    def _batch_extract_soccernet_dataset(self, soccernet_root: Path, output_root: Path):
        """Extract dataset from all matches."""
        matches = self._find_matches(soccernet_root)
        splits = self._split_by_season(matches)
        
        for split_name, match_list in splits.items():
            if not match_list:
                continue
                
            logger.info(f"Processing {split_name} split: {len(match_list)} matches")
            split_output = output_root / split_name
            
            for match_dir in tqdm(match_list, desc=f"Extracting {split_name}"):
                self._extract_match(match_dir, split_output)
    
    def _find_matches(self, root: Path) -> List[Path]:
        """Find all valid match directories."""
        matches = []
        video_pattern = self.paths_config['video_pattern']
        annotation_pattern = self.paths_config['annotation_pattern']
        
        for league in root.iterdir():
            if not league.is_dir():
                continue
            for season in league.iterdir():
                if not season.is_dir():
                    continue
                for match_dir in season.iterdir():
                    if not match_dir.is_dir():
                        continue
                    
                    # Check if both halves have required files
                    valid = False
                    for half in (1, 2):
                        video_file = match_dir / video_pattern.format(half=half)
                        annotation_file = match_dir / annotation_pattern.format(half=half)
                        if video_file.exists() and annotation_file.exists():
                            valid = True
                            break
                    
                    if valid:
                        matches.append(match_dir)
        
        return sorted(matches)
    
    def _split_by_season(self, matches: List[Path]) -> Dict[str, List[Path]]:
        """Split matches by season according to configuration."""
        season_splits = self.config.get('season_splits', {
            'train': ['2014-2015'],
            'val': ['2015-2016'],
            'test': ['2016-2017']
        })
        
        splits = {split: [] for split in season_splits.keys()}
        
        for match in matches:
            season = match.parent.name
            for split_name, seasons in season_splits.items():
                if season in seasons:
                    splits[split_name].append(match)
                    break
        
        return splits
    
    def _extract_match(self, match_dir: Path, split_output: Path):
        """Extract frames and labels from a single match."""
        video_pattern = self.paths_config['video_pattern']
        annotation_pattern = self.paths_config['annotation_pattern']
        
        for half in (1, 2):
            video_file = match_dir / video_pattern.format(half=half)
            annotation_file = match_dir / annotation_pattern.format(half=half)
            
            if not (video_file.exists() and annotation_file.exists()):
                continue
            
            match_name = match_dir.name.replace(' ', '_').replace('-', '_')
            output_dir = split_output / f"{match_name}_half{half}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self._extract_frames_and_labels(
                str(annotation_file),
                str(video_file),
                str(output_dir)
            )
    
    def _extract_frames_and_labels(
        self,
        json_path: str,
        video_path: str,
        output_dir: str
    ) -> int:
        """Extract frames and YOLO labels from a single video-annotation pair."""
        # Import visualization functions for alignment consistency
        try:
            # Try absolute import first
            from utils.visualization_utils import load_soccernet_json, make_scaler
        except ImportError:
            try:
                # Try adding current directory to path
                import sys
                from pathlib import Path as PathLib
                current_dir = PathLib(__file__).parent.parent
                if str(current_dir) not in sys.path:
                    sys.path.insert(0, str(current_dir))
                from utils.visualization_utils import load_soccernet_json, make_scaler
            except ImportError:
                # Fallback to original training directory
                try:
                    training_dir = PathLib(__file__).parent.parent.parent / "training"
                    if str(training_dir) not in sys.path:
                        sys.path.append(str(training_dir))
                    from visualize_maskrcnn_soccernet import load_soccernet_json, make_scaler
                except ImportError as e:
                    logger.error(f"Failed to import visualization functions: {e}")
                    raise
        
        out_root = Path(output_dir)
        img_dir = out_root / "images"
        lbl_dir = out_root / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        player_class_id = self._register_class_name('player')
        
        preds, size = load_soccernet_json(json_path)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        vW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        video_total_sec = (total_frames / fps) if total_frames > 0 else max(1.0, len(preds) / 2.0)
        
        # Get extraction parameters
        det_fps = self.config.get('det_fps')
        if det_fps is None:
            det_fps = (len(preds) / video_total_sec) if video_total_sec > 0 else 2.0
        
        det_start_sec = self.config.get('det_start_sec', 0.0)
        frame_shift = self.config.get('frame_shift', 0)
        max_samples = self.config.get('max_samples_per_half')
        min_box_size = self.config.get('min_box_size', 10)
        scale_mode = self.config.get('scale_mode', 'scale')
        jpeg_quality = self.config.get('jpeg_quality', 95)
        
        # Source resolution for bboxes
        if size and len(size) >= 3:
            srcH, srcW = int(size[1]), int(size[2])
        else:
            srcW, srcH = vW, vH
        scaler = make_scaler((srcW, srcH), (vW, vH), mode=scale_mode)
        
        det_count = len(preds)
        if det_count == 0:
            cap.release()
            return 0
        
        # Compute detection times
        t_det = np.array([i / det_fps + det_start_sec for i in range(det_count)], dtype=float)
        
        # Decide how many to export
        num_preds = det_count if max_samples is None else min(det_count, max_samples)
        
        exported = 0
        
        for i in range(num_preds):
            det = preds[i]
            frame_idx = int(np.floor(t_det[i] * fps)) + frame_shift
            frame_idx = max(0, min(total_frames - 1, frame_idx))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            bboxes = det.get("bboxes", []) or []
            if not bboxes:
                continue
            
            # Save image
            img_name = f"{Path(video_path).stem}_pred{i:06d}_frame{frame_idx:06d}.jpg"
            img_path = img_dir / img_name
            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            
            # Write labels
            lbl_path = lbl_dir / img_name.replace(".jpg", ".txt")
            valid = 0
            
            with open(lbl_path, "w") as f:
                for bb in bboxes:
                    x1, y1, x2, y2 = map(float, bb)
                    
                    # Scale/clamp to frame size
                    X1, Y1, X2, Y2 = scaler(x1, y1, x2, y2)
                    
                    if self.config.get('validate_boxes', True):
                        X1 = max(0, min(X1, vW - 1))
                        Y1 = max(0, min(Y1, vH - 1))
                        X2 = max(0, min(X2, vW))
                        Y2 = max(0, min(Y2, vH))
                    
                    w = X2 - X1
                    h = Y2 - Y1
                    if w < min_box_size or h < min_box_size or X2 <= X1 or Y2 <= Y1:
                        continue
                    
                    # YOLO normalized format
                    x_center = ((X1 + X2) / 2) / vW
                    y_center = ((Y1 + Y2) / 2) / vH
                    nw = w / vW
                    nh = h / vH
                    
                    # Clamp to [0, 1]
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    nw = max(0.0, min(1.0, nw))
                    nh = max(0.0, min(1.0, nh))
                    
                    # For SoccerNet extractions we only export the player class.
                    cls_id = player_class_id
                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}\n")
                    valid += 1
            
            if valid == 0:
                # Remove empty image/label
                try:
                    img_path.unlink(missing_ok=True)
                    lbl_path.unlink(missing_ok=True)
                except Exception:
                    pass
            else:
                exported += 1
        
        cap.release()
        return exported

    def _extract_snmot_dataset(self) -> Path:
        """Extract dataset from the SNMOT tracking format."""
        dataset_root = self._get_snmot_root()
        output_root = self._get_output_root()

        if not dataset_root.exists():
            raise FileNotFoundError(f"SNMOT dataset root not found: {dataset_root}")

        logger.info(f"Source: {dataset_root}")
        logger.info(f"Output: {output_root}")

        if output_root.exists() and self.config.get('clean_output', True):
            logger.info("Removing previous extracted dataset (clean_output=True)")
            shutil.rmtree(output_root)

        output_root.mkdir(parents=True, exist_ok=True)

        splits = self._resolve_snmot_splits(dataset_root)
        logger.info(f"Processing splits: {', '.join(splits)}")

        summary: Dict[str, Dict[str, int]] = {}
        for split_name in splits:
            source_dir = dataset_root / split_name
            if not source_dir.exists():
                logger.warning(f"Split directory missing, skipping: {source_dir}")
                continue

            stats = self._process_snmot_split(split_name, source_dir, output_root / split_name)
            summary[split_name] = stats

        if summary:
            logger.info("SNMOT extraction summary:")
            for split_name, stats in summary.items():
                logger.info(
                    "  %s: %d sequences, %d frames, %d labels",
                    split_name,
                    stats['sequences'],
                    stats['frames'],
                    stats['labels']
                )

        return output_root

    def _get_snmot_root(self) -> Path:
        """Resolve the SNMOT dataset root path."""
        workspace_root = Path(self.paths_config['workspace_root']).expanduser()
        dataset_subdir = self.paths_config.get('snmot_root')
        if not dataset_subdir:
            raise KeyError("snmot_root not defined in paths configuration")
        return (workspace_root / dataset_subdir).resolve()

    def _resolve_snmot_splits(self, dataset_root: Path) -> List[str]:
        """Determine which SNMOT splits to process."""
        snmot_cfg = self.config.get('snmot', {})
        configured = snmot_cfg.get('splits')
        if configured:
            candidates = [str(split) for split in configured]
        else:
            candidates: List[str] = []
            for key in ('train_split', 'val_split', 'test_split'):
                name = self.paths_config.get(key)
                if name and name not in candidates:
                    candidates.append(name)
            if not candidates:
                candidates = ['train', 'test']
        # Preserve order while removing duplicates
        seen = set()
        ordered = []
        for name in candidates:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        existing = [name for name in ordered if (dataset_root / name).exists()]
        if existing:
            return existing
        # Fallback: use all available subdirectories
        return [p.name for p in sorted(dataset_root.iterdir()) if p.is_dir()]

    def _process_snmot_split(self, split_name: str, source_dir: Path, target_dir: Path) -> Dict[str, int]:
        """Convert a single SNMOT split into YOLO format."""
        images_out = target_dir / "images"
        labels_out = target_dir / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        snmot_cfg = self.config.get('snmot', {})
        copy_images = snmot_cfg.get('copy_images', True)
        include_empty = snmot_cfg.get('include_empty_frames', False)

        sequence_dirs = [seq for seq in sorted(source_dir.iterdir()) if seq.is_dir()]
        stats = {'sequences': 0, 'frames': 0, 'labels': 0}

        if not sequence_dirs:
            logger.warning(f"No sequences found for split {split_name} in {source_dir}")
            return stats

        if self._instance_caps_enabled_for_split(split_name) and \
                self._instance_caps_cfg.get('shuffle_sequences', True):
            random.shuffle(sequence_dirs)

        for sequence_dir in tqdm(sequence_dirs, desc=f"Extracting {split_name.upper()}"):
            try:
                seq_stats = self._process_snmot_sequence(
                    sequence_dir,
                    images_out,
                    labels_out,
                    split_name=split_name,
                    copy_images=copy_images,
                    include_empty=include_empty
                )
            except Exception as exc:  # pragma: no cover - safety net for unexpected formats
                logger.error(f"Failed to process sequence {sequence_dir.name}: {exc}")
                continue

            stats['sequences'] += 1
            stats['frames'] += seq_stats['frames']
            stats['labels'] += seq_stats['labels']

        return stats

    def _process_snmot_sequence(
        self,
        sequence_dir: Path,
        images_out: Path,
        labels_out: Path,
        *,
        split_name: str,
        copy_images: bool,
        include_empty: bool
    ) -> Dict[str, int]:
        """Process a single SNMOT sequence directory."""
        meta = self._load_snmot_sequence_metadata(sequence_dir)
        annotations = self._load_snmot_annotations(sequence_dir, meta)

        stats = {'frames': 0, 'labels': 0}

        if not annotations and not include_empty:
            logger.debug(f"Sequence {meta['sequence_name']} has no annotations; skipping")
            return stats

        frame_ids = sorted(annotations.keys())
        if include_empty:
            total_frames = meta.get('total_frames') or 0
            if total_frames > 0:
                frame_ids = sorted(set(frame_ids).union(range(1, total_frames + 1)))

        balance_cfg = self._balance_config
        apply_caps = self._instance_caps_enabled_for_split(split_name)
        if apply_caps and self._instance_caps_cfg.get('shuffle_frames', True):
            random.shuffle(frame_ids)

        for frame_id in frame_ids:
            labels = annotations.get(frame_id, [])
            if not labels and not include_empty:
                continue

            label_names = self._class_names_from_labels(labels)
            if not self._should_keep_frame(labels, label_names, balance_cfg):
                continue

            if apply_caps:
                labels = self._apply_instance_caps(labels, split_name)
                if not labels and not include_empty:
                    continue

            src_image = meta['image_dir'] / f"{frame_id:06d}{meta['image_ext']}"
            if not src_image.exists():
                logger.debug(f"Missing frame {src_image}; skipping")
                continue

            image_stem = f"{meta['sequence_name']}_{frame_id:06d}"
            dst_image = images_out / f"{image_stem}{meta['image_ext']}"
            dst_label = labels_out / f"{image_stem}.txt"

            if copy_images:
                shutil.copy2(src_image, dst_image)
            else:
                try:
                    os.link(src_image, dst_image)
                except OSError:
                    shutil.copy2(src_image, dst_image)

            if labels:
                self._write_yolo_label_file(dst_label, labels)
                stats['labels'] += len(labels)
                self._maybe_duplicate_minority_frame(
                    labels,
                    label_names,
                    dst_image,
                    dst_label,
                    images_out,
                    labels_out,
                    image_stem,
                    meta['image_ext'],
                    stats,
                    balance_cfg
                )
            elif include_empty:
                dst_label.touch()

            stats['frames'] += 1

        return stats

    def _load_snmot_sequence_metadata(self, sequence_dir: Path) -> Dict[str, Any]:
        """Load sequence metadata from seqinfo.ini and gameinfo.ini."""
        seqinfo_path = sequence_dir / "seqinfo.ini"
        if not seqinfo_path.exists():
            raise FileNotFoundError(f"seqinfo.ini not found for sequence: {sequence_dir}")

        parser = configparser.ConfigParser()
        parser.read(seqinfo_path)
        if 'Sequence' not in parser:
            raise ValueError(f"seqinfo.ini missing [Sequence] section: {seqinfo_path}")

        section = parser['Sequence']
        sequence_name = section.get('name', sequence_dir.name)
        image_dir = sequence_dir / section.get('imDir', 'img1')
        image_ext = section.get('imExt', '.jpg')
        width = section.getint('imWidth', fallback=1920)
        height = section.getint('imHeight', fallback=1080)
        total_frames = section.getint('seqLength', fallback=0)
        frame_rate = section.getint('frameRate', fallback=25)

        tracklet_classes = self._parse_gameinfo_classes(sequence_dir / 'gameinfo.ini')

        if not tracklet_classes:
            logger.warning(f"No tracklet metadata parsed for sequence {sequence_name}")

        return {
            'sequence_name': sequence_name,
            'image_dir': image_dir,
            'image_ext': image_ext,
            'width': width,
            'height': height,
            'total_frames': total_frames,
            'frame_rate': frame_rate,
            'tracklet_classes': tracklet_classes,
        }

    def _parse_gameinfo_classes(self, gameinfo_path: Path) -> Dict[int, str]:
        """Parse class mappings from gameinfo.ini."""
        mapping: Dict[int, str] = {}

        if not gameinfo_path.exists():
            logger.warning(f"gameinfo.ini not found: {gameinfo_path}")
            return mapping

        parser = configparser.ConfigParser()
        parser.read(gameinfo_path)
        if 'Sequence' not in parser:
            return mapping

        for key, value in parser['Sequence'].items():
            if not key.startswith('trackletid_'):
                continue

            try:
                tracklet_id = int(key.split('_')[1])
            except (IndexError, ValueError):
                continue

            parts = [part.strip() for part in value.split(';') if part.strip()]
            base = parts[0] if parts else ''
            qualifier = ' '.join(parts[1:]) if len(parts) > 1 else ''

            class_name = self._sanitize_snmot_class_name(base, qualifier)
            if class_name:
                mapping[tracklet_id] = class_name

        return mapping

    def _sanitize_snmot_class_name(self, base: str, qualifier: str) -> str:
        """Convert raw SNMOT class strings into YOLO-friendly names."""
        base_tokens = self._normalize_tokens(base)
        if not base_tokens:
            return ''

        qualifier_tokens = self._normalize_tokens(qualifier)
        if qualifier_tokens and base_tokens[0] in {'referee'}:
            tokens = base_tokens + qualifier_tokens
        else:
            tokens = base_tokens

        return '_'.join(tokens)

    def _normalize_tokens(self, text: str) -> List[str]:
        """Normalize a class descriptor into sanitized tokens."""
        if not text:
            return []

        normalized = (
            text.lower()
            .replace('-', ' ')
            .replace('/', ' ')
            .replace('\\', ' ')
            .replace(':', ' ')
            .replace('.', ' ')
            .replace('_', ' ')
        )

        tokens = [token for token in normalized.split() if token]
        replacements = {
            'goalkeepers': 'goalkeeper',
            'keepers': 'goalkeeper',
            'keeper': 'goalkeeper',
            'goalie': 'goalkeeper',
            'referees': 'referee',
            'ref': 'referee',
        }

        sanitized: List[str] = []
        for token in tokens:
            token = replacements.get(token, token)
            if token.isdigit():
                continue
            if len(token) == 1 and token.isalpha():
                continue
            sanitized.append(token)

        return sanitized

    def _load_snmot_annotations(
        self,
        sequence_dir: Path,
        meta: Dict[str, Any]
    ) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
        """Load annotations for a sequence and convert to YOLO format."""
        gt_path = sequence_dir / "gt" / "gt.txt"
        if not gt_path.exists():
            logger.warning(f"Ground truth file missing for sequence {sequence_dir.name}: {gt_path}")
            return {}

        min_box = self.config.get('min_box_size', 0)
        snmot_cfg = self.config.get('snmot', {})
        min_conf = snmot_cfg.get('min_confidence', 0.0)
        clamp_boxes = snmot_cfg.get('clamp_boxes', True)

        width = meta['width']
        height = meta['height']
        tracklet_classes = meta['tracklet_classes']

        annotations: Dict[int, List[Tuple[int, float, float, float, float]]] = defaultdict(list)
        missing_tracklets: Set[int] = set()

        with open(gt_path, 'r', newline='') as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row or len(row) < 6:
                    continue

                try:
                    frame_id = int(float(row[0]))
                    track_id = int(float(row[1]))
                    x = float(row[2])
                    y = float(row[3])
                    w = float(row[4])
                    h = float(row[5])
                    conf = float(row[6]) if len(row) > 6 else 1.0
                except ValueError:
                    continue

                if conf < min_conf:
                    continue
                if w <= 0 or h <= 0:
                    continue

                class_name = tracklet_classes.get(track_id)
                if not class_name:
                    if track_id not in missing_tracklets:
                        missing_tracklets.add(track_id)
                    continue

                class_name = self._transform_class_name(class_name)
                if not class_name:
                    continue

                if min_box and (w < min_box or h < min_box):
                    continue

                if clamp_boxes:
                    x1 = max(0.0, min(x, width - 1))
                    y1 = max(0.0, min(y, height - 1))
                    x2 = max(0.0, min(x + w, width))
                    y2 = max(0.0, min(y + h, height))
                    w = x2 - x1
                    h = y2 - y1
                    x = x1
                    y = y1
                    if w <= 0 or h <= 0:
                        continue

                x_center = (x + w / 2.0) / width
                y_center = (y + h / 2.0) / height
                w_norm = w / width
                h_norm = h / height

                x_center = min(max(x_center, 0.0), 1.0)
                y_center = min(max(y_center, 0.0), 1.0)
                w_norm = min(max(w_norm, 0.0), 1.0)
                h_norm = min(max(h_norm, 0.0), 1.0)

                class_id = self._register_class_name(class_name)
                annotations[frame_id].append((class_id, x_center, y_center, w_norm, h_norm))

        if missing_tracklets:
            missing_preview = ", ".join(map(str, sorted(missing_tracklets)[:5]))
            if len(missing_tracklets) > 5:
                missing_preview += ", ..."
            logger.warning(
                "Sequence %s missing class metadata for track IDs: %s",
                meta['sequence_name'],
                missing_preview
            )

        return dict(annotations)

    def _transform_class_name(self, class_name: Optional[str]) -> Optional[str]:
        """Apply label transformation rules defined in configuration."""
        if not class_name:
            return None

        cfg = self._label_transform_cfg or {}
        if cfg.get('enabled') is False:
            return class_name

        original_name = class_name
        new_name: Optional[str] = None

        groups = cfg.get('groups') or {}
        for target_name, source_list in groups.items():
            if original_name in (source_list or []):
                new_name = target_name
                break

        if new_name is None:
            rename_map = cfg.get('rename') or {}
            if original_name in rename_map:
                new_name = rename_map[original_name]

        if new_name is None:
            default_group = cfg.get('default_group')
            if default_group and groups:
                new_name = default_group
            elif cfg.get('drop_unmatched'):
                return None
            else:
                new_name = original_name

        include_names = cfg.get('include')
        if include_names:
            include_set = set(include_names)
            if new_name not in include_set:
                return None

        exclude_names = cfg.get('exclude')
        if exclude_names and new_name in set(exclude_names):
            return None

        return new_name

    def _write_yolo_label_file(
        self,
        label_path: Path,
        labels: List[Tuple[int, float, float, float, float]]
    ):
        """Write YOLO labels to disk."""
        with open(label_path, 'w') as label_file:
            for cls_id, xc, yc, w, h in labels:
                label_file.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    def _class_names_from_labels(
        self,
        labels: List[Tuple[int, float, float, float, float]]
    ) -> Set[str]:
        names: Set[str] = set()
        for cls_id, *_ in labels:
            cls_name = self._get_class_name_for_id(cls_id)
            if cls_name:
                names.add(cls_name)
        return names

    def _get_class_name_for_id(self, class_id: int) -> Optional[str]:
        if 0 <= class_id < len(self._class_names):
            return self._class_names[class_id]
        return None

    def _instance_caps_enabled_for_split(self, split_name: Optional[str]) -> bool:
        if not split_name:
            return False
        cfg = self._instance_caps_cfg or {}
        if not cfg.get('enabled'):
            return False
        target_splits = cfg.get('target_splits')
        return not target_splits or split_name in target_splits

    def _apply_instance_caps(
        self,
        labels: List[Tuple[int, float, float, float, float]],
        split_name: str
    ) -> List[Tuple[int, float, float, float, float]]:
        if not labels:
            return labels

        class_caps = self._instance_caps_cfg.get('classes') or {}
        if not class_caps:
            return labels

        state = self._instance_caps_state.setdefault(
            split_name,
            defaultdict(int)
        )
        filtered: List[Tuple[int, float, float, float, float]] = []

        for label in labels:
            cls_id = label[0]
            class_name = self._get_class_name_for_id(cls_id)
            if not class_name:
                filtered.append(label)
                continue

            cap = class_caps.get(class_name)
            if cap is None:
                filtered.append(label)
                continue

            current = state[class_name]
            if current >= cap:
                continue

            state[class_name] = current + 1
            filtered.append(label)

        return filtered

    def _should_keep_frame(
        self,
        labels: List[Tuple[int, float, float, float, float]],
        label_names: Set[str],
        balance_cfg: Dict[str, Any]
    ) -> bool:
        """Decide whether to keep a frame based on balance configuration."""
        if not balance_cfg or not balance_cfg.get('enabled'):
            return True

        if not labels:
            empty_keep = float(balance_cfg.get('empty_keep_prob', 0.3))
            return random.random() < empty_keep

        minority_classes = set(balance_cfg.get('minority_classes') or [])
        if minority_classes and label_names & minority_classes:
            return True

        majority_classes = set(balance_cfg.get('majority_classes') or [])
        if majority_classes and label_names and label_names <= majority_classes:
            keep_prob = float(balance_cfg.get('majority_keep_prob', 0.35))
            return random.random() < keep_prob

        return True

    def _maybe_duplicate_minority_frame(
        self,
        labels: List[Tuple[int, float, float, float, float]],
        label_names: Set[str],
        src_image: Path,
        src_label: Path,
        images_out: Path,
        labels_out: Path,
        image_stem: str,
        image_ext: str,
        stats: Dict[str, int],
        balance_cfg: Dict[str, Any]
    ):
        if not balance_cfg or not balance_cfg.get('enabled'):
            return

        duplicate_count = int(balance_cfg.get('duplicate_minority', 0) or 0)
        if duplicate_count <= 0:
            return

        minority_classes = set(balance_cfg.get('minority_classes') or [])
        if not minority_classes or not (label_names & minority_classes):
            return

        for dup_idx in range(1, duplicate_count + 1):
            dup_suffix = f"_min{dup_idx:02d}"
            dup_image = images_out / f"{image_stem}{dup_suffix}{image_ext}"
            dup_label = labels_out / f"{image_stem}{dup_suffix}.txt"
            shutil.copy2(src_image, dup_image)
            shutil.copy2(src_label, dup_label)
            stats['frames'] += 1
            stats['labels'] += len(labels)

    def _register_class_name(self, class_name: str) -> int:
        """Register a class name and return its numeric identifier."""
        normalized = str(class_name).strip()
        if not normalized:
            raise ValueError("Class name cannot be empty")

        if normalized not in self._class_name_to_id:
            self._class_name_to_id[normalized] = len(self._class_names)
            self._class_names.append(normalized)

        return self._class_name_to_id[normalized]
    
    def _create_dataset_yaml(self, output_root: Path):
        """Create dataset.yaml configuration file."""
        class_names = self.config.get('class_names') or self._class_names
        if not class_names:
            class_names = ['player']
        
        config = {
            'path': str(output_root.absolute()),
            'train': self.paths_config.get('train_split', 'train'),
            'val': self.paths_config.get('val_split', 'val'),
            'test': self.paths_config.get('test_split', 'test'),
            'nc': len(class_names),
            'names': class_names
        }
        
        yaml_path = output_root / self.paths_config.get('dataset_yaml', 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        
        logger.info(f"Detected classes ({len(class_names)}): {class_names}")
        logger.info(f"Created dataset configuration: {yaml_path}")
    
    def ensure_dataset_yaml(self) -> Path:
        """Ensure dataset.yaml exists and return its path."""
        output_root = self._get_output_root()
        yaml_path = output_root / self.paths_config.get('dataset_yaml', 'dataset.yaml')
        
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"dataset.yaml not found at {yaml_path}. "
                f"Enable run_extraction or fix paths in configuration."
            )
        
        return yaml_path


def create_extractor(config: Dict[str, Any], paths_config: Dict[str, Any]) -> DatasetExtractor:
    """
    Factory function to create a dataset extractor.
    
    Args:
        config: Extraction configuration
        paths_config: Path configuration
        
    Returns:
        Configured DatasetExtractor instance
    """
    return DatasetExtractor(config, paths_config)