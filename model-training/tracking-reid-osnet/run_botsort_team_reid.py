"""Run BoT-SORT-style tracking with motion + ReID + team fusion.

Baseline features already implemented:
- Two-stage association: (IoU + appearance + team) then IoU-only recovery
- Inactive pool + ID reuse on new-track birth (relink)
- Multi-embedding memory per track (gallery)
- Team feature: torso crop + green mask + Lab[a,b] + 2-cluster KMeans
- Ball tracked separately (IoU-only single track)

NEW in this iteration:
- Dedicated inactive-pool reacquire pass (Hungarian) for true re-acquisition
- Configurable cut detection threshold + replay strict mode (tighten gates after cut)
- ReID embedding update policy (conf + box height + association similarity)
- Configurable gallery reduce: max/mean/median
- Spatial grid-bin penalty for reacquire (with replay scaling)
- Optional postprocess tracklet merging (writes a remapped CSV)

Not implemented:
- Jersey number module (out of scope)

Config:
- Default is `model-training/tracking/config.yaml` (run with zero args).
- Requires PyYAML: `pip install pyyaml`.

Example (Windows PowerShell):

```powershell
python .\model-training\tracking\run_botsort_team_reid.py

python .\model-training\tracking\run_botsort_team_reid.py `
    --config .\model-training\tracking\config.yaml `
    --video "C:\\path\\to\\match.mp4" `
    --device cuda:0 `
    --save_video .\outputs\tracked.mp4 `
    --save_txt .\outputs\tracks.csv
```

Output CSV columns:
frame_id,track_id,cls_id,conf,x1,y1,x2,y2,team_id,relinked,relink_source_id,relink_sim,relink_inactive_age
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from botsort_team_reid_tracker import BoTSORTTeamReIDTracker, Detection, OSNetReIDExtractor, ReIDExtractor
from util.config import apply_overrides, load_botsort_team_reid_config, parse_int_list
from team_clasifier import AutoLabEmbedder, AutomaticTeamClusterer


DEFAULT_DETECTOR = r"C:\Users\Admin\Desktop\FoMAC\FoMAC\model-training\ball-detection\models\player_ball_detector\weights\best.pt"
DEFAULT_REID = r"C:\Users\Admin\Desktop\sn-reid\sn-reid\log\model\model.pth.tar-60"
DEFAULT_CONFIG = str((THIS_DIR / "config.yaml").resolve())


def _set_determinism(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _infer_class_ids(model: YOLO) -> Tuple[int, int, int]:
    """Return (player_id, ball_id, referee_id) for your best.pt."""

    # Your best.pt reports: {0:'Player', 1:'Ball', 2:'Referee'}
    names = {int(k): str(v).lower() for k, v in model.names.items()}

    def find(substr: str, default: int) -> int:
        for k, v in names.items():
            if substr in v:
                return k
        return default

    player_id = find("player", 0)
    ball_id = find("ball", 1)
    referee_id = find("ref", 2)
    return player_id, ball_id, referee_id


def _cut_detect(prev_bgr: np.ndarray, cur_bgr: np.ndarray, thresh: float = 0.55) -> bool:
    """Simple shot boundary detector using HSV histogram correlation."""

    def hist(frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.resize(hsv, (320, 180), interpolation=cv2.INTER_AREA)
        h = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        h = cv2.normalize(h, h).flatten()
        return h

    h1 = hist(prev_bgr)
    h2 = hist(cur_bgr)
    corr = cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_CORREL)
    return corr < thresh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--detector_weights", type=str, default=None)
    ap.add_argument("--reid_weights", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--save_video", type=str, default=None)
    ap.add_argument("--save_txt", type=str, default=None)

    ap.add_argument("--w_iou", type=float, default=None)
    ap.add_argument("--w_app", type=float, default=None)
    ap.add_argument("--w_team", type=float, default=None)
    ap.add_argument("--iou_gate", type=float, default=None)
    ap.add_argument("--app_gate", type=float, default=None)
    ap.add_argument("--team_strict", type=int, default=None)

    ap.add_argument("--max_age", type=int, default=None)
    ap.add_argument("--min_hits", type=int, default=None)
    ap.add_argument("--alpha_embed", type=float, default=None)

    ap.add_argument("--cut_reset", type=int, default=None)
    ap.add_argument("--cut_mode", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)

    args = ap.parse_args()

    # Load config first (so we can run with zero args)
    config_path = Path(args.config).resolve()
    cfg = load_botsort_team_reid_config(str(config_path), base_dir=str(config_path.parent))

    overrides = {
        "video": args.video,
        "detector_weights": args.detector_weights,
        "reid_weights": args.reid_weights,
        "device": args.device,
        "save_video": args.save_video,
        "save_txt": args.save_txt,
        "w_iou": args.w_iou,
        "w_app": args.w_app,
        "w_team": args.w_team,
        "iou_gate": args.iou_gate,
        "app_gate": args.app_gate,
        "team_strict": (bool(args.team_strict) if args.team_strict is not None else None),
        "max_age": args.max_age,
        "min_hits": args.min_hits,
        "alpha_embed": args.alpha_embed,
        "cut_reset": (bool(args.cut_reset) if args.cut_reset is not None else None),
        "cut_mode": args.cut_mode,
        "seed": args.seed,
    }
    cfg = apply_overrides(cfg, overrides, base_dir=str(config_path.parent))

    _set_determinism(int(cfg.seed))

    def _require_file(p: str) -> Path:
        pp = Path(str(p))
        if not pp.exists():
            raise FileNotFoundError(str(pp))
        return pp

    video_path = _require_file(cfg.video)
    det_path = _require_file(cfg.detector_weights)
    reid_path = _require_file(cfg.reid_weights)

    device = str(cfg.device) if cfg.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available; falling back to cpu")
        device = "cpu"

    detector = YOLO(str(det_path))
    player_id, ball_id, referee_id = _infer_class_ids(detector)
    track_classes = (player_id, referee_id)

    reid = ReIDExtractor(
        str(reid_path),
        device=device,
        batch_size=int(getattr(cfg, "reid_batch_size", 64)),
        use_fp16=bool(getattr(cfg, "reid_fp16", True)),
    )

    def _resolve_extra_path(p: str) -> str:
        s = str(p or "").strip()
        if not s:
            return ""
        pp = Path(s)
        if not pp.is_absolute():
            pp = (config_path.parent / pp).resolve()
        return str(pp)

    osnet_enabled = bool(cfg.get("osnet.enabled", False))
    osnet = None
    if osnet_enabled:
        osnet_weights = _resolve_extra_path(str(cfg.get("osnet.weights", "")))
        if not osnet_weights or not Path(osnet_weights).exists():
            raise FileNotFoundError(f"OSNet enabled but weights not found: {osnet_weights!r}")
        osnet = OSNetReIDExtractor(
            osnet_weights,
            device=device,
            model_name=str(cfg.get("osnet.model_name", "osnet_x1_0")),
            batch_size=int(cfg.get("osnet.batch_size", 128)),
            use_fp16=bool(cfg.get("osnet.fp16", True)),
            input_hw=tuple(int(x) for x in (cfg.get("osnet.input_hw", [256, 128]) or [256, 128])),
        )

    tracker = BoTSORTTeamReIDTracker(
        w_iou=cfg.w_iou,
        w_app=cfg.w_app,
        w_team=cfg.w_team,
        iou_gate=cfg.iou_gate,
        app_gate=cfg.app_gate,
        team_strict=bool(cfg.team_strict),
        max_age=cfg.max_age,
        min_hits=cfg.min_hits,
        alpha_embed=cfg.alpha_embed,
        second_stage_iou=bool(getattr(cfg, "second_stage_iou", True)),
        iou_gate_second=float(getattr(cfg, "iou_gate_second", 0.2)),
        new_track_min_conf=float(getattr(cfg, "new_track_min_conf", 0.4)),
        track_classes=track_classes,
        ball_class=ball_id,

        relink_enabled=bool(getattr(cfg, "relink_enabled", True)),
        relink_max_age=int(getattr(cfg, "relink_max_age", 1800)),
        relink_app_gate=float(getattr(cfg, "relink_app_gate", 0.30)),
        relink_sim_margin=float(getattr(cfg, "relink_sim_margin", 0.05)),
        relink_team_strict=bool(getattr(cfg, "relink_team_strict", True)),
        relink_only_class=(player_id if bool(getattr(cfg, "relink_only_player", True)) else None),
        embed_gallery_size=int(cfg.get("reid.memory.size", int(getattr(cfg, "embed_gallery_size", 10))) if bool(cfg.get("reid.memory.enabled", True)) else 0),
        reid_memory_reduce=str(cfg.get("reid.memory.reduce", "max")),

        reid_update_min_det_conf=float(cfg.get("reid.update_policy.min_det_conf", 0.55)),
        reid_update_min_box_h=float(cfg.get("reid.update_policy.min_box_h", 80)),
        reid_update_min_sim_for_update=float(cfg.get("reid.update_policy.min_sim_for_update", 0.55)),

        team_penalize_classes=parse_int_list(str(getattr(cfg, "team_penalize_classes", "0"))),
        relink_max_center_dist_norm=float(cfg.get("spatial.hard_center_gate_norm", float(getattr(cfg, "relink_max_center_dist_norm", 0.0)))),

        reacquire_enabled=bool(cfg.get("reacquire.enabled", True)),
        reacquire_max_gap_frames=int(cfg.get("reacquire.max_gap_frames", 450)),
        reacquire_topk_candidates=int(cfg.get("reacquire.topk_candidates", 10)),
        reacquire_sim_gate=float(cfg.get("reacquire.sim_gate", 0.45)),
        reacquire_time_penalty=float(cfg.get("reacquire.time_penalty", 0.002)),

        spatial_enabled=bool(cfg.get("spatial.enabled", True)),
        spatial_grid=tuple(int(x) for x in (cfg.get("spatial.grid", [6, 3]) or [6, 3])),
        spatial_bin_penalty=float(cfg.get("spatial.bin_penalty", 0.15)),
        spatial_hard_center_gate_norm=float(cfg.get("spatial.hard_center_gate_norm", 0.60)),
        spatial_reduce_on_replay=bool(cfg.get("spatial.reduce_on_replay", True)),
        spatial_replay_scale=float(cfg.get("spatial.replay_scale", 0.5)),

        osnet_stage_enabled=osnet_enabled,
        osnet_app_gate=float(cfg.get("osnet.app_gate", 0.50)),
        osnet_iou_gate=float(cfg.get("osnet.iou_gate", 0.01)),
        osnet_w_iou=float(cfg.get("osnet.w_iou", 0.20)),
        osnet_w_app=float(cfg.get("osnet.w_app", 0.80)),
        osnet_w_team=float(cfg.get("osnet.w_team", 0.05)),
        osnet_update_min_sim_for_update=float(cfg.get("osnet.update_policy.min_sim_for_update", 0.55)),

        # Referee: keep one stable ID
        single_referee_id=bool(getattr(cfg, "single_referee_id", True)),
        referee_class_id=int(referee_id),
        referee_fixed_track_id=int(getattr(cfg, "referee_fixed_track_id", 9999)),
    )
    # --- Team calibration (same flow as existing run_tracker_with_teams_with_reid.py) ---
    embedder = AutoLabEmbedder()
    clusterer = AutomaticTeamClusterer(min_samples=80, seed=int(cfg.seed))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker.set_frame_size(w, h)

    # Collect jersey samples from early part of the match
    CALIB_SAMPLES = 600
    samples = 0
    frame_idx = 0
    while samples < CALIB_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 3 != 0:
            continue

        results = detector.predict(source=frame, verbose=False, device=device)
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            continue

        dets = results[0].boxes.data.detach().cpu().numpy()  # x1,y1,x2,y2,conf,cls
        for det in dets:
            cls_id = int(det[5])
            conf = float(det[4])
            if cls_id != player_id or conf < 0.70:
                continue
            x1, y1, x2, y2 = det[:4].astype(int)
            if (y2 - y1) < 50:
                continue
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            feat = embedder.get_features(crop)
            if feat is None:
                continue
            clusterer.collect(feat)
            samples += 1
            if samples >= CALIB_SAMPLES:
                break

    team_ready = clusterer.train()
    if not team_ready:
        print("⚠️  Team calibration failed (not enough jersey samples). Continuing with team_id=-1.")

    # Restart video
    cap.release()
    cap = cv2.VideoCapture(str(video_path))

    # Output writers
    out_vid = None
    if cfg.save_video:
        out_path = Path(cfg.save_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_vid = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))

    out_f = None
    if cfg.save_txt:
        out_path = Path(cfg.save_txt)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = open(out_path, "w", encoding="utf-8")
        out_f.write(
            "frame_id,track_id,cls_id,conf,x1,y1,x2,y2,team_id,relinked,relink_source_id,relink_sim,relink_inactive_age,assoc_stage,assoc_iou,assoc_app_sim,assoc_osnet_sim\n"
        )

    prev_frame = None
    frame_id = 0

    # Cut / replay mode
    cut_enabled = bool(cfg.get("cut_detect.enabled", bool(getattr(cfg, "cut_reset", False))))
    cut_threshold = float(cfg.get("cut_detect.threshold", 0.55))
    cut_min_interval = int(cfg.get("cut_detect.min_interval_frames", 15))
    cut_mode = str(cfg.get("cut_detect.mode", str(getattr(cfg, "cut_mode", "reset")))).lower().strip()
    last_cut_frame = -10**9

    replay_enabled = bool(cfg.get("replay_mode.enabled", False))
    replay_frames = int(cfg.get("replay_mode.frames", 45))
    replay_stricter = cfg.get("replay_mode.stricter", {})
    if not isinstance(replay_stricter, dict):
        replay_stricter = {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    t0 = time.time()
    last_print_t = t0
    progress_every_sec = float(getattr(cfg, "progress_every_sec", 5.0))

    referee_team_id = int(getattr(cfg, "referee_team_id", 2))
    max_frames = int(getattr(cfg, "max_frames", 0) or 0)

    # Tracklet summaries for optional postprocess merging
    tracklets: Dict[int, Dict[str, object]] = {}

    def jersey_color_bgr(crop_bgr: np.ndarray) -> Tuple[int, int, int]:
        """Estimate jersey color for drawing (torso + green mask)."""
        if crop_bgr is None or crop_bgr.size == 0:
            return (180, 180, 180)
        hh, ww = crop_bgr.shape[:2]
        if hh < 10 or ww < 10:
            return (180, 180, 180)

        y1 = int(hh * 0.15)
        y2 = int(hh * 0.60)
        x1 = int(ww * 0.20)
        x2 = int(ww * 0.80)
        if (y2 - y1) < 5 or (x2 - x1) < 5:
            roi = crop_bgr
        else:
            roi = crop_bgr[y1:y2, x1:x2]
            if roi.size == 0:
                roi = crop_bgr

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_green = np.array([30, 40, 40], dtype=np.uint8)
        upper_green = np.array([90, 255, 255], dtype=np.uint8)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        not_green = cv2.bitwise_not(mask_green)
        valid = roi[not_green > 0]
        if valid.shape[0] < 10:
            return (180, 180, 180)
        bgr = np.mean(valid.astype(np.float32), axis=0)
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    color_map = {0: (255, 100, 0), 1: (0, 0, 255), -1: (200, 200, 200)}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if max_frames > 0 and frame_id > max_frames:
            break

        now = time.time()
        if progress_every_sec > 0 and (now - last_print_t) >= progress_every_sec:
            elapsed = max(1e-6, now - t0)
            fps_proc = frame_id / elapsed
            if total_frames > 0:
                pct = 100.0 * (frame_id / total_frames)
                eta_sec = max(0.0, (total_frames - frame_id) / max(1e-6, fps_proc))
                eta_h = int(eta_sec // 3600)
                eta_m = int((eta_sec % 3600) // 60)
                eta_s = int(eta_sec % 60)
                print(f"[{frame_id}/{total_frames}] {pct:.2f}% | {fps_proc:.2f} FPS | ETA {eta_h:02d}:{eta_m:02d}:{eta_s:02d}")
            else:
                print(f"[{frame_id}] {fps_proc:.2f} FPS")
            last_print_t = now

        if cut_enabled and prev_frame is not None:
            if (frame_id - last_cut_frame) >= cut_min_interval and _cut_detect(prev_frame, frame, thresh=cut_threshold):
                last_cut_frame = frame_id
                if cut_mode == "inactive":
                    tracker.cut_to_inactive()
                else:
                    tracker.reset()
                if replay_enabled and replay_frames > 0:
                    tracker.enter_replay_mode(replay_frames, stricter=replay_stricter)  # tightens gates for N frames
        prev_frame = frame

        results = detector.predict(source=frame, verbose=False, device=device)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            dets_np = np.empty((0, 6), dtype=np.float32)
        else:
            dets_np = boxes.data.detach().cpu().numpy().astype(np.float32)

        dets: List[Detection] = []
        crops_for_reid: List[np.ndarray] = []
        reid_indices: List[int] = []

        # ReID speed knobs
        reid_every_n = max(1, int(getattr(cfg, "reid_every_n", 1)))
        do_reid_this_frame = (frame_id % reid_every_n) == 0
        reid_min_conf = float(getattr(cfg, "reid_min_conf", 0.25))
        reid_topk = max(1, int(getattr(cfg, "reid_topk", 60)))

        # Pre-select indices to compute ReID for (top-K by conf)
        if do_reid_this_frame and dets_np.shape[0] > 0:
            cand = []
            for i in range(dets_np.shape[0]):
                conf = float(dets_np[i][4])
                cls_id = int(dets_np[i][5])
                if cls_id in track_classes and conf >= reid_min_conf:
                    cand.append((conf, i))
            cand.sort(key=lambda x: x[0], reverse=True)
            reid_pick = {i for _, i in cand[:reid_topk]}
        else:
            reid_pick = set()

        # If relinking is enabled, make sure we compute embeddings for detections
        # that are likely to be new tracks on this frame (and for border-touching re-entries).
        if do_reid_this_frame and bool(getattr(cfg, "relink_enabled", True)) and dets_np.shape[0] > 0:
            border_px = 25.0
            new_track_min_conf = float(getattr(cfg, "new_track_min_conf", 0.4))

            def _touches_border_xyxy(x1f: float, y1f: float, x2f: float, y2f: float) -> bool:
                return (
                    (x1f <= border_px)
                    or (y1f <= border_px)
                    or (x2f >= (float(w) - border_px))
                    or (y2f >= (float(h) - border_px))
                )

            for i in range(dets_np.shape[0]):
                x1f, y1f, x2f, y2f, conf, cls = dets_np[i]
                cls_id = int(cls)
                if cls_id != player_id:
                    continue
                if float(conf) < reid_min_conf:
                    continue
                if float(conf) >= new_track_min_conf or _touches_border_xyxy(float(x1f), float(y1f), float(x2f), float(y2f)):
                    reid_pick.add(i)

        for i in range(dets_np.shape[0]):
            x1, y1, x2, y2, conf, cls = dets_np[i]
            cls_id = int(cls)
            bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

            # Team label via existing approach (Lab feature + 2-cluster model)
            team_id = -1
            emb = None

            if cls_id in track_classes:
                xi1, yi1, xi2, yi2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
                crop = frame[yi1:yi2, xi1:xi2]

                if cls_id == referee_id:
                    team_id = referee_team_id
                else:
                    feat = embedder.get_features(crop)
                    team_id = clusterer.predict(feat) if team_ready else -1

                if i in reid_pick:
                    crops_for_reid.append(crop)
                    reid_indices.append(len(dets))

            dets.append(Detection(bbox_xyxy=bbox, conf=float(conf), cls_id=cls_id, team_id=int(team_id), embedding=emb))

        if crops_for_reid:
            embs = reid.extract(crops_for_reid)
            for k, det_idx in enumerate(reid_indices):
                dets[det_idx].embedding = embs[k]

            if osnet is not None:
                embs2 = osnet.extract(crops_for_reid)
                for k, det_idx in enumerate(reid_indices):
                    dets[det_idx].embedding_osnet = embs2[k]

        confirmed = tracker.update(dets)

        # Write results + draw
        if out_f is not None or out_vid is not None:
            draw = frame.copy() if out_vid is not None else frame

            for t in confirmed:
                x1, y1, x2, y2 = t.bbox_xyxy.astype(int)
                team_id = int(t.team_id)
                if int(t.cls_id) == referee_id:
                    xi1, yi1, xi2, yi2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                    crop = frame[yi1:yi2, xi1:xi2]
                    color = jersey_color_bgr(crop)
                else:
                    color = color_map.get(team_id, (200, 200, 200))

                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    draw,
                    (f"ID:{t.track_id} R" if int(t.cls_id) == referee_id else f"ID:{t.track_id} T:{team_id}"),
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                if out_f is not None:
                    relinked = 0
                    relink_source_id = -1
                    relink_sim = 0.0
                    relink_inactive_age = -1
                    if getattr(t, "relink_source_id", -1) != -1 and not bool(getattr(t, "relink_reported", False)):
                        relinked = 1
                        relink_source_id = int(getattr(t, "relink_source_id", -1))
                        relink_sim = float(getattr(t, "relink_sim", 0.0))
                        relink_inactive_age = int(getattr(t, "relink_inactive_age", -1))
                        t.relink_reported = True

                    assoc_stage = str(getattr(t, "last_assoc_stage", ""))
                    assoc_iou = float(getattr(t, "last_assoc_iou", 0.0))
                    assoc_app_sim = float(getattr(t, "last_assoc_app_sim", 0.0))
                    assoc_osnet_sim = float(getattr(t, "last_assoc_osnet_sim", 0.0))
                    out_f.write(
                        f"{frame_id},{t.track_id},{t.cls_id},{t.conf:.4f},{x1},{y1},{x2},{y2},{team_id},{relinked},{relink_source_id},{relink_sim:.4f},{relink_inactive_age},{assoc_stage},{assoc_iou:.4f},{assoc_app_sim:.4f},{assoc_osnet_sim:.4f}\n"
                    )

                # Update tracklet summary (for postprocess merging)
                rec = tracklets.get(int(t.track_id))
                if rec is None:
                    rec = {
                        "start": int(frame_id),
                        "end": int(frame_id),
                        "cls": int(t.cls_id),
                        "team": int(team_id),
                        "emb": None,
                    }
                    tracklets[int(t.track_id)] = rec
                else:
                    rec["end"] = int(frame_id)
                    rec["cls"] = int(t.cls_id)
                    rec["team"] = int(team_id)

                emb = getattr(t, "embedding_ema", None)
                if emb is not None:
                    e = emb.astype(np.float32)
                    e = e / (np.linalg.norm(e) + 1e-6)
                    rec["emb"] = e

            bt = tracker.get_ball_track()
            if bt is not None:
                x1, y1, x2, y2 = bt.bbox_xyxy.astype(int)
                cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(draw, "Ball", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if out_vid is not None:
                out_vid.write(draw)

    cap.release()
    if out_vid is not None:
        out_vid.release()
    if out_f is not None:
        out_f.close()

    # Optional postprocess: merge tracklets and rewrite IDs
    if cfg.save_txt and bool(cfg.get("postprocess.merge_tracklets", False)):
        src = Path(cfg.save_txt)
        if src.exists():
            merged = src.with_name(src.stem + "_merged" + src.suffix)
            merge_sim_gate = float(cfg.get("postprocess.merge_sim_gate", 0.60))
            merge_max_gap = int(cfg.get("postprocess.merge_max_gap_frames", 300))

            # Union-find merge using team/class/gap + embedding similarity
            parent: Dict[int, int] = {tid: tid for tid in tracklets.keys()}

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a: int, b: int) -> None:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            items = sorted(tracklets.items(), key=lambda kv: int(kv[1].get("start", 0)))
            for i in range(len(items)):
                id_i, ti = items[i]
                emb_i = ti.get("emb", None)
                for j in range(i + 1, len(items)):
                    id_j, tj = items[j]
                    gap = int(int(tj.get("start", 0)) - int(ti.get("end", 0)))
                    if gap < 0:
                        continue
                    if gap > merge_max_gap:
                        break
                    if int(ti.get("cls", -1)) != int(tj.get("cls", -2)):
                        continue

                    emb_j = tj.get("emb", None)
                    if emb_i is None or emb_j is None:
                        continue

                    # Players: require same known team
                    if int(ti.get("cls", -1)) == player_id:
                        if int(ti.get("team", -1)) == -1 or int(tj.get("team", -1)) == -1:
                            continue
                        if int(ti.get("team", -1)) != int(tj.get("team", -1)):
                            continue

                    sim = float(np.dot(np.asarray(emb_i, dtype=np.float32), np.asarray(emb_j, dtype=np.float32)))
                    if sim >= merge_sim_gate:
                        union(id_i, id_j)

            # Write remapped CSV
            with open(src, "r", newline="", encoding="utf-8") as fin, open(merged, "w", newline="", encoding="utf-8") as fout:
                r = csv.DictReader(fin)
                w = csv.DictWriter(fout, fieldnames=r.fieldnames)
                w.writeheader()
                for row in r:
                    tid = int(row["track_id"])
                    row["track_id"] = str(find(tid))
                    w.writerow(row)


if __name__ == "__main__":
    main()
