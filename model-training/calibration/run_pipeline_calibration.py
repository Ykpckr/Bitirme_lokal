import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
# Ensure local calibration modules are importable when invoked from repo root.
sys.path.insert(0, str(THIS_DIR))


@dataclass
class CalibrationOutputs:
    map_video_path: str
    events_json_path: str
    frames_jsonl_path: str


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _load_hrnet_models(*, device: str, kp_weights: str, line_weights: str):
    import torch
    import torchvision.transforms as T

    from nbjw_calib.model.cls_hrnet import get_cls_net
    from nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l

    # Minimal configs matching demo.py
    cfg_kp = {
        "MODEL": {
            "IMAGE_SIZE": [960, 540],
            "NUM_JOINTS": 58,
            "PRETRAIN": "",
            "EXTRA": {
                "FINAL_CONV_KERNEL": 1,
                "STAGE1": {"NUM_MODULES": 1, "NUM_BRANCHES": 1, "BLOCK": "BOTTLENECK", "NUM_BLOCKS": [4], "NUM_CHANNELS": [64], "FUSE_METHOD": "SUM"},
                "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": "BASIC", "NUM_BLOCKS": [4, 4], "NUM_CHANNELS": [48, 96], "FUSE_METHOD": "SUM"},
                "STAGE3": {"NUM_MODULES": 4, "NUM_BRANCHES": 3, "BLOCK": "BASIC", "NUM_BLOCKS": [4, 4, 4], "NUM_CHANNELS": [48, 96, 192], "FUSE_METHOD": "SUM"},
                "STAGE4": {"NUM_MODULES": 3, "NUM_BRANCHES": 4, "BLOCK": "BASIC", "NUM_BLOCKS": [4, 4, 4, 4], "NUM_CHANNELS": [48, 96, 192, 384], "FUSE_METHOD": "SUM"},
            },
        }
    }
    cfg_lines = {
        "MODEL": {
            "IMAGE_SIZE": [960, 540],
            "NUM_JOINTS": 24,
            "PRETRAIN": "",
            "EXTRA": {
                "FINAL_CONV_KERNEL": 1,
                "STAGE1": {"NUM_MODULES": 1, "NUM_BRANCHES": 1, "BLOCK": "BOTTLENECK", "NUM_BLOCKS": [4], "NUM_CHANNELS": [64], "FUSE_METHOD": "SUM"},
                "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": "BASIC", "NUM_BLOCKS": [4, 4], "NUM_CHANNELS": [48, 96], "FUSE_METHOD": "SUM"},
                "STAGE3": {"NUM_MODULES": 4, "NUM_BRANCHES": 3, "BLOCK": "BASIC", "NUM_BLOCKS": [4, 4, 4], "NUM_CHANNELS": [48, 96, 192], "FUSE_METHOD": "SUM"},
                "STAGE4": {"NUM_MODULES": 3, "NUM_BRANCHES": 4, "BLOCK": "BASIC", "NUM_BLOCKS": [4, 4, 4, 4], "NUM_CHANNELS": [48, 96, 192, 384], "FUSE_METHOD": "SUM"},
            },
        }
    }

    dev = torch.device(device)

    model_kp = get_cls_net(cfg_kp)
    sd = torch.load(kp_weights, map_location=dev)
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model_kp.load_state_dict(sd)
    model_kp.to(dev)
    model_kp.eval()

    model_lines = get_cls_net_l(cfg_lines)
    sd2 = torch.load(line_weights, map_location=dev)
    if isinstance(sd2, dict) and any(k.startswith("module.") for k in sd2.keys()):
        sd2 = {k.replace("module.", ""): v for k, v in sd2.items()}
    model_lines.load_state_dict(sd2)
    model_lines.to(dev)
    model_lines.eval()

    tfms_resize = T.Compose([T.Resize((540, 960)), T.ToTensor()])

    return model_kp, model_lines, tfms_resize, dev


def _pitch_background(*, scale: int = 8, margin: int = 50) -> np.ndarray:
    # Match demo.py geometry
    pitch_length = 105
    pitch_width = 68
    img_width = int(pitch_length * scale + 2 * margin)
    img_height = int(pitch_width * scale + 2 * margin)

    bg = np.ones((img_height, img_width, 3), dtype=np.uint8) * 50
    bg[:, :, 1] = 100

    try:
        from sn_calibration_baseline.soccerpitch import SoccerPitch

        field = SoccerPitch()

        def world_to_minimap(pt3d: np.ndarray) -> Tuple[int, int]:
            mx = int((float(pt3d[0]) + pitch_length / 2) * scale + margin)
            my = int((float(pt3d[1]) + pitch_width / 2) * scale + margin)
            return mx, my

        for line in field.sample_field_points():
            pts = [world_to_minimap(np.asarray(p)) for p in line]
            cv2.polylines(bg, [np.array(pts, dtype=np.int32)], False, (255, 255, 255), 2)
    except Exception:
        # Best-effort: keep a plain pitch background.
        pass

    return bg


def _draw_map_frame(
    *,
    background: np.ndarray,
    camera: Any,
    persons: List[Tuple[np.ndarray, int, int]],
    balls: List[np.ndarray],
    scale: int = 8,
    margin: int = 50,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return (frame_bgr, meta) where meta contains projected world coords."""
    frame = background.copy()

    pitch_length = 105
    pitch_width = 68

    def world_to_minimap(pt3d: np.ndarray) -> Tuple[int, int]:
        mx = int((float(pt3d[0]) + pitch_length / 2) * scale + margin)
        my = int((float(pt3d[1]) + pitch_width / 2) * scale + margin)
        return mx, my

    players_out: List[Dict[str, Any]] = []

    # Players
    for bbox_xyxy, team_id, track_id in persons:
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox_xyxy.tolist()]
        except Exception:
            continue
        foot_x = (x1 + x2) / 2.0
        foot_y = float(y2)
        try:
            ground_pt = camera.unproject_point_on_planeZ0(np.array([foot_x, foot_y], dtype=np.float32))
            ground_pt = np.asarray(ground_pt).reshape(-1)
        except Exception:
            continue

        # Sanity check: keep roughly inside pitch bounds
        if not (abs(float(ground_pt[0])) < 120 and abs(float(ground_pt[1])) < 90):
            continue

        mx, my = world_to_minimap(ground_pt)
        # Dot color by team
        dot_c = (200, 200, 200)
        if int(team_id) == 0:
            dot_c = (255, 100, 100)
        elif int(team_id) == 1:
            dot_c = (100, 100, 255)

        cv2.circle(frame, (mx, my), 6, (255, 255, 255), -1)
        cv2.circle(frame, (mx, my), 4, dot_c, -1)

        players_out.append(
            {
                "track_id": int(track_id),
                "team_id": int(team_id),
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "world_xy": [float(ground_pt[0]), float(ground_pt[1])],
            }
        )

    # Ball (use first ball)
    ball_out: Optional[Dict[str, Any]] = None
    if balls:
        bbox = balls[0]
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox.tolist()]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0 + (y2 - y1) * 0.4
            ground_pt = camera.unproject_point_on_planeZ0(np.array([cx, cy], dtype=np.float32))
            ground_pt = np.asarray(ground_pt).reshape(-1)
            if abs(float(ground_pt[0])) < 120 and abs(float(ground_pt[1])) < 90:
                mx, my = world_to_minimap(ground_pt)
                cv2.circle(frame, (mx, my), 6, (0, 0, 0), -1)
                cv2.circle(frame, (mx, my), 4, (0, 165, 255), -1)
                ball_out = {
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "world_xy": [float(ground_pt[0]), float(ground_pt[1])],
                }
        except Exception:
            ball_out = None

    return frame, {"players": players_out, "ball": ball_out}


def run(
    *,
    source: str,
    out_map: str,
    out_events: str,
    out_frames: Optional[str],
    detector_weights: str,
    kp_weights: str,
    line_weights: str,
    conf_thres: float = 0.30,
    frames_stride: int = 1,
    progress_every: int = 25,
    max_frames: int = 0,
) -> CalibrationOutputs:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isfile(kp_weights):
        raise FileNotFoundError(f"Missing keypoints weights: {kp_weights}")
    if not os.path.isfile(line_weights):
        raise FileNotFoundError(f"Missing lines weights: {line_weights}")
    if not os.path.isfile(detector_weights):
        raise FileNotFoundError(f"Missing detector weights: {detector_weights}")

    model_kp, model_lines, tfms_resize, dev = _load_hrnet_models(
        device=device, kp_weights=kp_weights, line_weights=line_weights
    )

    from nbjw_calib.utils.utils_heatmap import (
        complete_keypoints,
        coords_to_dict,
        get_keypoints_from_heatmap_batch_maxpool,
        get_keypoints_from_heatmap_batch_maxpool_l,
    )
    from nbjw_calib.utils.utils_calib import FramebyFrameCalib
    from sn_calibration_baseline.camera import Camera

    # Detector
    from ultralytics import YOLO

    det = YOLO(detector_weights)

    # Optional tracker/team classifier (best-effort)
    # Prefer boxmot ByteTrack if available; otherwise use Ultralytics built-in tracker via YOLO.track(persist=True).
    tracker = None
    tracking_backend = "none"
    try:
        from boxmot import ByteTrack  # type: ignore

        tracker = ByteTrack(
            reid_weights=None,
            device=("cuda:0" if device == "cuda" else "cpu"),
            half=(device == "cuda"),
            frame_rate=25,
        )
        tracking_backend = "boxmot_bytetrack"
    except Exception:
        tracker = None
        tracking_backend = "ultralytics_track"

    embedder = None
    clusterer = None
    try:
        from team_clasifier import AutoLabEmbedder, AutomaticTeamClusterer

        embedder = AutoLabEmbedder()
        clusterer = AutomaticTeamClusterer()
    except Exception:
        embedder = None
        clusterer = None

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Stats for debugging/diagnosis
    stats = {
        "frames": 0,
        "calibration_ok_frames": 0,
        "frames_with_person_det": 0,
        "frames_with_ball_det": 0,
        "projected_player_points": 0,
        "projected_ball_points": 0,
        "events": 0,
        "tracking_backend": tracking_backend,
        "tracker_ok_frames": 0,
        "tracks_total": 0,
        "team_cluster_trained": False,
        "team_cluster_collected": 0,
    }

    try:
        progress_every = int(progress_every)
    except Exception:
        progress_every = 25
    if progress_every < 1:
        progress_every = 25

    try:
        max_frames = int(max_frames)
    except Exception:
        max_frames = 0
    if max_frames < 0:
        max_frames = 0

    # Prepare map writer (fixed pitch canvas)
    scale = 8
    margin = 50
    bg = _pitch_background(scale=scale, margin=margin)
    map_h, map_w = int(bg.shape[0]), int(bg.shape[1])

    out_map_p = Path(out_map)
    out_map_p.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_map_p), fourcc, fps, (map_w, map_h))
    if not vw.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create map video writer: {out_map}")

    write_frames = bool(out_frames and str(out_frames).strip())
    out_frames_p: Optional[Path] = None
    if write_frames:
        out_frames_p = Path(str(out_frames)).resolve()
        out_frames_p.parent.mkdir(parents=True, exist_ok=True)

    try:
        frames_stride = int(frames_stride)
    except Exception:
        frames_stride = 1
    if frames_stride < 1:
        frames_stride = 1

    # Possession/pass event state
    events: List[Dict[str, Any]] = []
    last_possessor: Optional[Dict[str, Any]] = None
    potential_possessor: Optional[Dict[str, Any]] = None
    potential_frames = 0
    frames_since_loss = 0

    # Track -> team mapping (filled once clustering is trained)
    team_by_track: Dict[int, int] = {}

    frame_idx = 0
    f_frames = None
    try:
        if write_frames and out_frames_p is not None:
            f_frames = open(out_frames_p, "w", encoding="utf-8")

        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            if max_frames > 0 and frame_idx >= max_frames:
                break

            stats["frames"] += 1

            if (frame_idx % progress_every) == 0:
                try:
                    msg = "Calibration çalışıyor"
                    if total_frames > 0:
                        pct = int((float(frame_idx) * 100.0) / float(total_frames))
                        msg = f"Calibration: {pct}%"
                    print(
                        "__PROGRESS__ "
                        + json.dumps(
                            {
                                "stage": "calibration",
                                "current": int(frame_idx),
                                "total": int(total_frames) if total_frames > 0 else 0,
                                "message": msg,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                except Exception:
                    pass

            t_sec = float(frame_idx) / float(max(1e-6, fps))

            person_dets: List[np.ndarray] = []
            ball_boxes: List[np.ndarray] = []

            # Detection + tracking
            if tracking_backend == "ultralytics_track":
                try:
                    # YOLO.track runs detection internally and attaches IDs when possible.
                    tr_res = det.track(frame_bgr, verbose=False, persist=True, conf=float(conf_thres), tracker="bytetrack.yaml")
                except Exception:
                    tr_res = None

                if tr_res and tr_res[0] is not None and getattr(tr_res[0], "boxes", None) is not None:
                    b = tr_res[0].boxes
                    try:
                        xyxy = b.xyxy.detach().cpu().numpy() if getattr(b, "xyxy", None) is not None else np.zeros((0, 4), dtype=np.float32)
                        confs = b.conf.detach().cpu().numpy() if getattr(b, "conf", None) is not None else np.zeros((xyxy.shape[0],), dtype=np.float32)
                        clss = b.cls.detach().cpu().numpy() if getattr(b, "cls", None) is not None else np.zeros((xyxy.shape[0],), dtype=np.float32)
                        ids = None
                        try:
                            ids = b.id.detach().cpu().numpy() if getattr(b, "id", None) is not None else None
                        except Exception:
                            ids = None
                    except Exception:
                        xyxy = np.zeros((0, 4), dtype=np.float32)
                        confs = np.zeros((0,), dtype=np.float32)
                        clss = np.zeros((0,), dtype=np.float32)
                        ids = None

                    # Build YOLO-like det rows for persons so we can reuse the ByteTrack path.
                    for i in range(int(xyxy.shape[0])):
                        try:
                            cls_id = int(clss[i])
                            c = float(confs[i])
                            if c < float(conf_thres):
                                continue
                            x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
                            if cls_id == 0:
                                tid = -1
                                if ids is not None and i < len(ids):
                                    try:
                                        tid = int(ids[i])
                                    except Exception:
                                        tid = -1
                                person_dets.append(np.array([x1, y1, x2, y2, c, float(cls_id), float(tid)], dtype=np.float32))
                            elif cls_id == 1:
                                ball_boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
                        except Exception:
                            continue
            else:
                # Plain YOLO detect (and optional boxmot tracking)
                pil_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = det(pil_rgb, verbose=False)
                boxes = (
                    results[0].boxes.data.cpu().numpy()
                    if results and results[0].boxes is not None
                    else np.zeros((0, 6), dtype=np.float32)
                )

                for bb in boxes:
                    if float(bb[4]) < float(conf_thres):
                        continue
                    cls_id = int(bb[5])
                    if cls_id == 0:
                        person_dets.append(bb)
                    elif cls_id == 1:
                        ball_boxes.append(bb[:4].copy())

            if len(person_dets) > 0:
                stats["frames_with_person_det"] += 1
            if len(ball_boxes) > 0:
                stats["frames_with_ball_det"] += 1

            # Tracking + team id
            persons: List[Tuple[np.ndarray, int, int]] = []
            if tracking_backend == "ultralytics_track" and len(person_dets) > 0:
                stats["tracker_ok_frames"] += 1
                # person_dets rows are [x1,y1,x2,y2,conf,cls,track_id]
                for d in person_dets:
                    try:
                        x1, y1, x2, y2 = [int(v) for v in d[:4].tolist()]
                        tid = int(float(d[6])) if len(d) >= 7 else -1
                    except Exception:
                        continue

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)

                    team_id = -1
                    if embedder is not None and clusterer is not None and (y2 - y1) > 5 and (x2 - x1) > 5:
                        crop = frame_bgr[y1:y2, x1:x2]
                        feat = None
                        try:
                            feat = embedder.get_features(crop)
                        except Exception:
                            feat = None

                        if feat is not None:
                            try:
                                if not clusterer.trained:
                                    # Collect a bit faster by sampling every 3 frames.
                                    if (frame_idx % 3) == 0:
                                        clusterer.collect(feat)
                                        stats["team_cluster_collected"] = int(stats.get("team_cluster_collected", 0)) + 1
                                        if len(getattr(clusterer, "data_bank", [])) >= 50:
                                            clusterer.train()
                                if clusterer.trained:
                                    team_id = int(clusterer.predict(feat))
                            except Exception:
                                team_id = -1

                    persons.append((np.array([x1, y1, x2, y2], dtype=np.int32), int(team_id), int(tid)))
                    if int(tid) != -1 and int(team_id) != -1:
                        team_by_track[int(tid)] = int(team_id)
                    try:
                        stats["tracks_total"] += 1
                    except Exception:
                        pass

            elif tracker is not None and len(person_dets) > 0:
                try:
                    tracks = tracker.update(np.array(person_dets), frame_bgr)
                except Exception:
                    tracks = None
                if tracks is not None:
                    stats["tracker_ok_frames"] += 1
                    for tr in tracks:
                        bbox = tr[:4].astype(int)
                        tid = int(tr[4])
                        x1, y1, x2, y2 = [int(v) for v in bbox.tolist()]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)
                        team_id = -1
                        if embedder is not None and clusterer is not None and (y2 - y1) > 5 and (x2 - x1) > 5:
                            crop = frame_bgr[y1:y2, x1:x2]
                            try:
                                feat = embedder.get_features(crop)
                            except Exception:
                                feat = None
                            if feat is not None:
                                try:
                                    if not clusterer.trained:
                                        clusterer.collect(feat)
                                        if len(getattr(clusterer, "data_bank", [])) >= 50:
                                            clusterer.train()
                                    if clusterer.trained:
                                        team_id = int(clusterer.predict(feat))
                                except Exception:
                                    team_id = -1
                        persons.append((np.array([x1, y1, x2, y2], dtype=np.int32), int(team_id), int(tid)))
                        if int(tid) != -1 and int(team_id) != -1:
                            team_by_track[int(tid)] = int(team_id)
                        try:
                            stats["tracks_total"] += 1
                        except Exception:
                            pass
            else:
                for d in person_dets:
                    bb = d[:4].astype(int)
                    persons.append((bb, -1, -1))

            try:
                if clusterer is not None:
                    stats["team_cluster_trained"] = bool(getattr(clusterer, "trained", False))
            except Exception:
                pass

            # HRNet keypoints/lines -> camera params
            try:
                import torch

                img_pil = ImageFromBGR(frame_bgr)
                img_tensor = tfms_resize(img_pil).unsqueeze(0).to(dev)
                with torch.no_grad():
                    heatmaps = model_kp(img_tensor)
                    heatmaps_l = model_lines(img_tensor)

                kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
                line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])

                kp_dict = coords_to_dict(kp_coords, threshold=0.05)
                lines_dict = coords_to_dict(line_coords, threshold=0.05)
                final_dict = complete_keypoints(kp_dict, lines_dict, w=960, h=540, normalize=True)
                keypoints_prediction = final_dict[0]

                cam = FramebyFrameCalib(iwidth=width, iheight=height, denormalize=True)
                cam.update(keypoints_prediction)
                calib_res = cam.heuristic_voting()

                if calib_res and math.isnan(_safe_float(calib_res.get("rep_err", 0.0), 0.0)):
                    calib_res = False
            except Exception:
                calib_res = False

            map_frame = bg.copy()
            meta_frame: Dict[str, Any] = {"players": [], "ball": None}
            rep_err = None

            if calib_res:
                try:
                    params = calib_res["cam_params"]
                    rep_err = _safe_float(calib_res.get("rep_err", None), None) if calib_res.get("rep_err") is not None else None

                    camera = Camera(iwidth=width, iheight=height)
                    camera.from_json_parameters(params)

                    map_frame, meta_frame = _draw_map_frame(
                        background=bg, camera=camera, persons=persons, balls=ball_boxes, scale=scale, margin=margin
                    )

                    try:
                        stats["calibration_ok_frames"] += 1
                        stats["projected_player_points"] += int(len((meta_frame.get("players") or [])))
                        stats["projected_ball_points"] += int(1 if (meta_frame.get("ball") is not None) else 0)
                    except Exception:
                        pass

                    # Possession/pass events (best-effort)
                    ball_pos = None
                    if meta_frame.get("ball") and meta_frame["ball"].get("world_xy"):
                        try:
                            ball_pos = np.array(meta_frame["ball"]["world_xy"], dtype=np.float32)
                        except Exception:
                            ball_pos = None

                    current_possessor = None
                    if ball_pos is not None:
                        min_dist = 1e9
                        for p in meta_frame.get("players") or []:
                            try:
                                team_id = int(p.get("team_id", -1))
                                track_id = int(p.get("track_id", -1))
                                wxy = p.get("world_xy")
                                if wxy is None:
                                    continue
                                ppos = np.array(wxy, dtype=np.float32)
                                dist = float(np.linalg.norm(ppos - ball_pos))
                                # Allow team_id=-1 (unknown) as long as tracking id exists.
                                if dist < 2.5 and dist < min_dist and track_id != -1:
                                    min_dist = dist
                                    current_possessor = {"track_id": track_id, "team_id": team_id, "pos": ppos}
                            except Exception:
                                continue

                    if current_possessor is not None:
                        frames_since_loss = 0
                        if last_possessor is None:
                            last_possessor = current_possessor
                            events.append(
                                {
                                    "t": t_sec,
                                    "source": "calibration",
                                    "type": "possession_start",
                                    "team_id": int(current_possessor["team_id"]),
                                    "player_track_id": int(current_possessor["track_id"]),
                                }
                            )
                        else:
                            is_same_team = int(last_possessor["team_id"]) == int(current_possessor["team_id"])
                            is_different_id = int(last_possessor["track_id"]) != int(current_possessor["track_id"])
                            try:
                                dist_between = float(np.linalg.norm(last_possessor["pos"] - current_possessor["pos"]))
                            except Exception:
                                dist_between = 0.0
                            is_physically_different = dist_between > 3.0

                            if is_same_team:
                                if is_different_id and is_physically_different:
                                    if potential_possessor is not None and int(potential_possessor["track_id"]) == int(current_possessor["track_id"]):
                                        potential_frames += 1
                                    else:
                                        potential_possessor = current_possessor
                                        potential_frames = 1

                                    if potential_frames >= 6:
                                        # Confirmed pass
                                        from_id = int(last_possessor["track_id"])
                                        to_id = int(current_possessor["track_id"])
                                        pass_dist = float(np.linalg.norm(last_possessor["pos"] - current_possessor["pos"]))
                                        events.append(
                                            {
                                                "t": t_sec,
                                                "source": "calibration",
                                                "type": "pass",
                                                "team_id": int(current_possessor["team_id"]),
                                                "from_player_track_id": from_id,
                                                "player_track_id": to_id,
                                                "distance_m": float(pass_dist),
                                            }
                                        )
                                        try:
                                            stats["events"] += 1
                                        except Exception:
                                            pass
                                        last_possessor = current_possessor
                                        potential_possessor = None
                                        potential_frames = 0
                                else:
                                    # Same player; update position
                                    last_possessor["pos"] = current_possessor["pos"]
                                    potential_possessor = None
                                    potential_frames = 0
                            else:
                                # Turnover / possession change after 5 frames confirmation
                                if potential_possessor is not None and int(potential_possessor["track_id"]) == int(current_possessor["track_id"]):
                                    potential_frames += 1
                                else:
                                    potential_possessor = current_possessor
                                    potential_frames = 1

                                if potential_frames >= 5:
                                    events.append(
                                        {
                                            "t": t_sec,
                                            "source": "calibration",
                                            "type": "possession_change",
                                            "from_team_id": int(last_possessor.get("team_id", -1)),
                                            "team_id": int(current_possessor["team_id"]),
                                            "from_player_track_id": int(last_possessor.get("track_id", -1)),
                                            "player_track_id": int(current_possessor["track_id"]),
                                        }
                                    )
                                    try:
                                        stats["events"] += 1
                                    except Exception:
                                        pass
                                    last_possessor = current_possessor
                                    potential_possessor = None
                                    potential_frames = 0
                    else:
                        frames_since_loss += 1
                        if frames_since_loss > 15:
                            potential_possessor = None
                            potential_frames = 0

                except Exception:
                    # If any calibration-dependent step fails, fall back to empty map frame.
                    pass

            # Always write a frame to keep sync with main video
            vw.write(map_frame)

            # Per-frame metadata (optional; can be huge)
            if f_frames is not None and (frame_idx % frames_stride == 0):
                rec = {
                    "frame_idx": int(frame_idx),
                    "t": float(t_sec),
                    "fps": float(fps),
                    "calibration_ok": bool(calib_res),
                    "rep_err": rep_err,
                    "data": meta_frame,
                }
                f_frames.write(json.dumps(rec, ensure_ascii=False) + "\n")

            frame_idx += 1
    finally:
        try:
            if f_frames is not None:
                f_frames.close()
        except Exception:
            pass

        cap.release()
        vw.release()

        # Backfill team ids in events if they were unknown (-1) at event time.
        try:
            for e in events:
                if not isinstance(e, dict):
                    continue
                et = str(e.get("type") or "")

                if et in ("possession_start", "possession_change"):
                    try:
                        pid = e.get("player_track_id")
                        if (e.get("team_id") is None) or int(e.get("team_id", -1)) == -1:
                            if pid is not None and int(pid) in team_by_track:
                                e["team_id"] = int(team_by_track[int(pid)])
                    except Exception:
                        pass
                    try:
                        fpid = e.get("from_player_track_id")
                        if (e.get("from_team_id") is None) or int(e.get("from_team_id", -1)) == -1:
                            if fpid is not None and int(fpid) in team_by_track:
                                e["from_team_id"] = int(team_by_track[int(fpid)])
                    except Exception:
                        pass

                if et == "pass":
                    try:
                        pid = e.get("player_track_id")
                        if (e.get("team_id") is None) or int(e.get("team_id", -1)) == -1:
                            if pid is not None and int(pid) in team_by_track:
                                e["team_id"] = int(team_by_track[int(pid)])
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            stats["team_by_track_size"] = int(len(team_by_track))
        except Exception:
            pass

        # Final progress tick
        try:
            print(
                "__PROGRESS__ "
                + json.dumps(
                    {
                        "stage": "calibration",
                        "current": int(total_frames if total_frames > 0 else frame_idx),
                        "total": int(total_frames) if total_frames > 0 else 0,
                        "message": "Calibration tamam",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception:
            pass

    out_events_p = Path(out_events)
    out_events_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_events_p, "w", encoding="utf-8") as f_ev:
        json.dump(
            {
                "schema_version": "1.0",
                "created_utc": None,
                "source": {"video_path": str(Path(source).resolve())},
                "artifacts": {
                    "map_video_path": str(out_map_p.resolve()),
                    **({"frames_jsonl_path": str(out_frames_p.resolve())} if write_frames and out_frames_p is not None else {}),
                },
                "stats": stats,
                "events": events,
            },
            f_ev,
            ensure_ascii=False,
            indent=2,
        )

    return CalibrationOutputs(
        map_video_path=str(out_map_p.resolve()),
        events_json_path=str(out_events_p.resolve()),
        frames_jsonl_path=str(out_frames_p.resolve()) if write_frames and out_frames_p is not None else "",
    )


def ImageFromBGR(frame_bgr: np.ndarray):
    from PIL import Image

    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, required=True, help="input video path")
    ap.add_argument("--out_map", type=str, required=True, help="output map video path (.mp4)")
    ap.add_argument("--out_events", type=str, required=True, help="output events json path")
    ap.add_argument("--out_frames", type=str, default="", help="output per-frame jsonl path (optional; can be huge)")
    ap.add_argument("--frames_stride", type=int, default=1, help="write every N frames into jsonl (only if out_frames is set)")
    ap.add_argument("--progress_every", type=int, default=25, help="emit progress every N frames")
    ap.add_argument("--max_frames", type=int, default=0, help="process at most N frames (0 = all)")
    ap.add_argument("--detector", type=str, default=str(THIS_DIR / "best.pt"), help="YOLO weights path")
    ap.add_argument("--kp_weights", type=str, default=str(THIS_DIR / "SV_kp.pth"), help="HRNet keypoints weights")
    ap.add_argument("--line_weights", type=str, default=str(THIS_DIR / "SV_lines.pth"), help="HRNet lines weights")
    ap.add_argument("--conf", type=float, default=0.30, help="detector confidence threshold")
    args = ap.parse_args()

    outs = run(
        source=str(args.source),
        out_map=str(args.out_map),
        out_events=str(args.out_events),
        out_frames=str(args.out_frames) if str(args.out_frames or "").strip() else None,
        detector_weights=str(args.detector),
        kp_weights=str(args.kp_weights),
        line_weights=str(args.line_weights),
        conf_thres=float(args.conf),
        frames_stride=int(args.frames_stride),
        progress_every=int(args.progress_every),
        max_frames=int(args.max_frames),
    )

    # Print a short machine-readable summary for the caller.
    print(
        json.dumps(
            {
                "map_video_path": outs.map_video_path,
                "events_json_path": outs.events_json_path,
                **({"frames_jsonl_path": outs.frames_jsonl_path} if outs.frames_jsonl_path else {}),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
