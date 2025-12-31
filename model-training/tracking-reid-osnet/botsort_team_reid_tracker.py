"""BoT-SORT-style tracker with Team + ReID fusion.

This module is used by `run_botsort_team_reid.py`.

Core ideas:
- Motion: Kalman filter prediction + IoU gating.
- Appearance: ConvNeXt-Large ReID embeddings (sn-reid checkpoint) + cosine distance.
- Team: home/away label (0/1) as a hard gate or weighted penalty.

Dependencies kept minimal: torch, torchvision, numpy, opencv-python
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Geometry / Costs
# -------------------------

def xyxy_to_cxcyah(b: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = b
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    cx = float(x1 + w / 2.0)
    cy = float(y1 + h / 2.0)
    a = w / h
    return np.array([cx, cy, a, h], dtype=np.float32)


def cxcyah_to_xyxy(m: np.ndarray) -> np.ndarray:
    cx, cy, a, h = [float(v) for v in m]
    h = max(1.0, h)
    w = max(1.0, a * h)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)


def iou_matrix(tracks_xyxy: np.ndarray, dets_xyxy: np.ndarray) -> np.ndarray:
    if tracks_xyxy.size == 0 or dets_xyxy.size == 0:
        return np.zeros((tracks_xyxy.shape[0], dets_xyxy.shape[0]), dtype=np.float32)
    out = np.zeros((tracks_xyxy.shape[0], dets_xyxy.shape[0]), dtype=np.float32)
    for i in range(tracks_xyxy.shape[0]):
        for j in range(dets_xyxy.shape[0]):
            out[i, j] = iou_xyxy(tracks_xyxy[i], dets_xyxy[j])
    return out


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Returns cosine similarity in [-1,1]. Inputs are (N,D) and (M,D)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-6)
    return (a_n @ b_n.T).astype(np.float32)


# -------------------------
# Hungarian assignment (no SciPy)
# -------------------------

def _hungarian(cost: np.ndarray) -> List[Tuple[int, int]]:
    """Hungarian algorithm for rectangular matrices.

    Returns list of (row, col) assignments minimizing total cost.
    """

    cost = np.asarray(cost, dtype=np.float64)
    n_rows, n_cols = cost.shape
    n = max(n_rows, n_cols)

    # Pad to square
    padded = np.full((n, n), cost.max() + 1.0, dtype=np.float64)
    padded[:n_rows, :n_cols] = cost

    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = padded[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # p[j] = assigned row for column j
    assignment: List[Tuple[int, int]] = []
    for j in range(1, n + 1):
        i = p[j]
        if 1 <= i <= n_rows and 1 <= j <= n_cols:
            assignment.append((i - 1, j - 1))
    return assignment


# -------------------------
# Kalman Filter (DeepSORT-style)
# -------------------------


class KalmanFilter:
    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel], axis=0)

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        cov = np.diag(np.square(std)).astype(np.float32)
        return mean.astype(np.float32), cov

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])).astype(np.float32)

        mean = self._motion_mat @ mean
        cov = self._motion_mat @ cov @ self._motion_mat.T + motion_cov
        return mean.astype(np.float32), cov.astype(np.float32)

    def project(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std)).astype(np.float32)
        mean = self._update_mat @ mean
        cov = self._update_mat @ cov @ self._update_mat.T
        return mean.astype(np.float32), (cov + innovation_cov).astype(np.float32)

    def update(self, mean: np.ndarray, cov: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, cov)

        chol = np.linalg.cholesky(projected_cov)
        # K = P H^T S^{-1}
        # where S = projected_cov (innovation covariance), P = cov, H = update_mat
        ph_t = cov @ self._update_mat.T  # (8,4)
        kalman_gain = np.linalg.solve(chol.T, np.linalg.solve(chol, ph_t.T)).T  # (8,4)

        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = cov - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean.astype(np.float32), new_cov.astype(np.float32)


# -------------------------
# ReID model wrapper (ConvNeXt-Large)
# -------------------------


class ConvNeXtLargeReID(nn.Module):
    """Model skeleton that matches sn-reid checkpoint structure.

    Checkpoint keys show:
    - backbone.features.* : ConvNeXt features
    - backbone.classifier.0 : LayerNorm(1536)
    - embed (1536->512) + embed_bn + classifier (training-time)

    We return L2-normalized 512-D embeddings.
    """

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        try:
            import torchvision
            from torchvision.models import convnext_large
        except Exception as e:  # pragma: no cover
            raise RuntimeError("torchvision is required for ConvNeXtLargeReID") from e

        self.backbone = convnext_large(weights=None)

        # Replace classifier with just LayerNorm to match checkpoint (no Linear)
        self.backbone.classifier = nn.Sequential(nn.LayerNorm(1536, eps=1e-6))

        self.embed = nn.Linear(1536, embed_dim)
        self.embed_bn = nn.BatchNorm1d(embed_dim)

        # Training classifier exists in checkpoint; kept for compatibility
        self.classifier = nn.Linear(embed_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mirror torchvision ConvNeXt forward, but stop at features -> pooling -> LN
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)

        x = self.embed(x)
        x = self.embed_bn(x)
        x = F.normalize(x, p=2, dim=1)
        return x


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module.") :]] = v
        else:
            out[k] = v
    return out


class ReIDExtractor:
    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        input_hw: Tuple[int, int] = (256, 128),
        batch_size: int = 64,
        use_fp16: bool = True,
    ):
        self.device = torch.device(device)
        self.input_hw = input_hw
        self.batch_size = int(batch_size)
        self.use_fp16 = bool(use_fp16)

        # Build model based on checkpoint shapes (robust to future changes)
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
        if not isinstance(sd, dict):
            raise RuntimeError("Unsupported ReID checkpoint format")
        sd = _strip_module_prefix(sd)

        if "embed.weight" not in sd:
            raise RuntimeError("Checkpoint missing embed.weight; cannot infer embedding head")
        embed_dim = int(sd["embed.weight"].shape[0])

        self.model = ConvNeXtLargeReID(embed_dim=embed_dim)

        # Make classifier output size match checkpoint (not used for inference)
        if "classifier.weight" in sd:
            num_classes = int(sd["classifier.weight"].shape[0])
            self.model.classifier = nn.Linear(embed_dim, num_classes, bias=True)

        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading ReID: {unexpected[:20]}")
        # Missing is OK if checkpoint doesn't include classifier, etc.

        self.model.eval().to(self.device)

        self._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self._mean = self._mean.to(self.device)
        self._std = self._std.to(self.device)

    @torch.inference_mode()
    def extract(self, crops_bgr: Sequence[np.ndarray]) -> np.ndarray:
        if len(crops_bgr) == 0:
            return np.zeros((0, 512), dtype=np.float32)

        # Infer embedding dim from model head
        out_dim = int(self.model.embed.out_features) if hasattr(self.model, "embed") else 512

        batch = []
        for crop in crops_bgr:
            if crop is None or crop.size == 0:
                # placeholder; will be overwritten with zeros
                batch.append(np.zeros((self.input_hw[0], self.input_hw[1], 3), dtype=np.uint8))
                continue
            rgb = crop[:, :, ::-1]
            resized = cv2_resize_rgb(rgb, self.input_hw)
            batch.append(resized)

        # Run in micro-batches to limit VRAM and improve throughput
        embs: List[np.ndarray] = []
        n = len(batch)
        bs = max(1, self.batch_size)

        autocast_enabled = self.use_fp16 and (self.device.type == "cuda")
        for start in range(0, n, bs):
            chunk = batch[start : start + bs]
            arr = np.stack(chunk, axis=0).astype(np.float32) / 255.0
            ten = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(self.device)
            ten = (ten - self._mean) / self._std

            if autocast_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model(ten)
            else:
                out = self.model(ten)

            embs.append(out.detach().float().cpu().numpy().astype(np.float32))

        if len(embs) == 0:
            return np.zeros((0, out_dim), dtype=np.float32)
        return np.concatenate(embs, axis=0)


def cv2_resize_rgb(img_rgb: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    import cv2

    h, w = hw
    return cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)


# -------------------------
# Optional ReID model wrapper (OSNet via torchreid)
# -------------------------


class OSNetReIDExtractor:
    """OSNet feature extractor using torchreid (deep-person-reid).

    This is optional. If torchreid is not installed, constructing this class will fail.
    A pretrained weights file is required.
    """

    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        model_name: str = "osnet_x1_0",
        input_hw: Tuple[int, int] = (256, 128),
        batch_size: int = 128,
        use_fp16: bool = True,
    ):
        if not weights_path:
            raise ValueError("OSNet weights_path is required")

        self.device = torch.device(device)
        self.input_hw = input_hw
        self.batch_size = int(batch_size)
        self.use_fp16 = bool(use_fp16)

        self._is_torchscript = False
        self.model = None

        # First try TorchScript (lets you use OSNet without torchreid).
        # Expectation: model takes BCHW float tensor normalized with ImageNet stats and returns (B, D).
        try:
            self.model = torch.jit.load(str(weights_path), map_location=self.device)
            self.model.eval()
            self._is_torchscript = True
        except Exception:
            self.model = None

        # Otherwise, try deep-person-reid torchreid without importing torchreid.__init__.
        # This avoids importing training/engine utilities that can pull in tensorboard/tensorflow.
        if self.model is None:
            try:
                import importlib.util
                import sys
                import types

                torchreid_spec = importlib.util.find_spec("torchreid")
                if torchreid_spec is None or not torchreid_spec.submodule_search_locations:
                    raise RuntimeError("torchreid not installed")
                torchreid_dir = str(list(torchreid_spec.submodule_search_locations)[0])

                if "torchreid" not in sys.modules:
                    pkg = types.ModuleType("torchreid")
                    pkg.__path__ = [torchreid_dir]
                    sys.modules["torchreid"] = pkg

                def _load(mod_name: str):
                    if mod_name in sys.modules:
                        return sys.modules[mod_name]
                    spec = importlib.util.find_spec(mod_name)
                    if spec is None or spec.loader is None:
                        raise RuntimeError(f"torchreid missing module: {mod_name}")
                    m = importlib.util.module_from_spec(spec)
                    sys.modules[mod_name] = m
                    spec.loader.exec_module(m)
                    return m

                models_mod = _load("torchreid.models")
                torchtools_mod = _load("torchreid.utils.torchtools")
                build_model = getattr(models_mod, "build_model")
                load_pretrained_weights = getattr(torchtools_mod, "load_pretrained_weights")

                self.model = build_model(name=str(model_name), num_classes=1, pretrained=False)
                load_pretrained_weights(self.model, str(weights_path))
                self.model.eval().to(self.device)
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "OSNet stage needs either (1) a TorchScript model at osnet.weights, or (2) deep-person-reid torchreid installed. "
                    "On Windows, installing deep-person-reid typically requires Git in PATH."
                ) from e

        self._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
        self._std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1).to(self.device)

    @torch.inference_mode()
    def extract(self, crops_bgr: Sequence[np.ndarray]) -> np.ndarray:
        if len(crops_bgr) == 0:
            return np.zeros((0, 512), dtype=np.float32)

        batch: List[np.ndarray] = []
        for crop in crops_bgr:
            if crop is None or crop.size == 0:
                batch.append(np.zeros((self.input_hw[0], self.input_hw[1], 3), dtype=np.uint8))
                continue
            rgb = crop[:, :, ::-1]
            resized = cv2_resize_rgb(rgb, self.input_hw)
            batch.append(resized)

        embs: List[np.ndarray] = []
        n = len(batch)
        bs = max(1, self.batch_size)

        autocast_enabled = self.use_fp16 and (self.device.type == "cuda")
        for start in range(0, n, bs):
            chunk = batch[start : start + bs]
            arr = np.stack(chunk, axis=0).astype(np.float32) / 255.0
            ten = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(self.device)
            ten = (ten - self._mean) / self._std

            if autocast_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model(ten)
            else:
                out = self.model(ten)

            out = F.normalize(out, p=2, dim=1)
            embs.append(out.detach().float().cpu().numpy().astype(np.float32))

        return np.concatenate(embs, axis=0) if embs else np.zeros((0, 512), dtype=np.float32)


# -------------------------
# Tracking data structures
# -------------------------


@dataclass
class Detection:
    bbox_xyxy: np.ndarray  # (4,)
    conf: float
    cls_id: int
    team_id: int = -1  # 0/1 for home/away; -1 unknown
    embedding: Optional[np.ndarray] = None  # (D,)
    embedding_osnet: Optional[np.ndarray] = None  # (D,)


@dataclass
class Track:
    track_id: int
    cls_id: int
    team_id: int
    mean: np.ndarray
    cov: np.ndarray
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    conf: float = 0.0
    bbox_xyxy: np.ndarray = None  # last updated bbox
    embedding_ema: Optional[np.ndarray] = None
    embedding_gallery: List[np.ndarray] = field(default_factory=list)

    embedding_osnet_ema: Optional[np.ndarray] = None
    embedding_osnet_gallery: List[np.ndarray] = field(default_factory=list)

    # Re-link event info (set only when a new track is created by reusing an inactive ID)
    relink_source_id: int = -1
    relink_sim: float = 0.0
    relink_inactive_age: int = -1
    relink_reported: bool = False

    # Bookkeeping for time/spatial penalties
    last_seen_frame: int = 0
    last_bin: Tuple[int, int] = (0, 0)

    # Debug: how this track was last updated (for CSV)
    last_assoc_stage: str = ""  # stage1 | osnet | iou | reacquire | new | referee
    last_assoc_iou: float = 0.0
    last_assoc_app_sim: float = 0.0
    last_assoc_osnet_sim: float = 0.0

    def predict(self, kf: KalmanFilter) -> None:
        self.mean, self.cov = kf.predict(self.mean, self.cov)
        self.age += 1
        self.time_since_update += 1

    def update(
        self,
        kf: KalmanFilter,
        det: Detection,
        alpha_embed: float,
        gallery_size: int = 10,
        allow_embedding_update: bool = True,
        allow_osnet_update: bool = True,
    ) -> None:
        m = xyxy_to_cxcyah(det.bbox_xyxy)
        self.mean, self.cov = kf.update(self.mean, self.cov, m)
        self.bbox_xyxy = det.bbox_xyxy.astype(np.float32)
        self.conf = float(det.conf)
        self.hits += 1
        self.time_since_update = 0

        # Update team label if known (keep stable if unknown)
        if det.team_id != -1:
            self.team_id = int(det.team_id)

        if det.embedding is not None and bool(allow_embedding_update):
            e = det.embedding.astype(np.float32)
            e = e / (np.linalg.norm(e) + 1e-6)

            self.embedding_gallery.append(e)
            if gallery_size > 0 and len(self.embedding_gallery) > int(gallery_size):
                self.embedding_gallery = self.embedding_gallery[-int(gallery_size) :]
            if self.embedding_ema is None:
                self.embedding_ema = e
            else:
                self.embedding_ema = alpha_embed * self.embedding_ema + (1.0 - alpha_embed) * e
                self.embedding_ema = self.embedding_ema / (np.linalg.norm(self.embedding_ema) + 1e-6)

        if det.embedding_osnet is not None and bool(allow_osnet_update):
            e2 = det.embedding_osnet.astype(np.float32)
            e2 = e2 / (np.linalg.norm(e2) + 1e-6)

            self.embedding_osnet_gallery.append(e2)
            if gallery_size > 0 and len(self.embedding_osnet_gallery) > int(gallery_size):
                self.embedding_osnet_gallery = self.embedding_osnet_gallery[-int(gallery_size) :]
            if self.embedding_osnet_ema is None:
                self.embedding_osnet_ema = e2
            else:
                self.embedding_osnet_ema = alpha_embed * self.embedding_osnet_ema + (1.0 - alpha_embed) * e2
                self.embedding_osnet_ema = self.embedding_osnet_ema / (np.linalg.norm(self.embedding_osnet_ema) + 1e-6)

    def to_xyxy(self) -> np.ndarray:
        return cxcyah_to_xyxy(self.mean[:4])


# -------------------------
# Main tracker
# -------------------------


class BoTSORTTeamReIDTracker:
    def __init__(
        self,
        w_iou: float = 0.6,
        w_app: float = 0.35,
        w_team: float = 0.05,
        iou_gate: float = 0.05,
        app_gate: float = 0.2,
        team_strict: bool = False,
        max_age: int = 30,
        min_hits: int = 3,
        alpha_embed: float = 0.9,
        second_stage_iou: bool = True,
        iou_gate_second: float = 0.2,
        new_track_min_conf: float = 0.4,
        track_classes: Sequence[int] = (0, 2),  # Player, Referee for your best.pt
        ball_class: int = 1,

        # Re-linking (re-identification after leaving the view)
        relink_enabled: bool = True,
        relink_max_age: int = 1800,
        relink_app_gate: float = 0.30,
        relink_sim_margin: float = 0.05,
        relink_team_strict: bool = True,
        relink_only_class: Optional[int] = None,

        # Keep a small gallery of recent embeddings per track to improve relink robustness
        embed_gallery_size: int = 10,

        # Multi-embedding memory scoring reduce
        reid_memory_reduce: str = "max",  # max|mean|median

        # ReID embedding update policy (prevent drift)
        reid_update_min_det_conf: float = 0.55,
        reid_update_min_box_h: float = 80.0,
        reid_update_min_sim_for_update: float = 0.55,

        # Only apply team penalty/strictness for these classes (default: Player only)
        team_penalize_classes: Sequence[int] = (0,),

        # Spatial constraint for relink candidates (normalized by frame diagonal; 0 disables)
        relink_max_center_dist_norm: float = 0.0,

        # Embedding update safeguard for IoU-only associations (prevents drift)
        embed_update_min_sim_iou_only: float = 0.30,

        # Dedicated inactive-pool reacquire pass
        reacquire_enabled: bool = True,
        reacquire_max_gap_frames: int = 450,
        reacquire_topk_candidates: int = 10,
        reacquire_sim_gate: float = 0.45,
        reacquire_time_penalty: float = 0.002,

        # Spatial grid penalty
        spatial_enabled: bool = True,
        spatial_grid: Tuple[int, int] = (6, 3),
        spatial_bin_penalty: float = 0.15,
        spatial_hard_center_gate_norm: float = 0.60,
        spatial_reduce_on_replay: bool = True,
        spatial_replay_scale: float = 0.5,

        # Optional extra appearance stage using OSNet embeddings
        osnet_stage_enabled: bool = False,
        osnet_app_gate: float = 0.50,
        osnet_iou_gate: float = 0.01,
        osnet_w_iou: float = 0.20,
        osnet_w_app: float = 0.80,
        osnet_w_team: float = 0.05,
        osnet_update_min_sim_for_update: float = 0.55,

        # Optional: allow OSNet to participate in relink (ID reuse from inactive pool)
        relink_use_osnet: bool = True,
        relink_osnet_gate: Optional[float] = None,

        # Referee handling: keep a single stable ID (one referee per match)
        single_referee_id: bool = True,
        referee_class_id: Optional[int] = 2,
        referee_fixed_track_id: int = 9999,
    ):
        self.w_iou = float(w_iou)
        self.w_app = float(w_app)
        self.w_team = float(w_team)
        self.iou_gate = float(iou_gate)
        self.app_gate = float(app_gate)
        self.team_strict = bool(team_strict)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.alpha_embed = float(alpha_embed)

        self.second_stage_iou = bool(second_stage_iou)
        self.iou_gate_second = float(iou_gate_second)
        self.new_track_min_conf = float(new_track_min_conf)

        self.track_classes = tuple(int(c) for c in track_classes)
        self.ball_class = int(ball_class)

        self.single_referee_id = bool(single_referee_id)
        self.referee_class_id = int(referee_class_id) if referee_class_id is not None else None
        self.referee_fixed_track_id = int(referee_fixed_track_id)

        self.relink_enabled = bool(relink_enabled)
        self.relink_max_age = int(relink_max_age)
        self.relink_app_gate = float(relink_app_gate)
        self.relink_sim_margin = float(relink_sim_margin)
        self.relink_team_strict = bool(relink_team_strict)
        self.relink_only_class = int(relink_only_class) if relink_only_class is not None else None
        self.embed_gallery_size = int(embed_gallery_size)

        self.reid_memory_reduce = str(reid_memory_reduce).lower().strip() or "max"
        self.reid_update_min_det_conf = float(reid_update_min_det_conf)
        self.reid_update_min_box_h = float(reid_update_min_box_h)
        self.reid_update_min_sim_for_update = float(reid_update_min_sim_for_update)

        self.team_penalize_classes = tuple(int(c) for c in team_penalize_classes)
        self.relink_max_center_dist_norm = float(relink_max_center_dist_norm)
        self.embed_update_min_sim_iou_only = float(embed_update_min_sim_iou_only)

        self.reacquire_enabled = bool(reacquire_enabled)
        self.reacquire_max_gap_frames = int(reacquire_max_gap_frames)
        self.reacquire_topk_candidates = int(reacquire_topk_candidates)
        self.reacquire_sim_gate = float(reacquire_sim_gate)
        self.reacquire_time_penalty = float(reacquire_time_penalty)

        self.spatial_enabled = bool(spatial_enabled)
        self.spatial_grid = (int(spatial_grid[0]), int(spatial_grid[1]))
        self.spatial_bin_penalty = float(spatial_bin_penalty)
        self.spatial_hard_center_gate_norm = float(spatial_hard_center_gate_norm)
        self.spatial_reduce_on_replay = bool(spatial_reduce_on_replay)
        self.spatial_replay_scale = float(spatial_replay_scale)

        self.osnet_stage_enabled = bool(osnet_stage_enabled)
        self.osnet_app_gate = float(osnet_app_gate)
        self.osnet_iou_gate = float(osnet_iou_gate)
        self.osnet_w_iou = float(osnet_w_iou)
        self.osnet_w_app = float(osnet_w_app)
        self.osnet_w_team = float(osnet_w_team)
        self.osnet_update_min_sim_for_update = float(osnet_update_min_sim_for_update)

        self.relink_use_osnet = bool(relink_use_osnet)
        self.relink_osnet_gate = float(relink_osnet_gate) if relink_osnet_gate is not None else None

        # Frame size (optional). Set via set_frame_size() for spatial relink constraints.
        self._frame_wh: Optional[Tuple[int, int]] = None

        # Frame counter + replay-mode stricter gating window
        self._frame_idx: int = 0
        self._replay_left: int = 0
        self._replay_overrides: Dict[str, float | bool | int] = {}

        self.kf = KalmanFilter()
        self.tracks: List[Track] = []
        self._next_id = 1

        # Inactive gallery: recently lost tracks for global ID re-linking
        self._inactive: List[Track] = []

        # Optional separate ball track (IoU-only, single id)
        self.ball_track: Optional[Track] = None
        self.ball_max_age = 10

        # Optional separate referee track (single stable id)
        self.referee_track: Optional[Track] = None

        # Debug stats (to verify which association stage is doing work)
        # Keys are cumulative counts since last consume.
        self._debug_period: Dict[str, int] = {
            "frames": 0,
            "stage1_matches": 0,
            "osnet_matches": 0,
            "osnet_tracks_with_emb": 0,
            "osnet_dets_with_emb": 0,
            "stage2_matches": 0,
            "reacquire_reactivated": 0,
            "new_tracks": 0,
        }

    def consume_debug_period(self) -> Dict[str, int]:
        """Return and reset debug counters accumulated since last call."""

        out = dict(self._debug_period)
        for k in list(self._debug_period.keys()):
            self._debug_period[k] = 0
        return out

    def set_frame_size(self, width: int, height: int) -> None:
        self._frame_wh = (int(width), int(height))

    def enter_replay_mode(self, frames: int, stricter: Optional[Dict[str, float | bool | int]] = None) -> None:
        self._replay_left = max(0, int(frames))
        self._replay_overrides = dict(stricter or {})

    def _in_replay(self) -> bool:
        return self._replay_left > 0

    def _effective_team_strict(self) -> bool:
        if self._in_replay() and "team_strict" in self._replay_overrides:
            return bool(self._replay_overrides["team_strict"])
        return bool(self.team_strict)

    def _effective_app_gate(self) -> float:
        if self._in_replay() and "sim_gate" in self._replay_overrides:
            return float(max(self.app_gate, float(self._replay_overrides["sim_gate"])))
        return float(self.app_gate)

    def _effective_reacquire_sim_gate(self) -> float:
        if self._in_replay() and "sim_gate" in self._replay_overrides:
            return float(max(self.reacquire_sim_gate, float(self._replay_overrides["sim_gate"])))
        return float(self.reacquire_sim_gate)

    def _effective_reacquire_time_penalty(self) -> float:
        if self._in_replay() and "time_penalty" in self._replay_overrides:
            return float(self._replay_overrides["time_penalty"])
        return float(self.reacquire_time_penalty)

    def _effective_reacquire_max_gap(self) -> int:
        if self._in_replay() and "max_gap_frames" in self._replay_overrides:
            return int(self._replay_overrides["max_gap_frames"])
        return int(self.reacquire_max_gap_frames)

    def _effective_spatial_scale(self) -> float:
        if self._in_replay() and self.spatial_reduce_on_replay:
            return float(self.spatial_replay_scale)
        return 1.0

    def _bin_for_bbox(self, bbox_xyxy: np.ndarray) -> Tuple[int, int]:
        if bbox_xyxy is None or self._frame_wh is None:
            return (0, 0)
        fw, fh = self._frame_wh
        gw, gh = self.spatial_grid
        cx = float((bbox_xyxy[0] + bbox_xyxy[2]) * 0.5)
        cy = float((bbox_xyxy[1] + bbox_xyxy[3]) * 0.5)
        bx = int(np.clip((cx / max(1.0, float(fw))) * gw, 0, gw - 1))
        by = int(np.clip((cy / max(1.0, float(fh))) * gh, 0, gh - 1))
        return (bx, by)

    def _gallery_reduce(self, sims: np.ndarray) -> float:
        if sims.size == 0:
            return 0.0
        m = self.reid_memory_reduce
        if m == "mean":
            return float(np.mean(sims))
        if m == "median":
            return float(np.median(sims))
        return float(np.max(sims))

    def _track_det_app_sim(self, tr: Track, det_emb: Optional[np.ndarray]) -> float:
        if det_emb is None:
            return 0.0
        det_e = det_emb.astype(np.float32)
        det_e = det_e / (np.linalg.norm(det_e) + 1e-6)
        if tr.embedding_gallery:
            g = np.stack(tr.embedding_gallery, axis=0).astype(np.float32)
        elif tr.embedding_ema is not None:
            g = tr.embedding_ema[None, :].astype(np.float32)
        else:
            return 0.0
        sims = (g @ det_e.reshape(-1, 1)).reshape(-1)
        return self._gallery_reduce(sims)

    def _track_det_osnet_sim(self, tr: Track, det_emb: Optional[np.ndarray]) -> float:
        if det_emb is None:
            return 0.0
        det_e = det_emb.astype(np.float32)
        det_e = det_e / (np.linalg.norm(det_e) + 1e-6)
        if tr.embedding_osnet_gallery:
            g = np.stack(tr.embedding_osnet_gallery, axis=0).astype(np.float32)
        elif tr.embedding_osnet_ema is not None:
            g = tr.embedding_osnet_ema[None, :].astype(np.float32)
        else:
            return 0.0
        sims = (g @ det_e.reshape(-1, 1)).reshape(-1)
        return self._gallery_reduce(sims)

    def cut_to_inactive(self) -> None:
        """Soft cut handling: move active tracks into inactive pool (for relink), then clear actives."""

        if not self.relink_enabled:
            self.reset()
            return

        for t in self.tracks:
            if t.embedding_ema is None:
                continue
            t.time_since_update = 0
            t.last_seen_frame = int(self._frame_idx)
            t.last_bin = self._bin_for_bbox(t.bbox_xyxy)
            self._inactive.append(t)

        self.tracks.clear()
        self.ball_track = None
        self.referee_track = None

    def reset(self) -> None:
        self.tracks.clear()
        self._inactive.clear()
        self.ball_track = None
        self.referee_track = None
        self._next_id = 1
        self._frame_idx = 0
        self._replay_left = 0
        self._replay_overrides = {}

    def _new_referee_track(self, det: Detection) -> Track:
        m = xyxy_to_cxcyah(det.bbox_xyxy)
        mean, cov = self.kf.initiate(m)
        tr = Track(
            track_id=int(self.referee_fixed_track_id),
            cls_id=int(det.cls_id),
            team_id=int(det.team_id),
            mean=mean,
            cov=cov,
            hits=max(int(self.min_hits), 1),
            age=1,
            time_since_update=0,
            conf=float(det.conf),
            bbox_xyxy=det.bbox_xyxy.astype(np.float32),
            embedding_ema=None,
            relink_source_id=-1,
            relink_sim=0.0,
            relink_inactive_age=-1,
            relink_reported=False,
            last_seen_frame=int(self._frame_idx),
            last_bin=(0, 0),
            last_assoc_stage="referee",
            last_assoc_iou=0.0,
            last_assoc_app_sim=0.0,
            last_assoc_osnet_sim=0.0,
        )
        tr.last_bin = self._bin_for_bbox(tr.bbox_xyxy)
        return tr

    def _new_track(self, det: Detection) -> Track:
        # Try to reuse a global ID from inactive gallery (player leaving view -> re-enter)
        track_id, relink_source_id, relink_sim, relink_inactive_age = self._allocate_track_id(det)

        m = xyxy_to_cxcyah(det.bbox_xyxy)
        mean, cov = self.kf.initiate(m)
        tr = Track(
            track_id=int(track_id),
            cls_id=int(det.cls_id),
            team_id=int(det.team_id),
            mean=mean,
            cov=cov,
            hits=1,
            age=1,
            time_since_update=0,
            conf=float(det.conf),
            bbox_xyxy=det.bbox_xyxy.astype(np.float32),
            embedding_ema=None,
            relink_source_id=int(relink_source_id),
            relink_sim=float(relink_sim),
            relink_inactive_age=int(relink_inactive_age),
            relink_reported=False,
            last_seen_frame=int(self._frame_idx),
            last_bin=(0, 0),
            last_assoc_stage="new",
            last_assoc_iou=0.0,
            last_assoc_app_sim=0.0,
            last_assoc_osnet_sim=0.0,
        )
        tr.last_bin = self._bin_for_bbox(tr.bbox_xyxy)
        if det.embedding is not None:
            tr.embedding_ema = det.embedding.astype(np.float32)
            tr.embedding_ema = tr.embedding_ema / (np.linalg.norm(tr.embedding_ema) + 1e-6)
            tr.embedding_gallery = [tr.embedding_ema.copy()]
        if det.embedding_osnet is not None:
            tr.embedding_osnet_ema = det.embedding_osnet.astype(np.float32)
            tr.embedding_osnet_ema = tr.embedding_osnet_ema / (np.linalg.norm(tr.embedding_osnet_ema) + 1e-6)
            tr.embedding_osnet_gallery = [tr.embedding_osnet_ema.copy()]
        return tr

    def _allocate_track_id(self, det: Detection) -> Tuple[int, int, float, int]:
        """Either allocate a fresh ID, or re-link to a recently lost track.

        Returns:
            (track_id, relink_source_id, relink_sim, relink_inactive_age)
            - relink_source_id == -1 if not relinked.
        """

        # Default: new id
        new_id = self._next_id
        self._next_id += 1

        relink_source_id = -1
        relink_sim = 0.0
        relink_inactive_age = -1

        if not self.relink_enabled:
            return new_id, relink_source_id, relink_sim, relink_inactive_age
        if self.relink_only_class is not None and int(det.cls_id) != int(self.relink_only_class):
            return new_id, relink_source_id, relink_sim, relink_inactive_age

        # 1) Try ConvNeXt-based relink (existing behavior)
        if det.embedding is not None:
            candidates: List[Tuple[float, int]] = []  # (sim, inactive_idx)
            det_e = det.embedding.astype(np.float32)
            det_e = det_e / (np.linalg.norm(det_e) + 1e-6)

            det_cx = float((det.bbox_xyxy[0] + det.bbox_xyxy[2]) * 0.5)
            det_cy = float((det.bbox_xyxy[1] + det.bbox_xyxy[3]) * 0.5)

            for idx, tr in enumerate(self._inactive):
                if tr.embedding_ema is None:
                    continue
                if self.relink_team_strict and det.team_id != -1 and tr.team_id != -1 and int(det.team_id) != int(tr.team_id):
                    continue
                if int(det.cls_id) != int(tr.cls_id):
                    continue

                if self.relink_max_center_dist_norm > 0 and self._frame_wh is not None and tr.bbox_xyxy is not None:
                    fw, fh = self._frame_wh
                    diag = float(np.hypot(float(fw), float(fh)))
                    if diag > 1e-6:
                        tr_cx = float((tr.bbox_xyxy[0] + tr.bbox_xyxy[2]) * 0.5)
                        tr_cy = float((tr.bbox_xyxy[1] + tr.bbox_xyxy[3]) * 0.5)
                        dist = float(np.hypot(det_cx - tr_cx, det_cy - tr_cy))
                        if (dist / diag) > float(self.relink_max_center_dist_norm):
                            continue

                sims: List[float] = []
                if tr.embedding_gallery:
                    for ge in tr.embedding_gallery:
                        ge2 = ge.astype(np.float32)
                        ge2 = ge2 / (np.linalg.norm(ge2) + 1e-6)
                        sims.append(float(np.dot(ge2, det_e)))
                elif tr.embedding_ema is not None:
                    te = tr.embedding_ema.astype(np.float32)
                    te = te / (np.linalg.norm(te) + 1e-6)
                    sims.append(float(np.dot(te, det_e)))

                if not sims:
                    continue
                sim = self._gallery_reduce(np.asarray(sims, dtype=np.float32))
                candidates.append((sim, idx))

            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                best_sim, best_idx = candidates[0]
                second_sim = candidates[1][0] if len(candidates) > 1 else None
                if best_sim >= self.relink_app_gate and (second_sim is None or (best_sim - second_sim) >= self.relink_sim_margin):
                    reused_tr = self._inactive[best_idx]
                    reused = int(reused_tr.track_id)
                    relink_inactive_age = int(getattr(reused_tr, "time_since_update", 0))
                    self._inactive.pop(best_idx)
                    relink_source_id = reused
                    relink_sim = float(best_sim)
                    return reused, relink_source_id, relink_sim, relink_inactive_age

        # 2) If ConvNeXt didn't relink and this would become a new ID, try OSNet-based relink
        if self.osnet_stage_enabled and self.relink_use_osnet and det.embedding_osnet is not None:
            os_candidates: List[Tuple[float, int]] = []
            for idx, tr in enumerate(self._inactive):
                if (tr.embedding_osnet_ema is None) and (len(tr.embedding_osnet_gallery) == 0):
                    continue
                if self.relink_team_strict and det.team_id != -1 and tr.team_id != -1 and int(det.team_id) != int(tr.team_id):
                    continue
                if int(det.cls_id) != int(tr.cls_id):
                    continue

                sim2 = float(self._track_det_osnet_sim(tr, det.embedding_osnet))
                os_candidates.append((sim2, idx))

            if os_candidates:
                os_candidates.sort(key=lambda x: x[0], reverse=True)
                best_sim2, best_idx2 = os_candidates[0]
                second_sim2 = os_candidates[1][0] if len(os_candidates) > 1 else None
                gate2 = float(self.relink_osnet_gate) if self.relink_osnet_gate is not None else float(self.relink_app_gate)
                if best_sim2 >= gate2 and (second_sim2 is None or (best_sim2 - second_sim2) >= self.relink_sim_margin):
                    reused_tr2 = self._inactive[best_idx2]
                    reused2 = int(reused_tr2.track_id)
                    relink_inactive_age = int(getattr(reused_tr2, "time_since_update", 0))
                    self._inactive.pop(best_idx2)
                    relink_source_id = reused2
                    relink_sim = float(best_sim2)
                    return reused2, relink_source_id, relink_sim, relink_inactive_age

        # Fall back to issuing a fresh ID
        return new_id, relink_source_id, relink_sim, relink_inactive_age

    def _associate(
        self,
        tracks: List[Track],
        dets: List[Detection],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int], np.ndarray]:
        if len(tracks) == 0 or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets))), np.zeros((len(tracks), len(dets)), dtype=np.float32)

        tr_boxes = np.stack([t.to_xyxy() for t in tracks], axis=0)
        det_boxes = np.stack([d.bbox_xyxy for d in dets], axis=0)

        ious = iou_matrix(tr_boxes, det_boxes)  # (T,D)

        # Appearance sims using per-track embedding gallery reduce
        tr_has = np.zeros(len(tracks), dtype=bool)
        det_has = np.zeros(len(dets), dtype=bool)
        det_idx = []
        det_stack = []
        for j, d in enumerate(dets):
            if d.embedding is None:
                continue
            det_has[j] = True
            det_idx.append(j)
            e = d.embedding.astype(np.float32)
            e = e / (np.linalg.norm(e) + 1e-6)
            det_stack.append(e)

        app_sims = np.zeros((len(tracks), len(dets)), dtype=np.float32)
        if det_stack:
            det_mat = np.stack(det_stack, axis=0)  # (Demb,E)
            for i, t in enumerate(tracks):
                if t.embedding_gallery:
                    g = np.stack(t.embedding_gallery, axis=0).astype(np.float32)  # (K,E)
                elif t.embedding_ema is not None:
                    g = t.embedding_ema[None, :].astype(np.float32)
                else:
                    continue
                tr_has[i] = True
                sims_kd = g @ det_mat.T  # (K,Demb)
                if self.reid_memory_reduce == "mean":
                    sims_red = np.mean(sims_kd, axis=0)
                elif self.reid_memory_reduce == "median":
                    sims_red = np.median(sims_kd, axis=0)
                else:
                    sims_red = np.max(sims_kd, axis=0)
                for kk, j in enumerate(det_idx):
                    app_sims[i, j] = float(sims_red[kk])

        # Team penalty (optionally only for specific classes)
        team_pen = np.zeros((len(tracks), len(dets)), dtype=np.float32)
        for i, t in enumerate(tracks):
            for j, d in enumerate(dets):
                if int(t.cls_id) not in self.team_penalize_classes or int(d.cls_id) not in self.team_penalize_classes:
                    team_pen[i, j] = 0.0
                    continue
                if t.team_id == -1 or d.team_id == -1:
                    team_pen[i, j] = 0.0
                else:
                    team_pen[i, j] = 0.0 if int(t.team_id) == int(d.team_id) else 1.0

        # Gates
        valid = np.ones((len(tracks), len(dets)), dtype=bool)

        valid &= ious >= self.iou_gate

        # Only gate by appearance if both have embeddings
        eff_app_gate = float(self._effective_app_gate())
        for i in range(len(tracks)):
            for j in range(len(dets)):
                if tr_has[i] and det_has[j]:
                    valid[i, j] &= app_sims[i, j] >= eff_app_gate

        if self._effective_team_strict():
            for i in range(len(tracks)):
                for j in range(len(dets)):
                    if tracks[i].team_id != -1 and dets[j].team_id != -1:
                        valid[i, j] &= team_pen[i, j] < 0.5

        # Combined cost
        iou_cost = 1.0 - ious
        app_cost = 1.0 - app_sims

        # If embeddings are missing for either side, do NOT add a constant appearance penalty.
        # Otherwise Hungarian will systematically avoid matching those pairs.
        has_app = (tr_has[:, None] & det_has[None, :])
        app_cost = np.where(has_app, app_cost, 0.0)
        cost = self.w_iou * iou_cost + self.w_app * app_cost + self.w_team * team_pen

        big = 1e6
        cost = np.where(valid, cost, big)

        matches = []
        for r, c in _hungarian(cost):
            if cost[r, c] >= big / 2:
                continue
            matches.append((r, c))

        matched_tracks = {m[0] for m in matches}
        matched_dets = {m[1] for m in matches}

        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_dets = [j for j in range(len(dets)) if j not in matched_dets]

        return matches, unmatched_tracks, unmatched_dets, app_sims

    def _associate_iou_only(
        self,
        tracks: List[Track],
        dets: List[Detection],
        iou_gate: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if len(tracks) == 0 or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))

        tr_boxes = np.stack([t.to_xyxy() for t in tracks], axis=0)
        det_boxes = np.stack([d.bbox_xyxy for d in dets], axis=0)
        ious = iou_matrix(tr_boxes, det_boxes)

        valid = ious >= float(iou_gate)
        cost = 1.0 - ious
        big = 1e6
        cost = np.where(valid, cost, big)

        matches: List[Tuple[int, int]] = []
        for r, c in _hungarian(cost):
            if cost[r, c] >= big / 2:
                continue
            matches.append((r, c))

        matched_tracks = {m[0] for m in matches}
        matched_dets = {m[1] for m in matches}
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_dets = [j for j in range(len(dets)) if j not in matched_dets]
        return matches, unmatched_tracks, unmatched_dets

    def _associate_osnet(
        self,
        tracks: List[Track],
        dets: List[Detection],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int], np.ndarray]:
        if len(tracks) == 0 or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets))), np.zeros((len(tracks), len(dets)), dtype=np.float32)

        tr_boxes = np.stack([t.to_xyxy() for t in tracks], axis=0)
        det_boxes = np.stack([d.bbox_xyxy for d in dets], axis=0)
        ious = iou_matrix(tr_boxes, det_boxes)

        tr_has = np.zeros(len(tracks), dtype=bool)
        det_has = np.zeros(len(dets), dtype=bool)

        det_idx: List[int] = []
        det_stack: List[np.ndarray] = []
        for j, d in enumerate(dets):
            if d.embedding_osnet is None:
                continue
            det_has[j] = True
            det_idx.append(j)
            e = d.embedding_osnet.astype(np.float32)
            e = e / (np.linalg.norm(e) + 1e-6)
            det_stack.append(e)

        os_sims = np.zeros((len(tracks), len(dets)), dtype=np.float32)
        if det_stack:
            det_mat = np.stack(det_stack, axis=0).astype(np.float32)
            for i, t in enumerate(tracks):
                if t.embedding_osnet_gallery:
                    g = np.stack(t.embedding_osnet_gallery, axis=0).astype(np.float32)
                elif t.embedding_osnet_ema is not None:
                    g = t.embedding_osnet_ema[None, :].astype(np.float32)
                else:
                    continue
                tr_has[i] = True
                sims_kd = g @ det_mat.T
                if self.reid_memory_reduce == "mean":
                    sims_red = np.mean(sims_kd, axis=0)
                elif self.reid_memory_reduce == "median":
                    sims_red = np.median(sims_kd, axis=0)
                else:
                    sims_red = np.max(sims_kd, axis=0)
                for kk, j in enumerate(det_idx):
                    os_sims[i, j] = float(sims_red[kk])

        team_pen = np.zeros((len(tracks), len(dets)), dtype=np.float32)
        for i, t in enumerate(tracks):
            for j, d in enumerate(dets):
                if int(t.cls_id) not in self.team_penalize_classes or int(d.cls_id) not in self.team_penalize_classes:
                    team_pen[i, j] = 0.0
                    continue
                if t.team_id == -1 or d.team_id == -1:
                    team_pen[i, j] = 0.0
                else:
                    team_pen[i, j] = 0.0 if int(t.team_id) == int(d.team_id) else 1.0

        valid = np.ones((len(tracks), len(dets)), dtype=bool)
        valid &= ious >= float(self.osnet_iou_gate)

        for i in range(len(tracks)):
            for j in range(len(dets)):
                if tr_has[i] and det_has[j]:
                    valid[i, j] &= os_sims[i, j] >= float(self.osnet_app_gate)

        if self._effective_team_strict():
            for i in range(len(tracks)):
                for j in range(len(dets)):
                    if tracks[i].team_id != -1 and dets[j].team_id != -1:
                        valid[i, j] &= team_pen[i, j] < 0.5

        iou_cost = 1.0 - ious
        os_cost = 1.0 - os_sims
        has_app = (tr_has[:, None] & det_has[None, :])
        os_cost = np.where(has_app, os_cost, 0.0)
        cost = (self.osnet_w_iou * iou_cost) + (self.osnet_w_app * os_cost) + (self.osnet_w_team * team_pen)

        big = 1e6
        cost = np.where(valid, cost, big)

        matches: List[Tuple[int, int]] = []
        for r, c in _hungarian(cost):
            if cost[r, c] >= big / 2:
                continue
            matches.append((r, c))

        matched_tracks = {m[0] for m in matches}
        matched_dets = {m[1] for m in matches}
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_dets = [j for j in range(len(dets)) if j not in matched_dets]
        return matches, unmatched_tracks, unmatched_dets, os_sims

    def _associate_inactive(self, dets: List[Detection]) -> List[Tuple[int, int, float, int]]:
        """Dedicated inactive-pool reacquire pass.

        Returns a list of (inactive_index, det_index, appearance_sim, gap_frames).
        """

        if not self.reacquire_enabled:
            return []
        if not self._inactive or not dets:
            return []

        det_idx = [j for j, d in enumerate(dets) if d.embedding is not None]
        if not det_idx:
            return []

        max_gap = int(self._effective_reacquire_max_gap())
        sim_gate = float(self._effective_reacquire_sim_gate())
        time_pen = float(self._effective_reacquire_time_penalty())
        spatial_scale = float(self._effective_spatial_scale())

        in_idx: List[int] = []
        for i, t in enumerate(self._inactive):
            gap = int(self._frame_idx - int(getattr(t, "last_seen_frame", 0)))
            if gap < 0 or gap > max_gap:
                continue
            if not (t.embedding_gallery or t.embedding_ema is not None):
                continue
            in_idx.append(i)

        if not in_idx:
            return []

        I = len(in_idx)
        D = len(det_idx)
        big = 1e6
        cost = np.full((I, D), big, dtype=np.float32)
        sim_mat = np.zeros((I, D), dtype=np.float32)

        # Pre-normalize detection embeddings
        det_stack = []
        for j in det_idx:
            e = dets[j].embedding.astype(np.float32)
            e = e / (np.linalg.norm(e) + 1e-6)
            det_stack.append(e)
        det_mat = np.stack(det_stack, axis=0)  # (D,E)

        hard_gate = float(max(self.relink_max_center_dist_norm, self.spatial_hard_center_gate_norm))

        for ii, tr_i in enumerate(in_idx):
            tr = self._inactive[tr_i]
            if tr.embedding_gallery:
                g = np.stack(tr.embedding_gallery, axis=0).astype(np.float32)
            elif tr.embedding_ema is not None:
                g = tr.embedding_ema[None, :].astype(np.float32)
            else:
                continue

            sims_kd = g @ det_mat.T
            if self.reid_memory_reduce == "mean":
                sims = np.mean(sims_kd, axis=0)
            elif self.reid_memory_reduce == "median":
                sims = np.median(sims_kd, axis=0)
            else:
                sims = np.max(sims_kd, axis=0)

            gap = int(self._frame_idx - int(getattr(tr, "last_seen_frame", 0)))
            for jj, dj in enumerate(det_idx):
                d = dets[dj]
                if int(d.cls_id) != int(tr.cls_id):
                    continue

                # Team gating/penalty only for configured classes
                team_pen = 0.0
                if int(tr.cls_id) in self.team_penalize_classes and int(d.cls_id) in self.team_penalize_classes:
                    if self._effective_team_strict() and tr.team_id != -1 and d.team_id != -1 and int(tr.team_id) != int(d.team_id):
                        continue
                    if tr.team_id != -1 and d.team_id != -1 and int(tr.team_id) != int(d.team_id):
                        team_pen = 1.0

                sim = float(sims[jj])
                if sim < sim_gate:
                    continue

                # Optional hard center-distance gate
                if hard_gate > 0 and self._frame_wh is not None and tr.bbox_xyxy is not None:
                    fw, fh = self._frame_wh
                    diag = float(np.hypot(float(fw), float(fh)))
                    if diag > 1e-6:
                        dcx = float((d.bbox_xyxy[0] + d.bbox_xyxy[2]) * 0.5)
                        dcy = float((d.bbox_xyxy[1] + d.bbox_xyxy[3]) * 0.5)
                        tcx = float((tr.bbox_xyxy[0] + tr.bbox_xyxy[2]) * 0.5)
                        tcy = float((tr.bbox_xyxy[1] + tr.bbox_xyxy[3]) * 0.5)
                        if (float(np.hypot(dcx - tcx, dcy - tcy)) / diag) > hard_gate:
                            continue

                spatial_pen = 0.0
                if self.spatial_enabled:
                    bx, by = self._bin_for_bbox(d.bbox_xyxy)
                    tbx, tby = getattr(tr, "last_bin", (bx, by))
                    spatial_pen = spatial_scale * self.spatial_bin_penalty * float(abs(int(bx) - int(tbx)) + abs(int(by) - int(tby)))

                sim_mat[ii, jj] = sim
                cost[ii, jj] = float((1.0 - sim) + (self.w_team * team_pen) + (time_pen * float(gap)) + spatial_pen)

        # Top-K restriction per detection
        topk = max(1, int(self.reacquire_topk_candidates))
        for jj in range(D):
            sims_col = sim_mat[:, jj]
            if np.all(sims_col <= 0):
                continue
            keep = np.argsort(-sims_col)[:topk]
            mask = np.ones(I, dtype=bool)
            mask[keep] = False
            cost[mask, jj] = big

        # Solve assignment (SciPy if available)
        try:
            from scipy.optimize import linear_sum_assignment  # type: ignore

            rr, cc = linear_sum_assignment(cost)
            pairs = list(zip(rr.tolist(), cc.tolist()))
        except Exception:
            pairs = _hungarian(cost)

        out: List[Tuple[int, int, float, int]] = []
        for r, c in pairs:
            if cost[r, c] >= big / 2:
                continue
            tr_i = in_idx[r]
            dj = det_idx[c]
            tr = self._inactive[tr_i]
            gap = int(self._frame_idx - int(getattr(tr, "last_seen_frame", 0)))
            sim = float(sim_mat[r, c])
            if gap > max_gap or sim < sim_gate:
                continue
            out.append((tr_i, dj, sim, gap))
        return out

    def update(self, detections: Sequence[Detection]) -> List[Track]:
        # Frame counter + replay window
        self._frame_idx += 1
        if self._replay_left > 0:
            self._replay_left -= 1

        # Debug counters (per-period)
        self._debug_period["frames"] += 1

        # Age inactive gallery and drop stale entries
        if self._inactive:
            for tr in self._inactive:
                tr.time_since_update += 1
            self._inactive = [tr for tr in self._inactive if tr.time_since_update <= self.relink_max_age]

        # Predict existing tracks
        for t in self.tracks:
            t.predict(self.kf)
        if self.referee_track is not None:
            self.referee_track.predict(self.kf)

        # Split ball detections and trackable classes
        dets_track_all = [d for d in detections if int(d.cls_id) in self.track_classes]
        dets_ref: List[Detection] = []
        dets_track: List[Detection] = dets_track_all
        if self.single_referee_id and self.referee_class_id is not None:
            dets_ref = [d for d in dets_track_all if int(d.cls_id) == int(self.referee_class_id)]
            dets_track = [d for d in dets_track_all if int(d.cls_id) != int(self.referee_class_id)]
        dets_ball = [d for d in detections if int(d.cls_id) == self.ball_class]

        # Update ball (IoU-only, single track)
        self._update_ball(dets_ball)

        # Update referee with a single fixed ID (separate from player association)
        referee_updated = False
        if self.single_referee_id and self.referee_class_id is not None:
            if dets_ref:
                best_ref = max(dets_ref, key=lambda d: float(d.conf))
                if self.referee_track is None or int(self.referee_track.track_id) != int(self.referee_fixed_track_id):
                    self.referee_track = self._new_referee_track(best_ref)
                else:
                    # Debug: mark referee update
                    self.referee_track.last_assoc_stage = "referee"
                    self.referee_track.last_assoc_iou = float(iou_xyxy(self.referee_track.to_xyxy(), best_ref.bbox_xyxy))
                    self.referee_track.last_assoc_app_sim = 0.0
                    self.referee_track.last_assoc_osnet_sim = 0.0
                    self.referee_track.update(
                        self.kf,
                        best_ref,
                        alpha_embed=0.0,
                        gallery_size=0,
                        allow_embedding_update=False,
                    )
                    self.referee_track.hits = max(int(self.min_hits), int(self.referee_track.hits) + 1)
                self.referee_track.last_seen_frame = int(self._frame_idx)
                self.referee_track.last_bin = self._bin_for_bbox(self.referee_track.bbox_xyxy)
                referee_updated = True
            else:
                if self.referee_track is not None:
                    self.referee_track.time_since_update += 1

        # Associate players/referees (two-stage like BoT-SORT)
        updated_track_idx: set[int] = set()
        remaining_det_idx: List[int] = list(range(len(dets_track)))

        confirmed_idx = [i for i, t in enumerate(self.tracks) if t.hits >= self.min_hits]
        if confirmed_idx and dets_track:
            confirmed_tracks = [self.tracks[i] for i in confirmed_idx]
            matches1, un_tr1, un_det1, app_sims1 = self._associate(confirmed_tracks, dets_track)
            self._debug_period["stage1_matches"] += int(len(matches1))
            for ti_local, di in matches1:
                ti = confirmed_idx[ti_local]
                det = dets_track[di]
                app_sim = float(app_sims1[ti_local, di])
                assoc_iou = float(iou_xyxy(self.tracks[ti].to_xyxy(), det.bbox_xyxy))
                bbox_h = float(det.bbox_xyxy[3] - det.bbox_xyxy[1])
                allow_embed_update = (
                    det.embedding is not None
                    and float(det.conf) >= self.reid_update_min_det_conf
                    and bbox_h >= self.reid_update_min_box_h
                    and app_sim >= self.reid_update_min_sim_for_update
                )

                self.tracks[ti].last_assoc_stage = "stage1"
                self.tracks[ti].last_assoc_iou = assoc_iou
                self.tracks[ti].last_assoc_app_sim = app_sim
                self.tracks[ti].last_assoc_osnet_sim = 0.0
                self.tracks[ti].update(
                    self.kf,
                    det,
                    alpha_embed=self.alpha_embed,
                    gallery_size=self.embed_gallery_size,
                    allow_embedding_update=allow_embed_update,
                    allow_osnet_update=allow_embed_update,
                )
                self.tracks[ti].last_seen_frame = int(self._frame_idx)
                self.tracks[ti].last_bin = self._bin_for_bbox(self.tracks[ti].bbox_xyxy)
                updated_track_idx.add(ti)
            remaining_det_idx = [d for d in remaining_det_idx if d in un_det1]
        else:
            matches1, un_tr1, app_sims1 = [], list(range(len(confirmed_idx))), np.zeros((len(confirmed_idx), len(dets_track)), dtype=np.float32)

        # Optional extra stage: OSNet appearance matching
        if self.osnet_stage_enabled and remaining_det_idx:
            os_track_idx = [i for i in range(len(self.tracks)) if i not in updated_track_idx]
            if os_track_idx:
                os_tracks = [self.tracks[i] for i in os_track_idx]
                os_dets = [dets_track[j] for j in remaining_det_idx]

                # Debug: how many candidates actually have OSNet embeddings
                self._debug_period["osnet_tracks_with_emb"] += int(
                    sum(1 for t in os_tracks if (t.embedding_osnet_ema is not None or len(t.embedding_osnet_gallery) > 0))
                )
                self._debug_period["osnet_dets_with_emb"] += int(sum(1 for d in os_dets if d.embedding_osnet is not None))

                matches_os, _un_tr_os, un_det_os, os_sims = self._associate_osnet(os_tracks, os_dets)
                self._debug_period["osnet_matches"] += int(len(matches_os))
                for ti_local, di_local in matches_os:
                    ti = os_track_idx[ti_local]
                    di = remaining_det_idx[di_local]
                    det = dets_track[di]

                    os_sim = float(os_sims[ti_local, di_local])
                    assoc_iou = float(iou_xyxy(self.tracks[ti].to_xyxy(), det.bbox_xyxy))
                    bbox_h = float(det.bbox_xyxy[3] - det.bbox_xyxy[1])

                    # Keep ConvNeXt update policy unchanged
                    app_sim = float(self._track_det_app_sim(self.tracks[ti], det.embedding))
                    allow_embed_update = (
                        det.embedding is not None
                        and float(det.conf) >= self.reid_update_min_det_conf
                        and bbox_h >= self.reid_update_min_box_h
                        and app_sim >= self.reid_update_min_sim_for_update
                    )
                    allow_osnet_update = (
                        det.embedding_osnet is not None
                        and float(det.conf) >= self.reid_update_min_det_conf
                        and bbox_h >= self.reid_update_min_box_h
                        and os_sim >= float(self.osnet_update_min_sim_for_update)
                    )

                    self.tracks[ti].last_assoc_stage = "osnet"
                    self.tracks[ti].last_assoc_iou = assoc_iou
                    self.tracks[ti].last_assoc_app_sim = float(app_sim)
                    self.tracks[ti].last_assoc_osnet_sim = float(os_sim)

                    self.tracks[ti].update(
                        self.kf,
                        det,
                        alpha_embed=self.alpha_embed,
                        gallery_size=self.embed_gallery_size,
                        allow_embedding_update=allow_embed_update,
                        allow_osnet_update=allow_osnet_update,
                    )
                    self.tracks[ti].last_seen_frame = int(self._frame_idx)
                    self.tracks[ti].last_bin = self._bin_for_bbox(self.tracks[ti].bbox_xyxy)
                    updated_track_idx.add(ti)

                remaining_det_idx = [remaining_det_idx[j] for j in un_det_os]

        # Second stage: IoU-only to recover short occlusions / missed embeddings
        if self.second_stage_iou and remaining_det_idx:
            unconfirmed_idx = [i for i, t in enumerate(self.tracks) if t.hits < self.min_hits]
            unmatched_confirmed_global = [confirmed_idx[i_local] for i_local in un_tr1] if confirmed_idx else []
            stage2_track_idx = [i for i in (unmatched_confirmed_global + unconfirmed_idx) if i not in updated_track_idx]
            if stage2_track_idx:
                stage2_tracks = [self.tracks[i] for i in stage2_track_idx]
                stage2_dets = [dets_track[j] for j in remaining_det_idx]
                matches2, _un_tr2, un_det2 = self._associate_iou_only(stage2_tracks, stage2_dets, self.iou_gate_second)
                self._debug_period["stage2_matches"] += int(len(matches2))
                for ti_local, di_local in matches2:
                    ti = stage2_track_idx[ti_local]
                    di = remaining_det_idx[di_local]
                    det = dets_track[di]
                    app_sim = float(self._track_det_app_sim(self.tracks[ti], det.embedding))
                    assoc_iou = float(iou_xyxy(self.tracks[ti].to_xyxy(), det.bbox_xyxy))
                    bbox_h = float(det.bbox_xyxy[3] - det.bbox_xyxy[1])
                    allow_embed_update = (
                        det.embedding is not None
                        and float(det.conf) >= self.reid_update_min_det_conf
                        and bbox_h >= self.reid_update_min_box_h
                        and app_sim >= self.reid_update_min_sim_for_update
                    )

                    self.tracks[ti].last_assoc_stage = "iou"
                    self.tracks[ti].last_assoc_iou = assoc_iou
                    self.tracks[ti].last_assoc_app_sim = float(app_sim)
                    self.tracks[ti].last_assoc_osnet_sim = 0.0
                    self.tracks[ti].update(
                        self.kf,
                        det,
                        alpha_embed=self.alpha_embed,
                        gallery_size=self.embed_gallery_size,
                        allow_embedding_update=allow_embed_update,
                        allow_osnet_update=False,
                    )
                    self.tracks[ti].last_seen_frame = int(self._frame_idx)
                    self.tracks[ti].last_bin = self._bin_for_bbox(self.tracks[ti].bbox_xyxy)
                    updated_track_idx.add(ti)
                remaining_det_idx = [remaining_det_idx[j] for j in un_det2]

        # Dedicated inactive reacquire pass (true re-acquisition)
        if remaining_det_idx:
            reacq_dets = [dets_track[j] for j in remaining_det_idx]
            reacq_matches = self._associate_inactive(reacq_dets)
            if reacq_matches:
                used_local: set[int] = set()
                reactivated_track_ids: set[int] = set()
                reacq_matches.sort(key=lambda x: float(x[2]), reverse=True)
                for inactive_i, det_local, sim, gap in reacq_matches:
                    if int(det_local) in used_local:
                        continue

                    tr = self._inactive[inactive_i]
                    det = reacq_dets[int(det_local)]

                    # Reactivate: keep ID + embedding memory, reset KF with current bbox
                    m = xyxy_to_cxcyah(det.bbox_xyxy)
                    tr.mean, tr.cov = self.kf.initiate(m)
                    tr.bbox_xyxy = det.bbox_xyxy.astype(np.float32)
                    tr.conf = float(det.conf)
                    tr.time_since_update = 0
                    tr.age = 1
                    tr.hits = max(int(self.min_hits), int(tr.hits))
                    if det.team_id != -1:
                        tr.team_id = int(det.team_id)
                    tr.last_seen_frame = int(self._frame_idx)
                    tr.last_bin = self._bin_for_bbox(tr.bbox_xyxy)

                    # Debug: reacquire association info
                    tr.last_assoc_stage = "reacquire"
                    tr.last_assoc_iou = 0.0
                    tr.last_assoc_app_sim = float(sim)
                    tr.last_assoc_osnet_sim = float(self._track_det_osnet_sim(tr, det.embedding_osnet))

                    # Log a relink-like event (per reacquire)
                    tr.relink_source_id = int(tr.track_id)
                    tr.relink_sim = float(sim)
                    tr.relink_inactive_age = int(gap)
                    tr.relink_reported = False

                    # Apply embedding update policy using the same similarity used for reacquire
                    bbox_h = float(det.bbox_xyxy[3] - det.bbox_xyxy[1])
                    allow_embed_update = (
                        det.embedding is not None
                        and float(det.conf) >= self.reid_update_min_det_conf
                        and bbox_h >= self.reid_update_min_box_h
                        and float(sim) >= self.reid_update_min_sim_for_update
                    )
                    if allow_embed_update and det.embedding is not None:
                        e = det.embedding.astype(np.float32)
                        e = e / (np.linalg.norm(e) + 1e-6)
                        tr.embedding_gallery.append(e)
                        if self.embed_gallery_size > 0 and len(tr.embedding_gallery) > int(self.embed_gallery_size):
                            tr.embedding_gallery = tr.embedding_gallery[-int(self.embed_gallery_size) :]
                        if tr.embedding_ema is None:
                            tr.embedding_ema = e
                        else:
                            tr.embedding_ema = self.alpha_embed * tr.embedding_ema + (1.0 - self.alpha_embed) * e
                            tr.embedding_ema = tr.embedding_ema / (np.linalg.norm(tr.embedding_ema) + 1e-6)

                    self.tracks.append(tr)
                    used_local.add(int(det_local))
                    reactivated_track_ids.add(int(tr.track_id))

                self._debug_period["reacquire_reactivated"] += int(len(reactivated_track_ids))

                # Remove reactivated tracks from inactive pool and dets from remaining list
                self._inactive = [t for t in self._inactive if int(t.track_id) not in reactivated_track_ids]
                remaining_det_idx = [d for k, d in enumerate(remaining_det_idx) if k not in used_local]

        # Age unmatched tracks
        survivors: List[Track] = []
        for idx, t in enumerate(self.tracks):
            if idx not in updated_track_idx:
                # keep but possibly remove if too old
                if t.time_since_update <= self.max_age:
                    survivors.append(t)
                else:
                    # Move to inactive gallery for possible re-linking
                    if self.relink_enabled and t.embedding_ema is not None:
                        # Re-purpose time_since_update as 'inactive age' while in gallery
                        t.time_since_update = 0
                        if int(getattr(t, "last_seen_frame", 0)) <= 0:
                            t.last_seen_frame = int(self._frame_idx)
                        if getattr(t, "last_bin", None) is None:
                            t.last_bin = self._bin_for_bbox(t.bbox_xyxy)
                        self._inactive.append(t)
            else:
                survivors.append(t)
        self.tracks = survivors

        # Create new tracks for unmatched detections
        new_tracks = 0
        for di in remaining_det_idx:
            d = dets_track[di]
            if float(d.conf) < self.new_track_min_conf:
                continue
            self.tracks.append(self._new_track(d))
            new_tracks += 1

        self._debug_period["new_tracks"] += int(new_tracks)

        # Return confirmed tracks
        confirmed = [t for t in self.tracks if t.hits >= self.min_hits and t.time_since_update == 0]
        if referee_updated and self.referee_track is not None:
            confirmed.append(self.referee_track)
        return confirmed

    def _update_ball(self, dets_ball: List[Detection]) -> None:
        # Predict
        if self.ball_track is not None:
            self.ball_track.predict(self.kf)

        if len(dets_ball) == 0:
            if self.ball_track is not None and self.ball_track.time_since_update > self.ball_max_age:
                self.ball_track = None
            return

        # Pick best ball detection by confidence
        best = max(dets_ball, key=lambda d: float(d.conf))

        if self.ball_track is None:
            self.ball_track = self._new_track(best)
            self.ball_track.cls_id = self.ball_class
            self.ball_track.team_id = -1
            self.ball_track.embedding_ema = None
            return

        # IoU check between predicted and det
        pred = self.ball_track.to_xyxy()
        if iou_xyxy(pred, best.bbox_xyxy) < 0.01:
            # Reset ball track if jump too big
            self.ball_track = self._new_track(best)
            self.ball_track.cls_id = self.ball_class
            self.ball_track.team_id = -1
            self.ball_track.embedding_ema = None
            return

        self.ball_track.update(self.kf, best, alpha_embed=0.0)

    def get_ball_track(self) -> Optional[Track]:
        if self.ball_track is None:
            return None
        if self.ball_track.time_since_update == 0:
            return self.ball_track
        return None
