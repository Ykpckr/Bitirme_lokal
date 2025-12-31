"""Team separation utilities used by tracking scripts.

This file provides the exact API expected by `run_tracker_with_teams_with_reid.py`:
- `AutoLabEmbedder.get_features(crop_bgr) -> np.ndarray | None`
- `AutomaticTeamClusterer.collect(feature)`
- `AutomaticTeamClusterer.train() -> bool`
- `AutomaticTeamClusterer.predict(feature) -> int` (0/1, or -1 if unknown)

Approach (matches the existing script's intent):
- Extract a compact color descriptor in CIE-Lab space from the *upper-body* region.
- Learn two clusters (home/away) during a calibration phase.

Dependencies: numpy, opencv-python
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np


class AutoLabEmbedder:
    """Extract a jersey-color descriptor using torso crop + green masking + Lab[a,b] mean.

    This matches the method you shared:
    - Crop torso region to focus on shirt.
    - Remove grass pixels in HSV.
    - Convert to Lab and average valid pixels.
    - Use only [a,b] (ignore L to reduce illumination effects).
    """

    def __init__(
        self,
        torso_y1: float = 0.15,
        torso_y2: float = 0.60,
        torso_x1: float = 0.20,
        torso_x2: float = 0.80,
        hsv_lower_green: tuple[int, int, int] = (30, 40, 40),
        hsv_upper_green: tuple[int, int, int] = (90, 255, 255),
        min_valid_pixels: int = 10,
    ):
        self.torso_y1 = float(torso_y1)
        self.torso_y2 = float(torso_y2)
        self.torso_x1 = float(torso_x1)
        self.torso_x2 = float(torso_x2)
        self.hsv_lower_green = np.array(hsv_lower_green, dtype=np.uint8)
        self.hsv_upper_green = np.array(hsv_upper_green, dtype=np.uint8)
        self.min_valid_pixels = int(min_valid_pixels)

    def get_features(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        h, w = crop_bgr.shape[:2]
        if h < 10 or w < 10:
            return None

        # 1) Torso crop
        y1 = int(h * self.torso_y1)
        y2 = int(h * self.torso_y2)
        x1 = int(w * self.torso_x1)
        x2 = int(w * self.torso_x2)
        if (y2 - y1) < 5 or (x2 - x1) < 5:
            return None

        crop = crop_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # 2) Grass removal in HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, self.hsv_lower_green, self.hsv_upper_green)
        not_green = cv2.bitwise_not(mask_green)

        # 3) Lab conversion
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        valid_pixels = lab[not_green > 0]
        if valid_pixels.shape[0] < self.min_valid_pixels:
            return None

        # 4) Mean Lab, keep only [a,b]
        mean_lab = np.mean(valid_pixels.astype(np.float32), axis=0)
        color_vector = mean_lab[1:3].astype(np.float32)  # (2,)
        return color_vector


def _kmeans_2(features: np.ndarray, seed: int = 0, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Very small KMeans(2) implementation to avoid extra deps.

    Uses squared Euclidean distance, like sklearn KMeans.
    """

    rng = np.random.RandomState(seed)
    n = features.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples for KMeans(2)")

    # Initialize centers with two random samples
    idx = rng.choice(n, size=2, replace=False)
    centers = features[idx].copy()

    for _ in range(iters):
        # Assign
        d0 = np.sum((features - centers[0]) ** 2, axis=1)
        d1 = np.sum((features - centers[1]) ** 2, axis=1)
        labels = (d1 < d0).astype(np.int64)

        # Update
        new_centers = centers.copy()
        for k in (0, 1):
            mask = labels == k
            if np.any(mask):
                new_centers[k] = features[mask].mean(axis=0)

        # Converge
        if np.allclose(new_centers, centers, atol=1e-6):
            centers = new_centers
            break
        centers = new_centers

    return centers, labels


@dataclass
class AutomaticTeamClusterer:
    """Collects jersey-color features and clusters them into 2 teams."""

    min_samples: int = 50
    seed: int = 0
    _buffer: List[np.ndarray] = field(default_factory=list)
    _centers: Optional[np.ndarray] = None

    def collect(self, feat: Optional[np.ndarray]) -> None:
        if feat is None:
            return
        self._buffer.append(feat.astype(np.float32))

    def train(self) -> bool:
        if len(self._buffer) < self.min_samples:
            return False
        feats = np.stack(self._buffer, axis=0)
        centers, _ = _kmeans_2(feats, seed=self.seed)
        self._centers = centers
        self._buffer.clear()
        return True

    def predict(self, feat: Optional[np.ndarray]) -> int:
        if feat is None or self._centers is None:
            return -1
        feat = feat.astype(np.float32).reshape(1, -1)
        d0 = np.sum((feat - self._centers[0].reshape(1, -1)) ** 2, axis=1)[0]
        d1 = np.sum((feat - self._centers[1].reshape(1, -1)) ** 2, axis=1)[0]
        return int(1 if d1 < d0 else 0)
