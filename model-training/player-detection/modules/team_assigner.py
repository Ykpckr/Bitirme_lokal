"""Team assignment utilities for post-training inference.

This module clusters early-frame color features (HSV histograms) to split
tracked players/goalkeepers into home/away teams and provides smoothed team
labels per track.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple
from collections import Counter, defaultdict, deque
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectionObservation:
    """Lightweight carrier for detector + tracker outputs."""

    frame_index: int
    track_id: Optional[int]
    bbox: Tuple[float, float, float, float]
    class_name: str
    confidence: float


@dataclass
class AssignedTeam:
    """Represents a resolved team assignment."""

    team_key: str
    team_name: str
    cluster_index: int
    confidence: float


class _MiniKMeans:
    """Tiny KMeans variant to avoid heavy dependencies."""

    def __init__(
        self,
        n_clusters: int = 2,
        n_init: int = 5,
        max_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_: Optional[np.ndarray] = None

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        if len(data) < self.n_clusters:
            raise ValueError("Not enough samples to fit KMeans")
        best_inertia = float("inf")
        best_centers = None
        best_labels = None
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_init):
            centers = self._init_centers(data, rng)
            labels = None
            for _ in range(self.max_iter):
                distances = self._pairwise_distances(data, centers)
                labels = np.argmin(distances, axis=1)
                new_centers = []
                for cluster_idx in range(self.n_clusters):
                    members = data[labels == cluster_idx]
                    if len(members) == 0:
                        new_centers.append(centers[cluster_idx])
                    else:
                        new_centers.append(members.mean(axis=0))
                new_centers = np.vstack(new_centers)
                if np.allclose(new_centers, centers):
                    break
                centers = new_centers

            assert labels is not None
            inertia = float(np.sum((data - centers[labels]) ** 2))
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()

        assert best_centers is not None and best_labels is not None
        self.cluster_centers_ = best_centers
        return best_labels

    def predict(self, samples: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("Model not fitted")
        distances = self._pairwise_distances(samples, self.cluster_centers_)
        return np.argmin(distances, axis=1)

    def _init_centers(self, data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        indices = rng.choice(len(data), self.n_clusters, replace=False)
        return data[indices]

    @staticmethod
    def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # ||a - b||^2 = a^2 + b^2 - 2ab
        a_sq = np.sum(a ** 2, axis=1, keepdims=True)
        b_sq = np.sum(b ** 2, axis=1)
        distances = a_sq + b_sq - 2 * np.dot(a, b.T)
        return np.maximum(distances, 0.0)


class TeamAssigner:
    """Assigns tracked detections to home/away teams via k-means clustering."""

    def __init__(
        self,
        teams_cfg: Dict[str, Dict[str, Any]],
        assignment_cfg: Dict[str, Any],
    ) -> None:
        self._teams_cfg = teams_cfg
        self._assignment_cfg = assignment_cfg

        self._upper_body_ratio = float(assignment_cfg.get("upper_body_ratio", 0.6))
        self._hist_bins = tuple(assignment_cfg.get("hist_bins", [24, 6, 3]))
        if len(self._hist_bins) != 3:
            raise ValueError("hist_bins must provide [H, S, V] bin counts")

        self._warmup_min = int(assignment_cfg.get("warmup_min_observations", 200))
        self._warmup_max = int(assignment_cfg.get("warmup_max_observations", 400))
        if self._warmup_min > self._warmup_max:
            raise ValueError("warmup_min_observations must be <= warmup_max_observations")

        self._smoothing_window = int(assignment_cfg.get("smoothing_window", 5))
        self._min_track_votes = int(assignment_cfg.get("min_track_votes", 3))
        self._random_state = int(assignment_cfg.get("random_state", 42))
        self._confidence_decay = float(assignment_cfg.get("confidence_decay", 2.0))

        self._warmup_features: List[np.ndarray] = []
        self._warmup_means: List[np.ndarray] = []
        self._kmeans: Optional[_MiniKMeans] = None
        self._cluster_team_map: Dict[int, str] = {}
        self._cluster_means: Dict[int, np.ndarray] = {}

        self._track_votes: Dict[int, Deque[str]] = defaultdict(lambda: deque(maxlen=self._smoothing_window))
        self._track_counts: Dict[int, int] = defaultdict(int)
        self._track_first_frame: Dict[int, int] = {}
        self._track_last_frame: Dict[int, int] = {}
        self._track_class_name: Dict[int, str] = {}
        self._team_assignment_per_track: Dict[int, AssignedTeam] = {}

        self._home_hint = self._color_hint_to_hsv("home")
        self._away_hint = self._color_hint_to_hsv("away")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        """Returns True once k-means has been fit."""

        return self._kmeans is not None

    def process_detection(
        self,
        frame: np.ndarray,
        observation: DetectionObservation,
    ) -> Optional[AssignedTeam]:
        """Process a single detection and return the assigned team if available."""

        feature = self._extract_feature(frame, observation.bbox)
        if feature is None:
            return None

        hist_vec, mean_hsv = feature
        if not self.ready:
            self._append_warmup_sample(hist_vec, mean_hsv)
            self._maybe_fit()
            return None

        cluster_idx, confidence = self._predict_cluster(hist_vec)
        team_key = self._cluster_team_map.get(cluster_idx)
        if team_key is None:
            return None

        final_team_key = self._register_track_vote(observation, team_key)
        team_name = self._teams_cfg.get(final_team_key, {}).get("name", final_team_key.title())
        assignment = AssignedTeam(
            team_key=final_team_key,
            team_name=team_name,
            cluster_index=cluster_idx,
            confidence=confidence,
        )
        if observation.track_id is not None:
            self._team_assignment_per_track[observation.track_id] = assignment
        return assignment

    def get_track_summaries(self) -> List[Dict[str, Any]]:
        """Return aggregated metadata per track ID."""

        summaries: List[Dict[str, Any]] = []
        for track_id, assignment in self._team_assignment_per_track.items():
            summaries.append(
                {
                    "track_id": track_id,
                    "team_key": assignment.team_key,
                    "team_name": assignment.team_name,
                    "cluster_index": assignment.cluster_index,
                    "confidence": assignment.confidence,
                    "first_frame": self._track_first_frame.get(track_id, -1),
                    "last_frame": self._track_last_frame.get(track_id, -1),
                    "num_observations": self._track_counts.get(track_id, 0),
                    "class_name": self._track_class_name.get(track_id, "unknown"),
                }
            )
        return sorted(summaries, key=lambda item: item["track_id"])

    def get_team_color(self, team_key: str) -> Tuple[int, int, int]:
        """Return a BGR color tuple for drawing purposes."""

        team_cfg = self._teams_cfg.get(team_key, {})
        if "draw_color_bgr" in team_cfg:
            color = team_cfg["draw_color_bgr"]
        elif "color_hint_rgb" in team_cfg:
            rgb = team_cfg["color_hint_rgb"]
            color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        else:
            color = (0, 255, 0) if team_key == "home" else (255, 0, 0)
        return tuple(map(int, color))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _append_warmup_sample(self, hist_vec: np.ndarray, mean_hsv: np.ndarray) -> None:
        if len(self._warmup_features) >= self._warmup_max:
            return
        self._warmup_features.append(hist_vec)
        self._warmup_means.append(mean_hsv)

    def _maybe_fit(self) -> None:
        if self.ready:
            return
        if len(self._warmup_features) < self._warmup_min:
            return
        features = np.stack(self._warmup_features)
        if len(features) > self._warmup_max:
            idx = np.random.default_rng(self._random_state).choice(len(features), self._warmup_max, replace=False)
            features = features[idx]
            means = np.stack(self._warmup_means)[idx]
        else:
            means = np.stack(self._warmup_means)
        self._kmeans = _MiniKMeans(n_clusters=2, random_state=self._random_state)
        labels = self._kmeans.fit_predict(features)
        self._cluster_means = self._compute_cluster_means(labels, means)
        self._cluster_team_map = self._resolve_cluster_team_map()
        logger.info("Team clustering initialized with %d samples", len(features))

    def _predict_cluster(self, hist_vec: np.ndarray) -> Tuple[int, float]:
        if self._kmeans is None:
            raise RuntimeError("KMeans is not fitted yet")
        cluster_idx = int(self._kmeans.predict(hist_vec[None, :])[0])
        centroid = self._kmeans.cluster_centers_[cluster_idx]
        distance = float(np.linalg.norm(hist_vec - centroid))
        confidence = float(np.exp(-self._confidence_decay * distance))
        return cluster_idx, confidence

    def _register_track_vote(self, observation: DetectionObservation, team_key: str) -> str:
        track_id = observation.track_id
        if track_id is None:
            return team_key

        self._track_counts[track_id] += 1
        self._track_first_frame.setdefault(track_id, observation.frame_index)
        self._track_last_frame[track_id] = observation.frame_index
        self._track_class_name[track_id] = observation.class_name

        votes = self._track_votes[track_id]
        votes.append(team_key)
        if len(votes) < max(1, min(self._min_track_votes, votes.maxlen or len(votes))):
            return team_key
        winner = Counter(votes).most_common(1)[0][0]
        return winner

    def _extract_feature(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = int(max(0, min(w - 1, np.floor(x1))))
        y1 = int(max(0, min(h - 1, np.floor(y1))))
        x2 = int(max(0, min(w, np.ceil(x2))))
        y2 = int(max(0, min(h, np.ceil(y2))))
        if x2 <= x1 or y2 <= y1:
            return None

        crop_height = max(2, int((y2 - y1) * self._upper_body_ratio))
        y_upper = min(h, y1 + crop_height)
        roi = frame[y1:y_upper, x1:x2]
        if roi.size == 0:
            return None

        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_bins, s_bins, v_bins = self._hist_bins
        hist = cv2.calcHist(
            [roi_hsv],
            channels=[0, 1, 2],
            mask=None,
            histSize=[h_bins, s_bins, v_bins],
            ranges=[0, 180, 0, 256, 0, 256],
        )
        hist = cv2.normalize(hist, hist).flatten()
        mean_hsv = roi_hsv.reshape(-1, 3).mean(axis=0) / np.array([180.0, 255.0, 255.0])
        return hist.astype(np.float32), mean_hsv.astype(np.float32)

    def _compute_cluster_means(
        self,
        labels: np.ndarray,
        mean_hsv: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        cluster_means: Dict[int, np.ndarray] = {}
        for cluster_idx in range(2):
            cluster_samples = mean_hsv[labels == cluster_idx]
            if len(cluster_samples) == 0:
                cluster_means[cluster_idx] = np.zeros(3, dtype=np.float32)
            else:
                cluster_means[cluster_idx] = cluster_samples.mean(axis=0)
        return cluster_means

    def _resolve_cluster_team_map(self) -> Dict[int, str]:
        if self._home_hint is None and self._away_hint is None:
            return {0: "home", 1: "away"}

        cluster_ids = [0, 1]
        best_map: Dict[int, str] = {}
        best_score = float("inf")

        for home_cluster in cluster_ids:
            away_cluster = 1 - home_cluster
            score = 0.0
            if self._home_hint is not None:
                score += self._color_distance(self._cluster_means[home_cluster], self._home_hint)
            if self._away_hint is not None:
                score += self._color_distance(self._cluster_means[away_cluster], self._away_hint)
            if score < best_score:
                best_score = score
                best_map = {home_cluster: "home", away_cluster: "away"}

        # Fallback if hints missing
        return best_map or {0: "home", 1: "away"}

    def _color_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _color_hint_to_hsv(self, team_key: str) -> Optional[np.ndarray]:
        cfg = self._teams_cfg.get(team_key, {})
        rgb = cfg.get("color_hint_rgb")
        if rgb is None:
            hex_value = cfg.get("color_hint_hex")
            if isinstance(hex_value, str) and hex_value.startswith("#") and len(hex_value) in {4, 7}:
                rgb = self._hex_to_rgb(hex_value)
        if rgb is None:
            return None
        rgb_arr = np.uint8([[list(map(int, rgb))]])
        hsv = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV)[0, 0]
        return hsv.astype(np.float32) / np.array([180.0, 255.0, 255.0], dtype=np.float32)

    @staticmethod
    def _hex_to_rgb(hex_value: str) -> Tuple[int, int, int]:
        hex_value = hex_value.lstrip("#")
        if len(hex_value) == 3:
            hex_value = "".join(ch * 2 for ch in hex_value)
        return tuple(int(hex_value[i : i + 2], 16) for i in (0, 2, 4))