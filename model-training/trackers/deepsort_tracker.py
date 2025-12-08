# trackers/deepsort_tracker.py
import numpy as np
import cv2
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import logging
from collections import deque
import torch
import torch.nn.functional as F

logger = logging.getLogger("DeepSortTracker")

def cosine_distance(a, b):
    if len(a)==0 or len(b)==0:
        return np.zeros((len(a), len(b))) + 1.0
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-6)
    return 1.0 - np.dot(a_n, b_n.T)

def iou(b1, b2):
    # b1 and b2 are [x1,y1,x2,y2]
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    if x2<x1 or y2<y1:
        return 0.0
    inter = (x2-x1)*(y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)

class Track:
    def __init__(self, bbox, feature, track_id, max_history=30):
        self.bbox = bbox  # [x1,y1,x2,y2]
        self.feature = feature  # embedding vector
        self.track_id = track_id
        self.hits = 1
        self.time_since_update = 0
        self.history = deque(maxlen=max_history)

    def update(self, bbox, feature):
        self.bbox = bbox
        self.feature = feature
        self.hits += 1
        self.time_since_update = 0
        self.history.append(bbox)

    def mark_missed(self):
        self.time_since_update += 1

class DeepSortTracker:
    def __init__(self, max_age=30, n_init=3, reid_path=None, device='cpu'):
        """
        Args:
            max_age: frames to keep dead tracks before deletion
            n_init: not used extensively here (placeholder)
            reid_path: optional path to ReID model weights (torch)
        """
        self.max_age = max_age
        self.n_init = n_init
        self.reid_path = reid_path
        self.device = device
        self.tracks = []
        self._next_id = 1
        self.reid_net = None
        if reid_path:
            try:
                # Attempt to load a torch model (user-provided architecture)
                self.reid_net = torch.load(reid_path, map_location=device)
                self.reid_net.eval()
                logger.info("Loaded ReID model from: %s", reid_path)
            except Exception as e:
                logger.warning(f"Could not load ReID model: {e}. Using fallback features.")

    def _extract_crop_feature(self, frame, bbox):
        x1,y1,x2,y2 = [int(max(0, v)) for v in bbox]
        h, w = frame.shape[:2]
        # clip
        x2 = min(x2, w-1); y2 = min(y2, h-1)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((128,), dtype=np.float32)
        # resize small, convert to RGB
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (64, 128))
        # simple L2-normalized color histogram + downsampled raw pixels
        hist = cv2.calcHist([crop], [0,1,2], None, [8,8,8], [0,256]*3).flatten()
        small = cv2.resize(crop, (8,16)).flatten().astype(np.float32)
        feat = np.concatenate([hist, small])
        feat = feat.astype(np.float32)
        feat /= (np.linalg.norm(feat) + 1e-6)
        return feat

    def _compute_features(self, frame, detections):
        """
        detections: list of [x1,y1,x2,y2,cls,conf]
        returns: numpy array (N, D)
        """
        feats = []
        for d in detections:
            bbox = d[:4]
            if self.reid_net:
                try:
                    # If reid_net is a torch model expecting a tensor
                    x1,y1,x2,y2 = [int(v) for v in bbox]
                    crop = frame[y1:y2, x1:x2]
                    if crop.size==0:
                        feats.append(np.zeros((256,), dtype=np.float32))
                        continue
                    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (128,256))
                    tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()/255.0
                    with torch.no_grad():
                        emb = self.reid_net(tensor.to(self.device))
                        emb = F.normalize(emb, dim=1).cpu().numpy().reshape(-1)
                        feats.append(emb)
                except Exception as e:
                    logger.debug(f"reid feature extraction failed: {e}")
                    feats.append(self._extract_crop_feature(frame, bbox))
            else:
                feats.append(self._extract_crop_feature(frame, bbox))
        return np.array(feats)

    def update(self, detections, frame):
        """
        detections: list of [x1,y1,x2,y2, cls, conf]
        frame: current BGR frame (numpy)
        returns: list of track dicts: {'track_id': id, 'bbox': [x1,y1,x2,y2], 'class': cls, 'conf': conf}
        """
        # Predict step: not using Kalman in this simplified tracker
        # Increase time_since_update for all tracks
        for t in self.tracks:
            t.mark_missed()

        if len(detections) == 0:
            # remove old tracks
            self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
            return [{'track_id': t.track_id, 'bbox': t.bbox, 'class': None, 'conf': None} for t in self.tracks]

        det_bboxes = [d[:4] for d in detections]
        det_feats = self._compute_features(frame, detections)

        # Build cost matrix (cosine distance)
        track_feats = np.array([t.feature for t in self.tracks]) if len(self.tracks)>0 else np.zeros((0, det_feats.shape[1]))
        if track_feats.shape[0] == 0:
            cost = np.zeros((0, det_feats.shape[0]))
        else:
            cost = cosine_distance(track_feats, det_feats)  # track x det

        assigned_tracks = set()
        assigned_dets = set()
        matches = []

        if cost.size != 0:
            # Hungarian on cost
            row_ind, col_ind = linear_sum_assignment(cost)
            for r,c in zip(row_ind, col_ind):
                if cost[r,c] < 0.5:  # threshold for matching
                    matches.append((r,c))
                    assigned_tracks.add(r)
                    assigned_dets.add(c)

        # Update matched
        for r,c in matches:
            t = self.tracks[r]
            t.update(detections[c][:4], det_feats[c])

        # Unmatched detections -> create new tracks
        for idx, d in enumerate(detections):
            if idx not in assigned_dets:
                new_t = Track(d[:4], det_feats[idx], self._next_id)
                self._next_id += 1
                self.tracks.append(new_t)

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]

        # Prepare output
        out = []
        for t in self.tracks:
            out.append({'track_id': t.track_id, 'bbox': [float(x) for x in t.bbox], 'class': None, 'conf': None})
        return out
