# tracking/strongsort.py
import numpy as np
import cv2
from .tracker_state import TrackState
from .appearance import AppearanceEncoder
from .matching import compute_cost_matrix, linear_assignment

class StrongSORT:
    def __init__(self, device='cpu', max_age=30, min_hits=2, iou_weight=0.6, app_weight=0.4, cost_threshold=0.7):
        self.device = device
        self.encoder = AppearanceEncoder(device=device, output_dim=256)
        self.tracks = []  # list of TrackState
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_weight = iou_weight
        self.app_weight = app_weight
        self.cost_threshold = cost_threshold

    def update(self, frame, detections):
        """
        detections: list of dicts: {"class","score","bbox"}
        frame: BGR numpy frame
        Returns: list of dicts: {"track_id","class","score","bbox","center"}
        """
        # 1) Extract bboxes and classes
        det_bboxes = [d['bbox'] for d in detections]
        det_classes = [d.get('class','') for d in detections]
        det_scores = [d.get('score',0.0) for d in detections]

        # 2) Prepare crops for appearance
        crops = []
        for bb in det_bboxes:
            x1,y1,x2,y2 = map(int, bb)
            # clamp
            h,w = frame.shape[:2]
            x1 = max(0, min(w-1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h-1, y1))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                crops.append(np.zeros((10,10,3), dtype=np.uint8))
            else:
                crops.append(frame[y1:y2, x1:x2])

        # 3) Compute embeddings
        if crops:
            embeds = self.encoder.encode(crops)
        else:
            embeds = np.zeros((0,256))

        # 4) Predict existing tracks
        for tr in self.tracks:
            tr.predict()

        # 5) Split detections by class: keep combined (players+ball) but matching should prefer same class
        # For simplicity, we'll only match same-class items: build lists per class
        new_tracks = []
        unmatched_dets_global = set(range(len(det_bboxes)))
        updated_tracks = set()
        for cls in set(det_classes + [tr.cls for tr in self.tracks]):
            # indices of detections of this class
            det_idxs = [i for i,c in enumerate(det_classes) if c == cls]
            # tracks of this class
            track_idxs = [i for i,tr in enumerate(self.tracks) if tr.cls == cls]

            if len(track_idxs) == 0:
                # create new tracks for all dets of this class
                for i in det_idxs:
                    tr = TrackState(det_bboxes[i], cls, embed=embeds[i] if i < len(embeds) else None)
                    new_tracks.append(tr)
                    if i in unmatched_dets_global:
                        unmatched_dets_global.remove(i)
                continue

            if len(det_idxs) == 0:
                # no detections for these tracks -> they will age
                continue

            # prepare lists
            tracks_list = [self.tracks[idx] for idx in track_idxs]
            dets_list = [det_bboxes[i] for i in det_idxs]
            det_embs = [embeds[i] if i < len(embeds) else None for i in det_idxs]

            cost = compute_cost_matrix(tracks_list, dets_list, det_embs, iou_weight=self.iou_weight, appearance_weight=self.app_weight)
            matches, unmatched_dets, unmatched_tracks = linear_assignment(cost, thresh=self.cost_threshold)

            # apply matches
            for r,c in matches:
                tr_idx = track_idxs[r]
                det_idx = det_idxs[c]
                self.tracks[tr_idx].update(det_bboxes[det_idx], embed=det_embs[c])
                updated_tracks.add(tr_idx)
                if det_idx in unmatched_dets_global:
                    unmatched_dets_global.remove(det_idx)

            # unmatched detections -> new tracks
            for c in unmatched_dets:
                det_idx = det_idxs[c]
                tr = TrackState(det_bboxes[det_idx], cls, embed=det_embs[c])
                new_tracks.append(tr)
                if det_idx in unmatched_dets_global:
                    unmatched_dets_global.remove(det_idx)

            # unmatched existing tracks: mark miss
            for r in unmatched_tracks:
                tr_idx = track_idxs[r]
                self.tracks[tr_idx].misses += 1

        # Append new tracks
        self.tracks.extend(new_tracks)

        # Prune dead tracks
        alive_tracks = []
        for tr in self.tracks:
            if tr.time_since_update > self.max_age:
                # drop
                continue
            else:
                alive_tracks.append(tr)
        self.tracks = alive_tracks

        # Compose output
        outputs = []
        for tr in self.tracks:
            # Only report tracks with minimum hits to avoid noisy short tracks
            if tr.hits >= self.min_hits or tr.time_since_update == 0:
                x1,y1,x2,y2 = tr.bbox
                cx = int((x1 + x2)/2)
                cy = int((y1 + y2)/2)
                outputs.append({
                    "track_id": tr.id,
                    "class": tr.cls,
                    "score": None,
                    "bbox": tr.bbox,
                    "center": [cx, cy],
                    "age": tr.age,
                    "time_since_update": tr.time_since_update
                })
        return outputs
