# trackers/simple_iou_tracker.py
# Very simple IOU-based tracker for fallback usage
# No external dependencies beyond numpy

import numpy as np
from typing import List, Tuple
import logging
logger = logging.getLogger(__name__)

def iou(bb1, bb2):
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    inter = w * h
    a = (bb1[2]-bb1[0])*(bb1[3]-bb1[1])
    b = (bb2[2]-bb2[0])*(bb2[3]-bb2[1])
    union = a + b - inter
    return inter / union if union>0 else 0.0

class SimpleIOUTracker:
    """
    Simple tracker that assigns detections to existing tracks by IOU.
    Not robust but OK as fallback.
    """
    def __init__(self, iou_threshold=0.3, max_age=30, min_hits=1):
        self.iou_thresh = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_id = 1
        self.tracks = {}  # id -> dict {bbox, age, hits, class_id}

    def update(self, detections: List[Tuple[int,int,int,int,float,int]]):
        """
        detections: list of tuples (x1,y1,x2,y2, conf, class_id)
        returns: list of tuples (track_id, x1,y1,x2,y2, class_id)
        """
        assigned = set()
        det_boxes = [d[:4] for d in detections]
        det_cls = [d[5] for d in detections]

        # Build candidate IoU matrix
        track_ids = list(self.tracks.keys())
        if len(track_ids)==0:
            # initialize
            out = []
            for i, d in enumerate(detections):
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {"bbox": d[:4], "age":0, "hits":1, "class_id": d[5]}
                out.append((tid, *d[:4], d[5]))
            return out

        iou_mat = np.zeros((len(track_ids), len(det_boxes)), dtype=float)
        for i, tid in enumerate(track_ids):
            for j, db in enumerate(det_boxes):
                iou_mat[i,j] = iou(self.tracks[tid]["bbox"], db)

        # Greedy match
        while True:
            i,j = divmod(iou_mat.argmax(), iou_mat.shape[1])
            if iou_mat.size==0:
                break
            if iou_mat[i,j] < self.iou_thresh:
                break
            tid = track_ids[i]
            if tid in assigned:
                iou_mat[i,j] = -1
                continue
            # assign
            self.tracks[tid]["bbox"] = det_boxes[j]
            self.tracks[tid]["age"] = 0
            self.tracks[tid]["hits"] += 1
            self.tracks[tid]["class_id"] = det_cls[j]
            assigned.add(tid)
            iou_mat[i,:] = -1
            iou_mat[:,j] = -1

        # unmatched detections -> create new tracks
        out = []
        for idx, d in enumerate(detections):
            matched = False
            for tid in assigned:
                if self.tracks[tid]["bbox"] == d[:4]:
                    matched = True
            if not matched:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {"bbox": d[:4], "age":0, "hits":1, "class_id": d[5]}
                assigned.add(tid)

        # Age unmatched tracks and remove old
        to_delete = []
        for tid in list(self.tracks.keys()):
            if tid not in assigned:
                self.tracks[tid]["age"] += 1
            if self.tracks[tid]["age"] > self.max_age:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        # Build out
        for tid, v in self.tracks.items():
            out.append((tid, *v["bbox"], v["class_id"]))
        return out
