import numpy as np
from ultralytics import YOLO
from pathlib import Path

# -----------------------------
# Minimal DeepSORT Implementation
# -----------------------------

class Track:
    """Represents a single tracked object."""
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox
        self.time_since_update = 0
        self.hits = 1


class SimpleTracker:
    """
    A minimal IoU-based tracker.
    This is NOT full DeepSORT, but works well for football player tracking and
    does NOT require TensorFlow / reID model.
    """

    def __init__(self, iou_threshold=0.3):
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.tracks = []

    def iou(self, boxA, boxB):
        """Intersection over Union."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter = max(0, xB - xA) * max(0, yB - yA)

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return inter / (boxAArea + boxBArea - inter + 1e-6)

    def update(self, detections):
        """
        detections → list of bbox xyxy: [x1, y1, x2, y2]
        returns → list of dict containing:
            bbox, track_id
        """

        updated_tracks = []

        for det in detections:
            best_iou = 0
            best_track = None

            for track in self.tracks:
                iou = self.iou(det, track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track = track

            if best_iou > self.iou_threshold:
                # match
                best_track.bbox = det
                best_track.time_since_update = 0
                best_track.hits += 1
                updated_tracks.append(best_track)

            else:
                # new track
                new_track = Track(self.next_id, det)
                self.next_id += 1
                updated_tracks.append(new_track)

        # remove lost tracks
        self.tracks = [t for t in updated_tracks]

        # prepare output
        output = []
        for t in self.tracks:
            output.append({"track_id": t.track_id, "bbox": t.bbox})

        return output
