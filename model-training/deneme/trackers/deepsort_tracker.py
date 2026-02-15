# trackers/deepsort_tracker.py
# Requires: pip install deep_sort_realtime opencv-python numpy
# Wrapper for Deep SORT (deep_sort_realtime)
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DS_AVAILABLE = True
except Exception:
    DS_AVAILABLE = False
    logger.warning("deep_sort_realtime not available. Install with: pip install deep_sort_realtime")

# Helper: convert xyxy -> xywh
def xyxy_to_xywh(bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return (int(x1), int(y1), int(w), int(h))

class DeepSortTracker:
    """
    Unified interface for DeepSort tracker.
    Usage:
        tracker = DeepSortTracker(max_age=30, n_init=3)
        tracks = tracker.update(detections)
    detections format: list of dicts:
      {"bbox": (x1,y1,x2,y2), "confidence": 0.9, "class_id": 1}
    Returns list of tracks:
      {"track_id": int, "bbox": (x1,y1,x2,y2), "class_id": int, "age": int, "time_since_update": int}
    """
    def __init__(self, max_age: int = 30, n_init: int = 3, max_cosine_distance: float = 0.2):
        self.max_age = max_age
        self.n_init = n_init
        self.max_cosine_distance = max_cosine_distance

        if DS_AVAILABLE:
            # DeepSort from library
            self.ds = DeepSort(max_age=self.max_age, n_init=self.n_init, max_cosine_distance=self.max_cosine_distance)
            logger.info("DeepSort tracker initialized (deep_sort_realtime).")
        else:
            logger.warning("DeepSort not available. Falling back to SimpleIOUTracker.")
            self.ds = None
            from .simple_iou_tracker import SimpleIOUTracker
            self.simple = SimpleIOUTracker(max_age=self.max_age, min_hits=self.n_init)

    def update(self, detections: List[Dict[str, Any]], frame_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Update tracker with detections.
        detections: [{"bbox": (x1,y1,x2,y2), "confidence": float, "class_id": int}, ...]
        """
        if DS_AVAILABLE and self.ds:
            # DeepSort expects xywh and confidence
            dets_for_ds = []
            for det in detections:
                x1,y1,x2,y2 = det['bbox']
                xywh = xyxy_to_xywh((x1,y1,x2,y2))
                dets_for_ds.append((xywh[0], xywh[1], xywh[2], xywh[3], det.get('confidence', 1.0), det.get('class_id', 0)))
            tracks = self.ds.update_tracks(dets_for_ds, frame=frame_id)
            out = []
            for t in tracks:
                if not t.is_confirmed():
                    continue
                track_id = t.track_id
                ltrb = t.to_ltrb()  # left top right bottom
                class_id = t.det_class if hasattr(t, 'det_class') else t.get_det_class() if hasattr(t, 'get_det_class') else 0
                out.append({
                    "track_id": int(track_id),
                    "bbox": (int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])),
                    "class_id": int(class_id),
                    "age": int(t.age),
                    "time_since_update": int(t.time_since_update)
                })
            return out
        else:
            # Simple IOU fallback
            simple_in = []
            for det in detections:
                x1,y1,x2,y2 = det['bbox']
                conf = det.get('confidence', 1.0)
                cls = det.get('class_id', 0)
                simple_in.append((x1,y1,x2,y2, conf, cls))
            tracks = self.simple.update(simple_in)
            # tracks: list of tuples (track_id, x1,y1,x2,y2, class_id)
            out = []
            for tr in tracks:
                track_id, x1,y1,x2,y2, cls = tr
                out.append({
                    "track_id": int(track_id),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "class_id": int(cls),
                    "age": 0,
                    "time_since_update": 0
                })
            return out

