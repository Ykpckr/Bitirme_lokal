# tracking/tracker_state.py
import numpy as np
from .kalman import KalmanFilter

class TrackState:
    _next_id = 1

    def __init__(self, bbox, cls, embed=None):
        """
        bbox: [x1,y1,x2,y2]
        cls: 'player' or 'ball'
        embed: appearance vector (np.array) or None
        """
        self.id = TrackState._next_id
        TrackState._next_id += 1

        self.cls = cls
        self.kf = KalmanFilter()
        self.mean, self.cov = self.kf.initiate(bbox)
        self.bbox = bbox  # last bbox in xyxy
        self.embed = embed
        self.hits = 1
        self.misses = 0
        self.age = 0
        self.time_since_update = 0

    def predict(self):
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.age += 1
        self.time_since_update += 1
        # update bbox from state (cx,cy,w,h)
        cx, cy, w, h = self.mean[0:4]
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        self.bbox = [int(x1), int(y1), int(x2), int(y2)]
        return self.bbox

    def update(self, bbox, embed=None):
        # convert bbox to measurement
        x1,y1,x2,y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        meas = [cx,cy,w,h]
        self.mean, self.cov = self.kf.update(self.mean, self.cov, meas)
        self.bbox = [int(x1), int(y1), int(x2), int(y2)]
        self.hits += 1
        self.misses = 0
        self.time_since_update = 0
        if embed is not None:
            self.embed = embed
