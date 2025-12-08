# spotters/event_spotter.py
import numpy as np
import json
import time
from pathlib import Path
from collections import deque

class EventSpotter:
    def __init__(self, window=5, shot_threshold=8.0, accel_threshold=2.0, fps=25):
        self.window = window
        self.ball_history = deque(maxlen=window)
        self.player_history = {}
        self.shot_threshold = shot_threshold  # in pixels per second (approx)
        self.accel_threshold = accel_threshold
        self.fps = fps
        self.events = []

    def update(self, frame_idx, detections, tracked_objects):
        # detections: [x1,y1,x2,y2,cls,conf]
        ball_boxes = [d for d in detections if d[4] == 1]
        player_boxes = [d for d in detections if d[4] == 0]

        ball_pos = self._get_ball_center(ball_boxes)
        self.ball_history.append((frame_idx, ball_pos))

        ball_speed = self._calculate_ball_speed()  # in pixels/frame
        ball_speed_s = ball_speed * self.fps if ball_speed is not None else None  # px/sec

        if ball_speed_s is not None and ball_speed_s > self.shot_threshold:
            self._record_event(frame_idx, "shot_detected", {"speed_px_per_s": ball_speed_s})

        # Player acceleration
        for track_id, box in tracked_objects:
            cx, cy = self._bbox_center(box)
            if track_id not in self.player_history:
                self.player_history[track_id] = deque(maxlen=self.window)
            self.player_history[track_id].append((frame_idx, (cx, cy)))
            accel = self._calculate_player_acceleration(track_id)
            if accel is not None:
                # accel in px/frame^2 -> convert to px/s^2
                accel_s = accel * (self.fps**2)
                if accel_s > self.accel_threshold:
                    self._record_event(frame_idx, "player_accelerated", {"track_id": track_id, "accel_px_per_s2": accel_s})

        return {"ball_speed_px_per_frame": float(ball_speed) if ball_speed is not None else None, "events": self.events[-10:]}

    def _bbox_center(self, box):
        x1, y1, x2, y2 = box
        return ( (x1+x2)/2.0, (y1+y2)/2.0 )

    def _get_ball_center(self, ball_boxes):
        if len(ball_boxes)==0:
            return None
        best = sorted(ball_boxes, key=lambda x: x[5] if len(x)>5 else 1, reverse=True)[0]
        x1,y1,x2,y2 = best[:4]
        return ( (x1+x2)/2.0, (y1+y2)/2.0 )

    def _calculate_ball_speed(self):
        if len(self.ball_history) < 2:
            return None
        (_, p1), (_, p2) = self.ball_history[-2], self.ball_history[-1]
        if p1 is None or p2 is None:
            return None
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        speed = np.sqrt(dx*dx + dy*dy)  # px per frame
        return float(speed)

    def _calculate_player_acceleration(self, track_id):
        hist = self.player_history.get(track_id)
        if hist is None or len(hist) < 3:
            return None
        (_, p1), (_, p2), (_, p3) = hist[-3], hist[-2], hist[-1]
        v1 = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        v2 = np.sqrt((p3[0]-p2[0])**2 + (p3[1]-p2[1])**2)
        accel = v2 - v1  # px/frame^2
        return float(accel)

    def _record_event(self, frame_idx, event_type, data):
        self.events.append({"frame": int(frame_idx), "type": event_type, "data": data, "ts": time.time()})

    def save_events(self, output_path):
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self.events, f, indent=2)
