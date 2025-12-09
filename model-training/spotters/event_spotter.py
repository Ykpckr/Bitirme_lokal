import numpy as np
import json
from collections import deque
from pathlib import Path
import time
import math

class EventSpotter:
    def __init__(self, window=5, shot_threshold=20.0, accel_threshold=100.0, fps: float = 25.0):
        self.window = int(window)
        # Hareketli ortalama için deque
        self.ball_history = deque(maxlen=self.window)
        self.player_history = {} 
        
        self.shot_threshold = float(shot_threshold)
        self.accel_threshold = float(accel_threshold)
        self.fps = float(fps)
        self.events = []
        
        self._last_ball_speed = 0.0

    def update(self, frame_idx, detections, tracked_objects):
        """
        detections: [x1, y1, x2, y2, cls, conf]
        tracked_objects: [(track_id, [x1, y1, x2, y2])]
        """
        # --- TOP ANALİZİ ---
        # Sınıf ID'si 1 olanları (veya modeline göre top hangisiyse) al
        ball_boxes = [d for d in detections if len(d) >= 5 and int(d[4]) == 1] # 1: Sports ball (COCO)
        ball_pos = self._get_ball_center(ball_boxes)
        
        self.ball_history.append((frame_idx, ball_pos))
        
        current_ball_speed = self._calculate_ball_speed()
        
        # Basit bir "Smoothing" (Ani fırlamaları engellemek için)
        if current_ball_speed is not None:
            smoothed_speed = 0.7 * current_ball_speed + 0.3 * self._last_ball_speed
            self._last_ball_speed = smoothed_speed
            
            if smoothed_speed > self.shot_threshold:
                # Son 1 saniyede benzer olay yoksa kaydet
                if not self._is_duplicate_event("shot_detected", frame_idx, buffer_frames=self.fps):
                    self._record_event(frame_idx, "shot_detected", {"speed": float(smoothed_speed)})
        else:
            smoothed_speed = 0.0

        # --- OYUNCU ANALİZİ ---
        for track_id, box in tracked_objects:
            cx, cy = self._bbox_center(box)
            if track_id not in self.player_history:
                self.player_history[track_id] = deque(maxlen=self.window)
            self.player_history[track_id].append((frame_idx, (float(cx), float(cy))))
            
            # İvme hesabı
            accel = self._calculate_player_acceleration(track_id)
            if accel and accel > self.accel_threshold:
                 if not self._is_duplicate_event("player_accelerated", frame_idx, buffer_frames=self.fps, track_id=track_id):
                    self._record_event(frame_idx, "player_accelerated", {
                        "track_id": track_id,
                        "accel": float(accel)
                    })

        return {
            "ball_speed": smoothed_speed,
            "events": self.events[-5:]
        }

    def _bbox_center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _get_ball_center(self, ball_boxes):
        if not ball_boxes: return None
        # En yüksek güven skoruna sahip topu seç
        best_ball = max(ball_boxes, key=lambda x: x[5])
        return self._bbox_center(best_ball[:4])

    def _calculate_ball_speed(self):
        if len(self.ball_history) < 2: return None
        (f1, p1), (f2, p2) = self.ball_history[-2], self.ball_history[-1]
        if p1 is None or p2 is None: return None
        
        dt = (f2 - f1) / self.fps
        if dt <= 0: return None
        
        dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        return dist / dt # px/s

    def _calculate_player_acceleration(self, track_id):
        hist = self.player_history.get(track_id)
        if not hist or len(hist) < 3: return None
        
        # Son 3 pozisyon: p1 -> p2 -> p3
        (t1, p1), (t2, p2), (t3, p3) = hist[-3], hist[-2], hist[-1]
        
        dt1 = (t2 - t1) / self.fps
        dt2 = (t3 - t2) / self.fps
        
        if dt1 <= 0 or dt2 <= 0: return None
        
        v1 = math.hypot(p2[0]-p1[0], p2[1]-p1[1]) / dt1
        v2 = math.hypot(p3[0]-p2[0], p3[1]-p2[1]) / dt2
        
        accel = abs(v2 - v1) / ((dt1 + dt2) / 2)
        return accel

    def _is_duplicate_event(self, event_type, current_frame, buffer_frames, track_id=None):
        """Aynı olayın ardışık karelerde tekrar tekrar kaydedilmesini önler."""
        limit_frame = current_frame - buffer_frames
        for e in reversed(self.events):
            if e["frame"] < limit_frame:
                break
            if e["type"] == event_type:
                if track_id is not None:
                    if e["data"].get("track_id") == track_id:
                        return True
                else:
                    return True
        return False

    def _record_event(self, frame, etype, data):
        self.events.append({
            "frame": int(frame),
            "type": etype,
            "data": data,
            "timestamp": time.time()
        })

    def save_events(self, path):
        with open(path, "w") as f:
            json.dump(self.events, f, indent=4)