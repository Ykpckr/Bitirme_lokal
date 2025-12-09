import numpy as np
import math
import json
import time
from collections import deque
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class EventSpotter:
    def __init__(self, cfg):
        """
        Olay ve istatistik yakalayıcı modül.
        Args:
            cfg (Config): config.yaml ayarları
        """
        self.cfg = cfg
        self.fps = cfg.video.get('target_fps', 25.0)
        
        # Ayarlar
        self.shot_threshold = cfg.spotter.get('shot_threshold', 25.0) # px/frame veya m/s
        self.accel_threshold = cfg.spotter.get('accel_threshold', 100.0)
        window_size = cfg.spotter.get('smooth_window', 5)
        
        # Veri Yapıları
        self.ball_history = deque(maxlen=window_size)
        self.player_histories = {} # {track_id: deque([pos1, pos2...])}
        self.events = []
        
        self.window_size = window_size
        self._last_ball_speed = 0.0

    def update(self, frame_idx, detections, tracks):
        """
        Her karede çağrılır.
        Args:
            frame_idx (int): Kare numarası
            detections (list): Raw YOLO çıktıları (Top için) -> [x1,y1,x2,y2,conf,cls]
            tracks (list): Tracker çıktıları (Oyuncular için) -> [{'track_id':.., 'bbox':..}]
        """
        spot_info = {
            "ball_speed": 0.0,
            "events": []
        }

        # --- 1. TOP ANALİZİ (YOLO Detections Kullanır) ---
        # Class ID 1 genelde toptur (Config'e göre değişebilir ama standart 1'dir)
        ball_box = None
        best_conf = 0.0
        
        for det in detections:
            # det: [x1, y1, x2, y2, conf, cls]
            if int(det[5]) == 1: # Ball Class
                if det[4] > best_conf:
                    best_conf = det[4]
                    ball_box = det[:4]
        
        if ball_box is not None:
            center = self._get_center(ball_box)
            self.ball_history.append((frame_idx, center))
        else:
            self.ball_history.append((frame_idx, None))

        # Top Hız Hesabı
        speed = self._calculate_ball_speed()
        if speed is not None:
            # Smoothing (Ani değişimleri yumuşat)
            smoothed_speed = 0.7 * speed + 0.3 * self._last_ball_speed
            self._last_ball_speed = smoothed_speed
            spot_info["ball_speed"] = smoothed_speed

            # Olay: Şut Algılama
            if smoothed_speed > self.shot_threshold:
                if not self._is_duplicate_event("shot", frame_idx, buffer=self.fps):
                    event = {
                        "frame": frame_idx,
                        "type": "shot",
                        "value": float(f"{smoothed_speed:.2f}"),
                        "desc": f"Shot detected! Speed: {smoothed_speed:.1f} px/s"
                    }
                    self.events.append(event)
                    spot_info["events"].append(event)
                    logger.info(f"⚽ ŞUT ALGILANDI! Frame: {frame_idx}, Hız: {smoothed_speed:.1f}")

        # --- 2. OYUNCU ANALİZİ (Tracker Output Kullanır) ---
        current_ids = set()
        for t in tracks:
            tid = t['track_id']
            bbox = t['bbox']
            center = self._get_center(bbox)
            current_ids.add(tid)
            
            if tid not in self.player_histories:
                self.player_histories[tid] = deque(maxlen=self.window_size)
            
            self.player_histories[tid].append((frame_idx, center))
            
            # İvme Hesabı
            accel = self._calculate_player_accel(tid)
            if accel and accel > self.accel_threshold:
                if not self._is_duplicate_event("sprint", frame_idx, buffer=self.fps, track_id=tid):
                    event = {
                        "frame": frame_idx,
                        "type": "sprint",
                        "track_id": tid,
                        "value": float(f"{accel:.2f}"),
                        "desc": f"Player {tid} sprinting!"
                    }
                    self.events.append(event)
                    # spot_info["events"].append(event) # Ekrana çok yazı basmamak için kapalı tutabiliriz

        # Kayıp oyuncuları hafızadan sil (Memory leak önleme)
        if frame_idx % 100 == 0:
            self._cleanup_histories(current_ids)

        return spot_info

    def save_events(self, output_dir):
        """Olayları JSON olarak kaydeder"""
        path = Path(output_dir) / "match_events.json"
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.events, f, indent=4)
            logger.info(f"Olaylar kaydedildi: {path}")
        except Exception as e:
            logger.error(f"Olaylar kaydedilemedi: {e}")

    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _calculate_ball_speed(self):
        """Son konumlara göre piksel/saniye hızı hesaplar"""
        if len(self.ball_history) < 2: return None
        
        # Son iki geçerli (None olmayan) konumu bul
        valid_points = [p for p in self.ball_history if p[1] is not None]
        if len(valid_points) < 2: return None
        
        (f1, p1), (f2, p2) = valid_points[-2], valid_points[-1]
        
        # Frame farkı (Eğer top bir süre kaybolup geri geldiyse hızı doğru ölçmek için)
        frame_diff = f2 - f1
        if frame_diff <= 0: return None
        
        time_elapsed = frame_diff / self.fps
        dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        
        return dist / time_elapsed

    def _calculate_player_accel(self, tid):
        """Basit ivme hesabı: |v2 - v1| / t"""
        hist = self.player_histories[tid]
        if len(hist) < 3: return None
        
        # Son 3 nokta: p1 -> p2 -> p3
        (t1, p1), (t2, p2), (t3, p3) = hist[-3], hist[-2], hist[-1]
        
        dt = 1.0 / self.fps
        
        # Hızlar
        v1 = math.hypot(p2[0]-p1[0], p2[1]-p1[1]) / ( (t2-t1) * dt )
        v2 = math.hypot(p3[0]-p2[0], p3[1]-p2[1]) / ( (t3-t2) * dt )
        
        # İvme
        accel = abs(v2 - v1) / dt
        return accel

    def _is_duplicate_event(self, etype, frame_idx, buffer, track_id=None):
        """Aynı olayın peş peşe (buffer süresi içinde) kaydedilmesini önler"""
        limit_frame = frame_idx - buffer
        
        # Listeyi tersten tara (en yeniler sonda)
        for event in reversed(self.events):
            if event['frame'] < limit_frame:
                break
            if event['type'] == etype:
                if track_id is not None:
                    if event.get('track_id') == track_id:
                        return True
                else:
                    return True
        return False

    def _cleanup_histories(self, current_ids):
        """Ekranda olmayan oyuncuların geçmiş verisini temizle"""
        # Python 3.x'te dictionary boyutu değişirken iterasyon hatası almamak için list(keys)
        for tid in list(self.player_histories.keys()):
            if tid not in current_ids:
                del self.player_histories[tid]