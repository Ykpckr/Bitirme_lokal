import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        # Takımlar için renk paleti (İleride takım ayrımı eklenirse diye)
        self.colors = [
            (255, 0, 0),   # Mavi
            (0, 0, 255),   # Kırmızı
            (0, 255, 0),   # Yeşil
            (0, 255, 255), # Sarı
            (255, 0, 255), # Mor
        ]

    def draw_tracks(self, frame, tracks):
        """
        Takip edilen oyuncuları çizer.
        tracks: [{'track_id': int, 'bbox': [x1,y1,x2,y2]}, ...]
        """
        for t in tracks:
            tid = t['track_id']
            x1, y1, x2, y2 = map(int, t['bbox'])
            
            # Renk seçimi (ID'ye göre sabit renk)
            color = self.colors[tid % len(self.colors)]
            
            # 1. Kutu (Köşeleri ovalimsi veya kalın çizgi)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 2. ID Etiketi (Kutunun üstünde şık bir kutucuk)
            label = f"ID: {tid}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            # Etiket arka planı
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            
            # Etiket yazısı (Beyaz)
            cv2.putText(frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def draw_ball(self, frame, detections):
        """Topu (Class ID: 1) çizer"""
        for det in detections:
            # det: [x1, y1, x2, y2, score, cls]
            if int(det[5]) == 1: 
                x1, y1, x2, y2 = map(int, det[:4])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Topun etrafına parlak bir çember
                cv2.circle(frame, (cx, cy), 5, (0, 165, 255), -1) # Turuncu
                cv2.circle(frame, (cx, cy), 10, (0, 165, 255), 2)
        return frame

    def draw_stats(self, frame, spot_info, frame_idx, fps):
        """Sol üst köşeye istatistikleri basar"""
        # Panel arka planı (Yarı saydam siyah)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 120), (0, 0, 0), -1)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Satırlar
        lines = [
            f"Frame: {frame_idx}",
            f"FPS: {fps:.1f}",
        ]
        
        if spot_info.get("ball_speed"):
            lines.append(f"Ball Speed: {spot_info['ball_speed']:.1f} px/s")
        
        # Son olay (Varsa)
        if spot_info.get("events"):
            last_event = spot_info["events"][-1]
            lines.append(f"EVENT: {last_event['type'].upper()} ({last_event['value']})")

        # Yazdır
        y = 25
        for line in lines:
            color = (0, 255, 0) if "EVENT" not in line else (0, 0, 255)
            cv2.putText(frame, line, (10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
            y += 25
            
        return frame