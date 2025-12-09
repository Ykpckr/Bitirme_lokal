import cv2
from pathlib import Path

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"fps": None, "width": 0, "height": 0, "frames": 0}
    
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    cap.release()
    return info

def draw_detections(img, detections, tracks, spot_info=None):
    """
    detections: [x1, y1, x2, y2, cls, conf]
    tracks: [{'track_id': int, 'bbox': [x1,y1,x2,y2]}]
    """
    # 1. Tracker Çizimi (Mavi Kutu ve ID)
    for t in tracks:
        tid = t['track_id']
        x1, y1, x2, y2 = map(int, t['bbox'])
        
        # Kutu
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
        
        # Etiket Arka Planı
        label = f"ID:{tid}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 200, 0), -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 2. Ham Detection Çizimi (Opsiyonel - Sadece topu kırmızı çizelim)
    for d in detections:
        x1, y1, x2, y2, cls, conf = d
        if int(cls) == 1: # Top ise (COCO class ID'ye dikkat et)
            cv2.circle(img, (int((x1+x2)/2), int((y1+y2)/2)), 5, (0, 0, 255), -1)

    # 3. Spotter Bilgisi
    if spot_info and spot_info.get('ball_speed'):
        spd = spot_info['ball_speed']
        if spd > 10: # Çok yavaş hareketleri yazdırma
            cv2.putText(img, f"Ball Speed: {spd:.1f} px/s", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return img

def add_text_overlay(img, lines, pos=(10, 20), line_height=25, color=(0, 255, 0)):
    x, y = pos
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + i * line_height), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)