# utils/visualization_utils.py
import cv2
from pathlib import Path
import numpy as np
import os

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"fps": None, "width": 640, "height": 480, "frames": 0}
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"fps": fps, "width": w, "height": h, "frames": frames}

def draw_detections(img, detections, tracks, spot_info=None):
    # detections: [x1,y1,x2,y2,cls,conf], tracks: list of {'track_id','bbox'}
    colors = {0: (0,255,0), 1:(0,0,255)}
    # draw detections
    for d in detections:
        x1,y1,x2,y2,cls,conf = d
        c = tuple(int(v) for v in colors.get(int(cls), (255,255,0)))
        cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), c, 2)
        cv2.putText(img, f"{cls}:{conf:.2f}", (int(x1), max(15,int(y1)-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

    # draw tracks
    for t in tracks:
        tid = t['track_id']
        x1,y1,x2,y2 = t['bbox']
        cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (255,200,0), 2)
        cv2.putText(img, f"ID:{tid}", (int(x1), int(y2)+12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 1)

    # optional: overlay spot_info
    if spot_info:
        if 'ball_speed_px_per_frame' in spot_info and spot_info['ball_speed_px_per_frame'] is not None:
            speed = spot_info['ball_speed_px_per_frame']
            cv2.putText(img, f"Ball speed(px/frame): {speed:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    return img

def add_text_overlay(img, lines, pos=(10,50), line_height=20, color=(255,255,255)):
    x,y = pos
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + i*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img
