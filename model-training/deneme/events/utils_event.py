# events/utils_event.py
import math

def center_from_bbox(bbox):
    x1,y1,x2,y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (cx, cy)

def speed(prev, curr, dt=1.0):
    # Euclidean distance per frame (pixels/frame)
    if prev is None:
        return 0.0
    return math.hypot(curr[0]-prev[0], curr[1]-prev[1]) / dt

def is_close(p1, p2, threshold):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1]) <= threshold
