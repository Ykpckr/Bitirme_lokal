# tracking/run_tracker.py
import json
import cv2
from pathlib import Path
from .strongsort import StrongSORT
from utils.visualization_utils import draw_detections, add_text_overlay, create_video_writer, get_video_info

def run_tracker_on_video(
    video_path,
    detection_generator,  # function that yields (frame_idx, frame, detections) for the video
    output_video_path=None,
    output_json_path=None,
    device='cpu',
    viz=True
):
    info = get_video_info(video_path)
    fps = info['fps']
    frame_size = (info['width'], info['height'])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    writer = None
    if output_video_path:
        writer = create_video_writer(str(output_video_path), fps, frame_size)

    tracker = StrongSORT(device=device)
    results_log = []

    for frame_idx, frame, detections in detection_generator(video_path):
        tracks = tracker.update(frame, detections)

        # create overlay detections for viz
        viz_items = []
        for d in detections:
            viz_items.append({"bbox": d['bbox'], "color": (255,255,0)})
        for t in tracks:
            x1,y1,x2,y2 = t['bbox']
            viz_items.append({"bbox": t['bbox'], "color": (0,255,0)})
            add_text_overlay(frame, f"ID:{t['track_id']}", (x1, y1-8))

        if writer:
            writer.write(frame)

        # log
        results_log.append({
            "frame_idx": frame_idx,
            "detections": detections,
            "tracks": tracks
        })

    # save json
    if output_json_path:
        Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results_log, f, indent=2)

    if writer:
        writer.release()
    cap.release()
    return results_log
