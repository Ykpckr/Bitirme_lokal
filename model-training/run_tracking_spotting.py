#!/usr/bin/env python3
# run_tracking_spotting.py
import os
import argparse
from pathlib import Path
import logging
import cv2
import time
from ultralytics import YOLO
import numpy as np

from utils.visualization_utils import (
    get_video_info, draw_detections, add_text_overlay, ensure_dir
)
from trackers.deepsort_tracker import DeepSortTracker
from spotters.event_spotter import EventSpotter
from utils.config_utils import create_cli_parser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_tracking_spotting")


def main():
    parser = create_cli_parser()
    parser.add_argument('--model', required=True, help='Path to YOLO .pt model (player+ball)')
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--save_video', action='store_true', help='Save output video')
    parser.add_argument('--reid', default=None, help='Optional ReID model file (torch) for better tracking')
    parser.add_argument('--conf', type=float, default=0.3, help='YOLO confidence threshold')
    parser.add_argument('--max_age', type=int, default=200, help='Tracker max_age')
    parser.add_argument('--n_init', type=int, default=3, help='Tracker n_init')
    args = parser.parse_args()

    model_path = Path(args.model)
    video_path = Path(args.video)
    output_dir = Path(args.output)
    ensure_dir(output_dir)

    logger.info(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    # Video info
    vid_info = get_video_info(str(video_path))
    fps = vid_info['fps'] or 25
    frame_w, frame_h = vid_info['width'], vid_info['height']
    logger.info(f"Video FPS: {fps}, size: {frame_w}x{frame_h}")

    # Initialize tracker (with optional ReID)
    tracker = DeepSortTracker(max_age=args.max_age, n_init=args.n_init, reid_path=args.reid)

    # Initialize spotter (use fps to convert speed units properly)
    spotter = EventSpotter(window=int(max(3, round(fps/5))), shot_threshold=20.0, accel_threshold=5.0, fps=fps)

    # Video IO
    cap = cv2.VideoCapture(str(video_path))
    save_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = output_dir / f"tracked_{video_path.stem}.mp4"
        save_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, frame_h))
        logger.info(f"Saving output video to: {out_path}")

    frame_idx = 0
    start_t = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Run YOLO inference (Ultralytics - returns Results object)
        results = model(frame, conf=args.conf, verbose=False)
        r = results[0]
        dets = []
        # Extract bboxes: xyxy, conf, cls
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), cls, conf in zip(boxes, class_ids, confs):
                dets.append([float(x1), float(y1), float(x2), float(y2), int(cls), float(conf)])

        # Update tracker: tracker expects detections list and frame
        tracks = tracker.update(dets, frame)

        # Convert tracker tracks to format for spotter: [(id, bbox)]
        tracked_objects = [(t['track_id'], t['bbox']) for t in tracks]

        # Update spotter
        spot_info = spotter.update(frame_idx, dets, tracked_objects)

        # Draw detections + tracks on frame
        annotated = draw_detections(frame.copy(), dets, tracks, spot_info=spot_info)

        # Add overlay stats
        add_text_overlay(annotated, [
            f"Frame: {frame_idx}",
            f"Tracks: {len(tracks)}",
            f"Events logged: {len(spotter.events)}"
        ])

        # Show / Save
        cv2.imshow("Tracking & Spotting", annotated)
        if args.save_video and save_writer:
            save_writer.write(annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if save_writer:
        save_writer.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - start_t
    logger.info(f"Done. Processed {frame_idx} frames in {elapsed:.2f}s ({frame_idx/elapsed:.2f} fps)")

    # Save event log
    event_file = output_dir / f"{video_path.stem}_events.json"
    spotter.save_events(str(event_file))
    logger.info(f"Events saved to: {event_file}")


if __name__ == "__main__":
    main()
