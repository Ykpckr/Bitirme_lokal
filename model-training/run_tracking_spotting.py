#!/usr/bin/env python3
# run_tracking_spotting.py  (DÜZELTİLMİŞ)
import os
import argparse
from pathlib import Path
import logging
import cv2
import time
from ultralytics import YOLO
import numpy as np

# utils & modules (senin repo yapına göre relative import olacak)
from utils.visualization_utils import (
    get_video_info, draw_detections, add_text_overlay, ensure_dir
)
from trackers.deepsort_tracker import DeepSortTracker
from spotters.event_spotter import EventSpotter
from utils.config_utils import create_cli_parser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_tracking_spotting")


def safe_tracks_to_dicts(tracks):
    """
    Tracker döndürdüğü formatlar:
      - [(id, [x1,y1,x2,y2]), ...]  (tuples)
      - [{'track_id': id, 'bbox': [x1,y1,x2,y2]}, ...] (dicts)
    Bu helper her iki durumu da dict listesine çevirir.
    """
    out = []
    for t in tracks:
        if isinstance(t, dict):
            tid = t.get('track_id') or t.get('id') or t.get('track')
            bbox = t.get('bbox') or t.get('box') or t.get('xyxy')
            out.append({'track_id': int(tid), 'bbox': [float(x) for x in bbox]})
        elif isinstance(t, (tuple, list)) and len(t) >= 2:
            out.append({'track_id': int(t[0]), 'bbox': [float(x) for x in t[1]]})
        else:
            # bilinmeyen format - atla
            continue
    return out


def main():
    parser = create_cli_parser()
    parser.add_argument('--model', required=True, help='Path to YOLO .pt model (player+ball)')
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--save_video', action='store_true', help='Save output video')
    parser.add_argument('--reid', default=None, help='Optional ReID model file (torch) for better tracking (not used by minimal tracker)')
    parser.add_argument('--conf', type=float, default=0.3, help='YOLO confidence threshold')
    parser.add_argument('--max_age', type=int, default=200, help='Tracker max_age (frames)')
    parser.add_argument('--n_init', type=int, default=3, help='Tracker n_init (not used by minimal tracker)')
    args = parser.parse_args()

    model_path = Path(args.model)
    video_path = Path(args.video)
    output_dir = Path(args.output)
    ensure_dir(output_dir)

    logger.info(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    # Video info
    vid_info = get_video_info(...) 
    fps = vid_info['fps'] or 25
    frame_w, frame_h = vid_info.get('width'), vid_info.get('height')
    logger.info(f"Video FPS: {fps}, size: {frame_w}x{frame_h}")

    # Initialize tracker (minimal DeepSort-like implemented)
    tracker = DeepSortTracker(
        max_age=args.max_age,
        max_cosine=0.6,
        max_spatial_dist=120,
    )

    # Initialize spotter
    # NOTE: Eğer EventSpotter.__init__ fps parametresi alıyorsa buraya fps=fps ekle,
    # ama verilen EventSpotter kodunda fps yoktu -> bu yüzden eklemedim.
    spotter = EventSpotter(window=int(max(3, round(fps/5))), shot_threshold=20.0, accel_threshold=200.0, fps=fps)

    # Video IO
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Could not open video: %s", video_path)
        return

    save_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = output_dir / f"tracked_{video_path.stem}.mp4"
        save_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, frame_h))
        logger.info(f"Saving output video to: {out_path}")

    frame_idx = 0
    start_t = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Run YOLO inference
            results = model(frame, conf=args.conf, verbose=False)
            r = results[0]
            dets = []
            # Extract bboxes: xyxy, conf, cls
            if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), cls, conf in zip(boxes, class_ids, confs):
                    dets.append([float(x1), float(y1), float(x2), float(y2), int(cls), float(conf)])


            # YOLO: [x1,y1,x2,y2,cls,conf] → Tracker: sadece [x1,y1,x2,y2]
            tracker_inputs = [d[:4] for d in dets]

            # --- TRACKER UPDATE ---
            # Important fix: pass detections + features=None + frame_id
            # (previously frame was passed as features which caused errors)
            tracks = tracker.update(tracker_inputs, features=None, frame_id=frame_idx)


            # Convert to both formats as needed
            tracked_objects = [(t[0], t[1]) for t in tracks]  # for spotter (list of tuples)
            tracks_for_draw = safe_tracks_to_dicts(tracks)    # for draw function (list of dicts)

            # Update spotter (expects list of detections and list of (id,bbox))
            spot_info = spotter.update(frame_idx, dets, tracked_objects)

            # Draw detections + tracks on frame
            annotated = draw_detections(frame.copy(), dets, tracks_for_draw, spot_info=spot_info)

            # Add overlay stats
            add_text_overlay(annotated, [
                f"Frame: {frame_idx}",
                f"Tracks: {len(tracked_objects)}",
                f"Events logged: {len(spotter.events)}"
            ])

            # Show / Save
            cv2.imshow("Tracking & Spotting", annotated)
            if args.save_video and save_writer:
                save_writer.write(annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception("Processing error: %s", e)
    finally:
        cap.release()
        if save_writer:
            save_writer.release()
        cv2.destroyAllWindows()

    elapsed = time.time() - start_t
    logger.info(f"Done. Processed {frame_idx} frames in {elapsed:.2f}s ({(frame_idx/elapsed):.2f} fps)")

    # Save event log
    event_file = output_dir / f"{video_path.stem}_events.json"
    spotter.save_events(str(event_file))
    logger.info(f"Events saved to: {event_file}")


if __name__ == "__main__":
    main()
