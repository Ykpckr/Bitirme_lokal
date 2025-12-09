"""
Make Player Crops from YOLO Detections

Extracts player bounding boxes from video using YOLO detector
and organizes them into ReID dataset structure.
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO


def extract_crops_from_video(
    video_path: str,
    output_dir: str,
    yolo_weights: str,
    conf_threshold: float = 0.5,
    player_class: int = 0,
    max_frames: int = None,
    frame_interval: int = 10,
    min_box_area: int = 1000
):
    """
    Extract player crops from video.
    
    Args:
        video_path: Path to input video
        output_dir: Output directory for crops
        yolo_weights: Path to YOLO weights
        conf_threshold: Detection confidence threshold
        player_class: Class ID for players (0 in your model)
        max_frames: Maximum number of frames to process (None for all)
        frame_interval: Process every Nth frame
        min_box_area: Minimum bounding box area to keep
    """
    # Load YOLO model
    print(f"Loading YOLO model from {yolo_weights}")
    model = YOLO(yolo_weights)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {video_path}")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Frame interval: {frame_interval}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process frames
    frame_idx = 0
    saved_crops = 0
    
    pbar = tqdm(total=min(total_frames, max_frames) if max_frames else total_frames)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if max_frames is not None and frame_idx >= max_frames:
            break
        
        # Process every Nth frame
        if frame_idx % frame_interval == 0:
            # Run YOLO detection
            results = model(frame, conf=conf_threshold, verbose=False)
            
            # Extract player detections
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Only keep player class
                    if cls != player_class:
                        continue
                    
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Check box size
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area < min_box_area:
                        continue
                    
                    # Crop player
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size == 0:
                        continue
                    
                    # Save crop
                    crop_filename = (
                        f"frame_{frame_idx:06d}_"
                        f"det_{saved_crops:06d}_"
                        f"conf_{conf:.2f}.jpg"
                    )
                    crop_path = output_dir / crop_filename
                    cv2.imwrite(str(crop_path), crop)
                    
                    saved_crops += 1
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    print(f"\n✓ Extracted {saved_crops} player crops")
    print(f"  Saved to: {output_dir}")
    
    return saved_crops


def organize_crops_by_tracklet(
    tracks_file: str,
    crops_dir: str,
    output_dir: str,
    min_crops_per_id: int = 10
):
    """
    Organize crops by track ID (person ID).
    
    This function assumes you have tracking results with track IDs.
    
    Args:
        tracks_file: Path to tracking results (CSV or TXT)
        crops_dir: Directory with extracted crops
        output_dir: Output directory (train/query/gallery structure)
        min_crops_per_id: Minimum crops per identity to keep
    """
    import pandas as pd
    
    # Load tracking results
    # Expected format: frame, track_id, x1, y1, x2, y2, conf, ...
    print(f"Loading tracks from {tracks_file}")
    
    if tracks_file.endswith('.csv'):
        tracks = pd.read_csv(tracks_file)
    else:
        # MOT format: frame, track_id, x, y, w, h, conf, -1, -1, -1
        tracks = pd.read_csv(
            tracks_file,
            sep=',',
            header=None,
            names=['frame', 'track_id', 'x', 'y', 'w', 'h',
                   'conf', 'x1', 'y1', 'z']
        )
    
    # Group by track_id
    track_groups = tracks.groupby('track_id')
    
    print(f"Found {len(track_groups)} unique track IDs")
    
    # Create output directories
    output_dir = Path(output_dir)
    train_dir = output_dir / 'train'
    query_dir = output_dir / 'query'
    gallery_dir = output_dir / 'gallery'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir.mkdir(parents=True, exist_ok=True)
    
    crops_dir = Path(crops_dir)
    
    # Process each track
    valid_tracks = 0
    
    for track_id, track_data in tqdm(track_groups, desc="Organizing"):
        # Get frames for this track
        frames = track_data['frame'].values
        
        if len(frames) < min_crops_per_id:
            continue
        
        # Find matching crops
        track_crops = []
        
        for frame_num in frames:
            # Find crop for this frame
            pattern = f"frame_{frame_num:06d}_*.jpg"
            matching_crops = list(crops_dir.glob(pattern))
            
            if len(matching_crops) > 0:
                track_crops.append(matching_crops[0])
        
        if len(track_crops) < min_crops_per_id:
            continue
        
        # Split into train/query/gallery
        num_crops = len(track_crops)
        num_query = max(1, num_crops // 10)  # 10% for query
        num_gallery = max(2, num_crops // 5)  # 20% for gallery
        
        # Random split
        indices = np.random.permutation(num_crops)
        query_indices = indices[:num_query]
        gallery_indices = indices[num_query:num_query + num_gallery]
        train_indices = indices[num_query + num_gallery:]
        
        # Copy files
        pid_str = f"pid_{track_id:04d}"
        
        # Train
        if len(train_indices) > 0:
            train_pid_dir = train_dir / pid_str
            train_pid_dir.mkdir(exist_ok=True)
            
            for idx in train_indices:
                src = track_crops[idx]
                dst = train_pid_dir / f"{src.stem}.jpg"
                shutil.copy(src, dst)
        
        # Query
        if len(query_indices) > 0:
            query_pid_dir = query_dir / pid_str
            query_pid_dir.mkdir(exist_ok=True)
            
            for idx in query_indices:
                src = track_crops[idx]
                dst = query_pid_dir / f"{src.stem}.jpg"
                shutil.copy(src, dst)
        
        # Gallery
        if len(gallery_indices) > 0:
            gallery_pid_dir = gallery_dir / pid_str
            gallery_pid_dir.mkdir(exist_ok=True)
            
            for idx in gallery_indices:
                src = track_crops[idx]
                dst = gallery_pid_dir / f"{src.stem}.jpg"
                shutil.copy(src, dst)
        
        valid_tracks += 1
    
    print(f"\n✓ Organized {valid_tracks} valid track IDs")
    print(f"  Output directory: {output_dir}")


def main():
    """Main script."""
    parser = argparse.ArgumentParser(
        description='Extract player crops from YOLO detections'
    )
    
    # Video input
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Path to input video'
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Output directory'
    )
    
    # YOLO settings
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Path to YOLO weights (default: from config)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='Detection confidence threshold'
    )
    parser.add_argument(
        '--player-class',
        type=int,
        default=0,
        help='Player class ID'
    )
    
    # Processing settings
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum frames to process'
    )
    parser.add_argument(
        '--frame-interval',
        type=int,
        default=10,
        help='Process every Nth frame'
    )
    parser.add_argument(
        '--min-box-area',
        type=int,
        default=1000,
        help='Minimum bounding box area'
    )
    
    # Track-based organization
    parser.add_argument(
        '--tracks',
        type=str,
        default=None,
        help='Path to tracking results for organizing by person ID'
    )
    parser.add_argument(
        '--min-crops-per-id',
        type=int,
        default=10,
        help='Minimum crops per person ID'
    )
    
    args = parser.parse_args()
    
    # Load YOLO weights path from config if not provided
    if args.weights is None:
        config_path = Path(__file__).parent.parent / 'configs' / 'example_config.yaml'
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        args.weights = cfg['paths']['yolo_weights']
    
    print("="*60)
    print("Player Crop Extraction")
    print("="*60)
    
    if args.tracks is None:
        # Simple extraction without track IDs
        extract_crops_from_video(
            video_path=args.video,
            output_dir=args.out,
            yolo_weights=args.weights,
            conf_threshold=args.conf,
            player_class=args.player_class,
            max_frames=args.max_frames,
            frame_interval=args.frame_interval,
            min_box_area=args.min_box_area
        )
        
        print("\nNote: Crops are not organized by person ID.")
        print("To organize by person ID, provide --tracks argument")
        print("with tracking results.")
    else:
        # Extract crops first
        crops_dir = Path(args.out) / 'raw_crops'
        
        extract_crops_from_video(
            video_path=args.video,
            output_dir=str(crops_dir),
            yolo_weights=args.weights,
            conf_threshold=args.conf,
            player_class=args.player_class,
            max_frames=args.max_frames,
            frame_interval=args.frame_interval,
            min_box_area=args.min_box_area
        )
        
        # Organize by track ID
        print("\n" + "="*60)
        print("Organizing crops by person ID")
        print("="*60)
        
        organize_crops_by_tracklet(
            tracks_file=args.tracks,
            crops_dir=str(crops_dir),
            output_dir=args.out,
            min_crops_per_id=args.min_crops_per_id
        )
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
