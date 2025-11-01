#!/usr/bin/env python3
"""
🎬 Quick Video Prediction Example
Chelsea vs Swansea Match Analysis with YOLO Ball & Player Detection
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from video_prediction import VideoPredictor

def main():
    """Run prediction on Chelsea vs Swansea video"""
    
    print("🎯 YOLO Player & Ball Detection - Chelsea vs Swansea")
    print("=" * 60)
    
    # Configuration
    MODEL_PATH = "C:/Users/Admin/Desktop/bitirme/FoMACBitirme/model-training/ball-detection/models/player_ball_detector/weights/best.pt"
    VIDEO_PATH = "C:/Users/Admin/Desktop/bitirme/FoMACBitirme/model-training/ball-detection/soccerNet/england_epl/2015-2016/2015-08-08 - 19-30 Chelsea 2 - 2 Swansea/1_720p.mkv"
    NUM_FRAMES = 100
    FRAMES_PER_GROUP = 10
    CONFIDENCE = 0.1
    OUTPUT_DIR = "predictions_output"
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("Please make sure you have trained the model first!")
        return
    
    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video not found: {VIDEO_PATH}")
        print("Please check the video path!")
        return
    
    print(f"🏋️ Model: {MODEL_PATH}")
    print(f"🎬 Video: {VIDEO_PATH}")
    print(f"🎲 Total frames: {NUM_FRAMES}")
    print(f"📊 Frames per group: {FRAMES_PER_GROUP}")
    print(f"🎯 Confidence threshold: {CONFIDENCE}")
    print()
    
    try:
        # Create predictor
        predictor = VideoPredictor(MODEL_PATH, CONFIDENCE)
        
        # Process video
        results = predictor.process_video(
            video_path=VIDEO_PATH,
            num_frames=NUM_FRAMES,
            frames_per_group=FRAMES_PER_GROUP,
            output_dir=OUTPUT_DIR
        )
        
        print()
        print("🎉 BEFORE/AFTER Comparison completed successfully!")
        print("📊 Final Results:")
        print(f"   Groups processed: {results['total_groups']}")
        print(f"   Frames per group: {results['frames_per_group']}")
        print(f"   Total frames: {results['total_frames_processed']}")
        print()
        class_order = list(results.get('improvement', results['after_predictions']).keys())
        if not class_order:
            class_order = sorted(results['after_predictions'].keys())

        print("🔴 BEFORE Training (Player-only model):")
        for class_name in class_order:
            count = results['before_predictions'].get(class_name, 0)
            print(f"   {class_name}: {count}")

        print()
        print("🟢 AFTER Training (Player + Ball model):")
        for class_name in class_order:
            count = results['after_predictions'].get(class_name, 0)
            print(f"   {class_name}: {count}")

        print()
        print("📈 IMPROVEMENT:")
        for class_name in class_order:
            delta = results['improvement'].get(class_name, 0)
            sign = "+" if delta >= 0 else ""
            print(f"   {class_name}: {sign}{delta}")
        print(f"   Processing time: {results['processing_time']:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()