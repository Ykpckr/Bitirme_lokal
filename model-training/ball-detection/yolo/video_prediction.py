#!/usr/bin/env python3
"""
🎯 YOLO Player & Ball Detection Video Predictor
Processes video files and performs real-time prediction with visual output
"""

import cv2
import numpy as np
import random
import os
import sys
from pathlib import Path
import time
from datetime import datetime
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoPredictor:
    """Video prediction class for YOLO player & ball detection"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.3):
        """
        Initialize the video predictor
        
        Args:
            model_path: Path to trained YOLO model
            confidence_threshold: Minimum confidence for predictions
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Class labels and visualization colors will be initialized after the model loads
        self.class_names = {}
        self.class_colors = {}
        
        # Load the trained model
        logger.info(f"🏋️ Loading YOLO model from: {self.model_path}")
        try:
            self.model = YOLO(str(self.model_path))
            logger.info("✅ Model loaded successfully")
            self._initialize_classes()
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
        
        # Load original player-only model for comparison
        original_model_path = "C:/Users/Admin/Desktop/bitirme/FoMACBitirme/model-training/player-detection/models/football_detector_optimized/weights/best.pt"
        logger.info(f"🔄 Loading original player-only model from: {original_model_path}")
        try:
            self.original_model = YOLO(original_model_path)
            logger.info("✅ Original model loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Could not load original model: {e}")
            self.original_model = None

    def _initialize_classes(self):
        """Load class names from the model and prepare visualization colors."""
        model_names = getattr(self.model, 'names', None)

        if isinstance(model_names, dict):
            self.class_names = {int(key): value for key, value in model_names.items()}
        elif isinstance(model_names, (list, tuple)):
            self.class_names = {idx: name for idx, name in enumerate(model_names)}

        if not self.class_names:
            # Fallback for known training classes
            self.class_names = {0: 'Player', 1: 'Ball', 2: 'Referee'}

        default_colors = {
            'Player': (0, 255, 0),
            'Ball': (0, 0, 255),
            'Referee': (255, 165, 0)
        }

        self.class_colors = {
            class_id: default_colors.get(name, self._generate_color_for_class(class_id, name))
            for class_id, name in self.class_names.items()
        }

    @staticmethod
    def _generate_color_for_class(class_id: int, class_name: str) -> tuple:
        """Generate a deterministic color for unknown classes."""
        seed = class_id * 9973 + sum(ord(ch) for ch in class_name)
        rng = random.Random(seed)
        return (
            rng.randint(64, 255),
            rng.randint(64, 255),
            rng.randint(64, 255)
        )

    def _count_predictions_by_class(self, predictions: list) -> dict:
        """Aggregate detections per class for statistics and reporting."""
        counts = {name: 0 for name in self.class_names.values()}
        for pred in predictions:
            class_name = pred.get('class_name', 'Unknown')
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts
    
    def extract_random_frames(self, video_path: str, num_frames: int = 100, frames_per_group: int = 10) -> list:
        """
        Extract random consecutive frame groups from video
        
        Args:
            video_path: Path to video file
            num_frames: Total number of frames to extract (default: 100)
            frames_per_group: Number of consecutive frames per group (default: 10)
            
        Returns:
            List of groups, each containing (frame_number, frame_image) tuples
        """
        logger.info(f"🎬 Opening video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        logger.info(f"📊 Video Info:")
        logger.info(f"   Total Frames: {total_frames}")
        logger.info(f"   FPS: {fps:.2f}")
        logger.info(f"   Duration: {duration:.2f} seconds")
        
        # Calculate number of groups
        num_groups = num_frames // frames_per_group
        logger.info(f"🎲 Creating {num_groups} groups with {frames_per_group} consecutive frames each")
        
        # Generate random starting points for each group
        # Make sure we have enough space for consecutive frames
        max_start_frame = total_frames - frames_per_group
        if max_start_frame < num_groups:
            logger.warning(f"⚠️ Video too short for {num_groups} groups, reducing to available space")
            num_groups = max(1, max_start_frame // frames_per_group)
        
        # Generate random starting frames for each group
        start_frames = sorted(random.sample(range(0, max_start_frame, frames_per_group * 2), num_groups))
        
        # Extract frame groups
        frame_groups = []
        for group_idx, start_frame in enumerate(start_frames):
            logger.info(f"📸 Extracting group {group_idx + 1}/{num_groups} starting from frame {start_frame}")
            
            group_frames = []
            for i in range(frames_per_group):
                frame_num = start_frame + i
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if ret:
                    group_frames.append((frame_num, frame))
                else:
                    logger.warning(f"⚠️ Could not read frame {frame_num}")
            
            if group_frames:
                frame_groups.append(group_frames)
        
        cap.release()
        logger.info(f"✅ Extracted {len(frame_groups)} groups with total {sum(len(g) for g in frame_groups)} frames")
        return frame_groups
    
    def predict_frame_comparison(self, frame: np.ndarray) -> tuple:
        """
        Perform prediction comparison on a single frame (BEFORE vs AFTER)
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (before_predictions, before_frame, after_predictions, after_frame)
        """
        # BEFORE: Player-only predictions
        before_predictions = []
        before_frame = frame.copy()
        
        if self.original_model:
            results_before = self.original_model(frame, conf=self.confidence_threshold, verbose=False)
            if results_before[0].boxes is not None:
                boxes = results_before[0].boxes
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    confidence = boxes.conf[i].cpu().numpy()
                    # Original model only has players (class 0)
                    before_predictions.append({
                        'class_id': 0,
                        'class_name': 'Player',
                        'confidence': confidence,
                        'bbox': (int(x1), int(y1), int(x2), int(y2))
                    })
            before_frame = self.annotate_frame(before_frame, before_predictions)
        
        # AFTER: Player + Ball predictions
        after_predictions = []
        after_frame = frame.copy()
        
        results_after = self.model(frame, conf=self.confidence_threshold, verbose=False)
        if results_after[0].boxes is not None:
            boxes = results_after[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                confidence = boxes.conf[i].cpu().numpy()
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.class_names.get(class_id, f"Class {class_id}")

                after_predictions.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': (int(x1), int(y1), int(x2), int(y2))
                })
        
        after_frame = self.annotate_frame(after_frame, after_predictions)
        
        return before_predictions, before_frame, after_predictions, after_frame
    
    def annotate_frame(self, frame: np.ndarray, predictions: list) -> np.ndarray:
        """
        Annotate frame with predictions
        
        Args:
            frame: Input frame
            predictions: List of prediction dictionaries
            
        Returns:
            Annotated frame
        """
        for pred in predictions:
            x1, y1, x2, y2 = pred['bbox']
            class_name = pred['class_name']
            confidence = pred['confidence']
            color = self.class_colors.get(pred['class_id'], (255, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(frame, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def display_and_save_comparison_groups(self, frame_groups_with_predictions: list, output_dir: str):
        """
        Display and save each group with BEFORE/AFTER comparison
        
        Args:
            frame_groups_with_predictions: List of groups with comparison predictions
            output_dir: Directory to save results
        """
        logger.info("🖼️ Saving BEFORE/AFTER comparison results for each group...")
        
        # Create main output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_output_dir = os.path.join(output_dir, f"comparison_analysis_{timestamp}")
        os.makedirs(main_output_dir, exist_ok=True)
        
        for group_idx, group_frames in enumerate(frame_groups_with_predictions):
            logger.info(f"💾 Processing group {group_idx + 1}/{len(frame_groups_with_predictions)}")
            
            # Create group directory
            group_dir = os.path.join(main_output_dir, f"group_{group_idx + 1:02d}")
            os.makedirs(group_dir, exist_ok=True)
            
            # Calculate grid size for BEFORE/AFTER comparison
            frames_per_group = len(group_frames)
            cols = 2  # BEFORE and AFTER columns
            rows = frames_per_group
            
            # Create figure for this group
            fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
            fig.suptitle(f'🔄 Group {group_idx + 1} - BEFORE vs AFTER Training Comparison', 
                        fontsize=16, fontweight='bold')
            
            # Ensure axes is 2D
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            # Statistics
            before_stats = {name: 0 for name in self.class_names.values()}
            after_stats = {name: 0 for name in self.class_names.values()}
            class_order = list(self.class_names.values())
            
            for frame_idx, (frame_num, original_frame, before_preds, before_frame, after_preds, after_frame) in enumerate(group_frames):
                before_counts = self._count_predictions_by_class(before_preds)
                after_counts = self._count_predictions_by_class(after_preds)

                for name in before_counts:
                    if name not in class_order:
                        class_order.append(name)
                for name in after_counts:
                    if name not in class_order:
                        class_order.append(name)

                for name, count in before_counts.items():
                    before_stats[name] = before_stats.get(name, 0) + count
                for name, count in after_counts.items():
                    after_stats[name] = after_stats.get(name, 0) + count

                before_title_parts = [f"{name}: {count}" for name, count in before_counts.items() if count > 0]
                after_title_parts = [f"{name}: {count}" for name, count in after_counts.items() if count > 0]
                before_title = " | ".join(before_title_parts) if before_title_parts else "No detections"
                after_title = " | ".join(after_title_parts) if after_title_parts else "No detections"

                # BEFORE (left column)
                ax_before = axes[frame_idx, 0]
                rgb_before = cv2.cvtColor(before_frame, cv2.COLOR_BGR2RGB)
                ax_before.imshow(rgb_before)
                ax_before.set_title(f'BEFORE (Frame {frame_num})\n{before_title}', 
                                  fontweight='bold', color='red')
                ax_before.axis('off')
                
                # AFTER (right column)
                ax_after = axes[frame_idx, 1]
                rgb_after = cv2.cvtColor(after_frame, cv2.COLOR_BGR2RGB)
                ax_after.imshow(rgb_after)
                ax_after.set_title(f'AFTER (Frame {frame_num})\n{after_title}', 
                                 fontweight='bold', color='green')
                ax_after.axis('off')
                
                # Add comparison indicators
                if after_counts.get('Ball', 0) > before_counts.get('Ball', 0):
                    ax_after.text(0.02, 0.98, '🎯 NEW BALL DETECTED!', 
                                transform=ax_after.transAxes, fontsize=12, 
                                verticalalignment='top', fontweight='bold',
                                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            
            # Add column headers
            fig.text(0.25, 0.95, '🔴 BEFORE Training (Player-only)', 
                    fontsize=14, fontweight='bold', ha='center', color='red')
            fig.text(0.75, 0.95, '🟢 AFTER Training (Player + Ball)', 
                    fontsize=14, fontweight='bold', ha='center', color='green')
            
            # Add group summary
            start_frame = group_frames[0][0] if group_frames else 0
            end_frame = group_frames[-1][0] if group_frames else 0
            
            summary_order = list(class_order)
            summary_before = " | ".join(
                f"{name}: {before_stats.get(name, 0)}" for name in summary_order
            )
            summary_after = " | ".join(
                f"{name}: {after_stats.get(name, 0)}" for name in summary_order
            )
            summary_improvement = " | ".join(
                f"{name}: {after_stats.get(name, 0) - before_stats.get(name, 0):+d}" for name in summary_order
            )

            summary_text = f"""
📊 GROUP {group_idx + 1} ANALYSIS (Frames {start_frame}-{end_frame})
🔴 BEFORE: {summary_before}
🟢 AFTER:  {summary_after}
📈 IMPROVEMENT: {summary_improvement}
            """
            
            fig.text(0.02, 0.02, summary_text.strip(),
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.92, bottom=0.15)
            
            # Save the group comparison
            save_path = os.path.join(group_dir, f"before_after_comparison.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Group {group_idx + 1} comparison saved to: {save_path}")
            
            # Save individual BEFORE/AFTER frames
            for frame_idx, (frame_num, original_frame, before_preds, before_frame, after_preds, after_frame) in enumerate(group_frames):
                # Save BEFORE frame
                before_path = os.path.join(group_dir, f"frame_{frame_num:06d}_BEFORE.png")
                cv2.imwrite(before_path, before_frame)
                
                # Save AFTER frame
                after_path = os.path.join(group_dir, f"frame_{frame_num:06d}_AFTER.png")
                cv2.imwrite(after_path, after_frame)
            
            plt.close()
        
        logger.info(f"✅ All {len(frame_groups_with_predictions)} groups saved with BEFORE/AFTER comparison!")
        logger.info(f"📁 Results saved in: {main_output_dir}")
    
    def process_video(self, video_path: str, num_frames: int = 100, frames_per_group: int = 10,
                     output_dir: str = None) -> dict:
        """
        Process video file and perform BEFORE/AFTER predictions comparison
        
        Args:
            video_path: Path to video file
            num_frames: Total number of frames to process (default: 100)
            frames_per_group: Number of consecutive frames per group (default: 10)
            output_dir: Directory to save results
            
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = time.time()
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Extract random frame groups
        frame_groups = self.extract_random_frames(video_path, num_frames, frames_per_group)
        
        # Process each group with BEFORE/AFTER comparison
        logger.info(f"🔍 Processing {len(frame_groups)} groups for BEFORE/AFTER comparison...")
        
        frame_groups_with_predictions = []
        base_class_names = list(self.class_names.values())
        before_stats = {name: 0 for name in base_class_names}
        after_stats = {name: 0 for name in base_class_names}
        dynamic_class_order = list(base_class_names)
        total_frames_processed = 0
        
        for group_idx, group_frames in enumerate(frame_groups):
            logger.info(f"Processing group {group_idx + 1}/{len(frame_groups)} ({len(group_frames)} frames)")
            
            group_with_predictions = []
            for frame_idx, (frame_num, frame) in enumerate(group_frames):
                # Get BEFORE/AFTER predictions
                before_preds, before_frame, after_preds, after_frame = self.predict_frame_comparison(frame)
                
                # Update statistics
                before_counts = self._count_predictions_by_class(before_preds)
                after_counts = self._count_predictions_by_class(after_preds)

                for name in before_counts:
                    if name not in dynamic_class_order:
                        dynamic_class_order.append(name)
                for name in after_counts:
                    if name not in dynamic_class_order:
                        dynamic_class_order.append(name)

                for name, count in before_counts.items():
                    before_stats[name] = before_stats.get(name, 0) + count
                for name, count in after_counts.items():
                    after_stats[name] = after_stats.get(name, 0) + count
                
                group_with_predictions.append((frame_num, frame, before_preds, before_frame, after_preds, after_frame))
                total_frames_processed += 1
            
            frame_groups_with_predictions.append(group_with_predictions)
        
        # Calculate statistics
        processing_time = time.time() - start_time
        avg_time_per_frame = processing_time / total_frames_processed if total_frames_processed > 0 else 0
        
        results = {
            'video_path': video_path,
            'total_groups': len(frame_groups),
            'frames_per_group': frames_per_group,
            'total_frames_processed': total_frames_processed,
            'before_predictions': before_stats,
            'after_predictions': after_stats,
            'improvement': {
                name: after_stats.get(name, 0) - before_stats.get(name, 0)
                for name in dynamic_class_order
            },
            'class_names': {class_id: name for class_id, name in self.class_names.items()},
            'processing_time': processing_time,
            'avg_time_per_frame': avg_time_per_frame,
            'frame_groups_with_predictions': frame_groups_with_predictions
        }
        
        # Log statistics
        logger.info("📊 BEFORE/AFTER Processing Statistics:")
        logger.info(f"   Groups processed: {len(frame_groups)}")
        logger.info(f"   Frames per group: {frames_per_group}")
        logger.info(f"   Total frames processed: {total_frames_processed}")
        stats_order = dynamic_class_order
        before_summary = " | ".join(f"{name}: {before_stats.get(name, 0)}" for name in stats_order)
        after_summary = " | ".join(f"{name}: {after_stats.get(name, 0)}" for name in stats_order)
        improvement_summary = " | ".join(
            f"{name}: {results['improvement'].get(name, 0):+d}" for name in stats_order
        )
        logger.info(f"   � BEFORE - {before_summary}")
        logger.info(f"   🟢 AFTER  - {after_summary}")
        logger.info(f"   📈 IMPROVEMENT - {improvement_summary}")
        logger.info(f"   Processing time: {processing_time:.2f} seconds")
        logger.info(f"   Average time per frame: {avg_time_per_frame:.2f} seconds")
        
        # Save results if output directory specified
        if output_dir and frame_groups_with_predictions:
            self.display_and_save_comparison_groups(frame_groups_with_predictions, output_dir)
        
        return results

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='🎯 YOLO Player & Ball Detection Video Predictor')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--frames', type=int, default=100,
                       help='Total number of frames to process (default: 100)')
    parser.add_argument('--frames-per-group', type=int, default=10,
                       help='Number of consecutive frames per group (default: 10)')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Confidence threshold for predictions (default: 0.3)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory to save results (optional)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        logger.error(f"❌ Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.video):
        logger.error(f"❌ Video file not found: {args.video}")
        return
    
    # Create predictor
    predictor = VideoPredictor(args.model, args.confidence)
    
    # Process video
    results = predictor.process_video(
        video_path=args.video,
        num_frames=args.frames,
        frames_per_group=args.frames_per_group,
        output_dir=args.output
    )
    
    logger.info("🎉 Video processing completed!")

if __name__ == "__main__":
    main()