"""
Quick Start Example for ReID Module

This script demonstrates the basic workflow for using the ReID module.
"""

import sys
from pathlib import Path

# Add reid module to path
reid_root = Path(__file__).parent
sys.path.insert(0, str(reid_root))


def example_extract_crops():
    """Example: Extract player crops from video."""
    print("\n" + "="*60)
    print("Example 1: Extract Player Crops")
    print("="*60)
    
    print("""
# Extract player crops from video
python scripts/make_crops_from_yolo.py \\
    --video /path/to/match.mp4 \\
    --out data/reid/raw_crops \\
    --conf 0.5 \\
    --frame-interval 10

# With tracking results (organizes by person ID)
python scripts/make_crops_from_yolo.py \\
    --video /path/to/match.mp4 \\
    --out data/reid \\
    --tracks /path/to/tracking_results.txt \\
    --min-crops-per-id 10
    """)


def example_train():
    """Example: Train ReID model."""
    print("\n" + "="*60)
    print("Example 2: Train ReID Model")
    print("="*60)
    
    print("""
# Train from scratch
python engine/train.py --cfg configs/example_config.yaml

# Resume from checkpoint
python engine/train.py \\
    --cfg configs/example_config.yaml \\
    --resume outputs/reid/checkpoints/latest.pt

# Monitor training
tensorboard --logdir outputs/reid/logs
    """)


def example_evaluate():
    """Example: Evaluate ReID model."""
    print("\n" + "="*60)
    print("Example 3: Evaluate ReID Model")
    print("="*60)
    
    print("""
# Evaluate best model
python engine/evaluate.py --cfg configs/example_config.yaml

# Evaluate specific checkpoint
python engine/evaluate.py \\
    --cfg configs/example_config.yaml \\
    --checkpoint outputs/reid/checkpoints/epoch_50.pt
    """)


def example_export():
    """Example: Export model."""
    print("\n" + "="*60)
    print("Example 4: Export Model for Inference")
    print("="*60)
    
    print("""
# Export best model
python engine/export.py --cfg configs/example_config.yaml

# Export and test
python engine/export.py --cfg configs/example_config.yaml --test
    """)


def example_inference():
    """Example: Use ReID for inference."""
    print("\n" + "="*60)
    print("Example 5: ReID Inference")
    print("="*60)
    
    print("""
from integration.embedder_infer import ReIDEmbedder
import numpy as np

# Initialize embedder
embedder = ReIDEmbedder('outputs/reid/checkpoints/best_reid.pt')

# Extract embedding from crop
player_crop = ...  # Your player crop image (numpy array)
embedding = embedder.get_embedding(player_crop)
print(f"Embedding shape: {embedding.shape}")  # (256,)

# Extract with bounding box
full_frame = ...  # Full frame image
bbox = (x1, y1, x2, y2)  # Bounding box coordinates
embedding = embedder.get_embedding(full_frame, bbox=bbox)

# Batch processing
crops = [crop1, crop2, crop3, ...]
embeddings = embedder.get_embeddings_batch(crops)
print(f"Batch embeddings shape: {embeddings.shape}")  # (N, 256)
    """)


def example_tracking_integration():
    """Example: Integrate with tracking."""
    print("\n" + "="*60)
    print("Example 6: Tracking Integration")
    print("="*60)
    
    print("""
from integration.cost_matrix import build_cost, compute_iou_matrix
from integration.embedder_infer import ReIDEmbedder
from scipy.optimize import linear_sum_assignment

# Initialize embedder
embedder = ReIDEmbedder('outputs/reid/checkpoints/best_reid.pt')

# Get tracks and detections
track_boxes = np.array([[x1, y1, x2, y2], ...])  # (N, 4)
detection_boxes = np.array([[x1, y1, x2, y2], ...])  # (M, 4)

# Extract embeddings
track_embeddings = embedder.get_embeddings_batch(
    [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in track_boxes]
)  # (N, 256)

detection_embeddings = embedder.get_embeddings_batch(
    [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in detection_boxes]
)  # (M, 256)

# Compute IoU matrix
iou_matrix = compute_iou_matrix(track_boxes, detection_boxes)

# Build combined cost matrix
cost_matrix = build_cost(
    iou_matrix,
    track_embeddings,
    detection_embeddings,
    alpha=0.6,  # IoU weight
    beta=0.4,   # Appearance weight
    max_distance=0.7
)

# Solve assignment problem
row_indices, col_indices = linear_sum_assignment(cost_matrix)

# Process matches
for track_idx, det_idx in zip(row_indices, col_indices):
    if cost_matrix[track_idx, det_idx] < 0.7:
        print(f"Track {track_idx} matched with detection {det_idx}")
        # Update track...
    """)


def example_test_modules():
    """Example: Test individual modules."""
    print("\n" + "="*60)
    print("Example 7: Test Modules")
    print("="*60)
    
    print("""
# Test backbone
python models/backbone_resnet50.py

# Test BNNeck head
python models/head_bnneck.py

# Test triplet loss
python losses/triplet.py

# Test cost matrix
python integration/cost_matrix.py

# Test embedder
python integration/embedder_infer.py
    """)


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("ReID Module - Quick Start Examples")
    print("="*60)
    print("\nThis script shows example usage of the ReID module.")
    print("For detailed documentation, see README.md")
    
    example_extract_crops()
    example_train()
    example_evaluate()
    example_export()
    example_inference()
    example_tracking_integration()
    example_test_modules()
    
    print("\n" + "="*60)
    print("For more information:")
    print("  - Read README.md")
    print("  - Check configs/example_config.yaml")
    print("  - Run individual test scripts")
    print("="*60)
    print()


if __name__ == "__main__":
    main()
