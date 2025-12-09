# Player Re-Identification (ReID) Module

Person re-identification module specifically designed for football player tracking. This module extracts discriminative embeddings from player crops and enables robust multi-object tracking by combining motion (IoU) and appearance (ReID) cues.

## Features

- **ResNet50 Backbone**: ImageNet pretrained for strong feature extraction
- **BNNeck Architecture**: Batch normalization neck for improved generalization
- **Combined Loss**: Cross-entropy + Triplet loss with hard mining
- **P×K Sampling**: Identity-balanced batch sampling for effective metric learning
- **256D Embeddings**: L2-normalized feature vectors for efficient comparison
- **mAP & Rank@K Metrics**: Standard ReID evaluation protocols
- **YOLO Integration**: Seamless integration with existing player detector
- **Tracking Cost Matrix**: α·IoU + β·cosine distance for robust association

## Project Structure

```
reid/
├── configs/
│   └── example_config.yaml      # Configuration file
├── datasets/
│   ├── __init__.py
│   └── soccer_reid.py         # Dataset loader with P×K sampling
├── models/
│   ├── __init__.py
│   ├── backbone_resnet50.py   # ResNet50 feature extractor
│   └── head_bnneck.py         # BNNeck head with L2 normalization
├── losses/
│   ├── __init__.py
│   └── triplet.py             # Triplet loss with batch-hard mining
├── engine/
│   ├── __init__.py
│   ├── train.py               # Training engine
│   ├── evaluate.py            # Evaluation engine (mAP, Rank@K)
│   └── export.py              # Model export
├── integration/
│   ├── __init__.py
│   ├── embedder_infer.py      # Real-time embedding extraction
│   └── cost_matrix.py         # IoU + cosine distance cost
├── scripts/
│   ├── __init__.py
│   └── make_crops_from_yolo.py  # Extract player crops from video
└── README.md
```

## Installation

```bash
# Navigate to reid directory
cd /home/airborne/Desktop/eray/FoMAC/model-training/reid

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Extract Player Crops from Video

Extract player bounding boxes from video using the trained YOLO detector:

```bash
# Basic extraction (without track IDs)
python scripts/make_crops_from_yolo.py \
    --video /path/to/match.mp4 \
    --out data/reid/raw_crops \
    --conf 0.5 \
    --frame-interval 10

# With tracking results (organizes by person ID)
python scripts/make_crops_from_yolo.py \
    --video /path/to/match.mp4 \
    --out data/reid \
    --tracks /path/to/tracking_results.txt \
    --min-crops-per-id 10
```

**Parameters:**
- `--video`: Input video path
- `--out`: Output directory
- `--weights`: YOLO weights (default: from config)
- `--conf`: Detection confidence threshold (default: 0.5)
- `--player-class`: Player class ID (default: 0)
- `--frame-interval`: Process every Nth frame (default: 10)
- `--tracks`: Tracking results for organizing by person ID
- `--min-crops-per-id`: Minimum crops per identity (default: 10)

**Expected Output Structure:**
```
data/reid/
├── train/
│   ├── pid_0001/
│   │   ├── frame_000100_det_000001_conf_0.95.jpg
│   │   └── ...
│   ├── pid_0002/
│   └── ...
├── query/
│   └── pid_0001/
│       └── ...
└── gallery/
    └── pid_0001/
        └── ...
```

### 2. Train ReID Model

Train the ReID model with combined CE + Triplet loss:

```bash
python engine/train.py --cfg configs/example_config.yaml
```

**Resume from checkpoint:**
```bash
python engine/train.py \
    --cfg configs/example_config.yaml \
    --resume outputs/reid/checkpoints/latest.pt
```

**Training Configuration** (`configs/example_config.yaml`):
- **Epochs**: 80
- **Batch Size**: 64 (P×K = 16×4)
- **Learning Rate**: 3e-4 with warmup + cosine annealing
- **Loss Weights**: CE=1.0, Triplet=1.0
- **Triplet Margin**: 0.3
- **Image Size**: 256×128

**Outputs:**
- `outputs/reid/checkpoints/best_reid.pt` - Best model (highest mAP)
- `outputs/reid/checkpoints/latest.pt` - Latest checkpoint
- `outputs/reid/logs/` - TensorBoard logs

**Monitor Training:**
```bash
tensorboard --logdir outputs/reid/logs
```

### 3. Evaluate ReID Model

Evaluate trained model on query/gallery splits:

```bash
python engine/evaluate.py --cfg configs/example_config.yaml
```

**With specific checkpoint:**
```bash
python engine/evaluate.py \
    --cfg configs/example_config.yaml \
    --checkpoint outputs/reid/checkpoints/best_reid.pt
```

**Evaluation Metrics:**
- **mAP**: Mean Average Precision
- **Rank-1**: Top-1 accuracy
- **Rank-5**: Top-5 accuracy
- **Rank-10**: Top-10 accuracy

### 4. Export Model for Inference

Export trained model for deployment:

```bash
python engine/export.py --cfg configs/example_config.yaml
```

**With testing:**
```bash
python engine/export.py \
    --cfg configs/example_config.yaml \
    --test
```

**Output:**
- `outputs/reid/checkpoints/best_reid.pt` - Exported model ready for inference

### 5. Integration with Tracking

#### Extract Embeddings

```python
from integration.embedder_infer import ReIDEmbedder

# Initialize embedder
embedder = ReIDEmbedder('outputs/reid/checkpoints/best_reid.pt')

# Extract embedding from crop
embedding = embedder.get_embedding(player_crop_image)
# Returns: (256,) numpy array, L2-normalized

# Extract embedding with bounding box
embedding = embedder.get_embedding(
    full_frame,
    bbox=(x1, y1, x2, y2)
)

# Batch processing
embeddings = embedder.get_embeddings_batch(crop_images)
# Returns: (N, 256) numpy array
```

#### Build Cost Matrix for Tracking

```python
from integration.cost_matrix import build_cost, compute_iou_matrix

# Compute IoU matrix
iou_matrix = compute_iou_matrix(track_boxes, detection_boxes)
# track_boxes: (N, 4) [x1, y1, x2, y2]
# detection_boxes: (M, 4) [x1, y1, x2, y2]

# Build combined cost matrix
cost_matrix = build_cost(
    iou_matrix,
    track_embeddings,   # (N, 256)
    detection_embeddings,  # (M, 256)
    alpha=0.6,  # IoU weight
    beta=0.4,   # Appearance weight
    max_distance=0.7
)
# Returns: (N, M) cost matrix

# Use with Hungarian algorithm for data association
from scipy.optimize import linear_sum_assignment
row_indices, col_indices = linear_sum_assignment(cost_matrix)
```

#### Convenience Function

```python
from integration import get_embedding

# First call (initialize embedder)
embedding = get_embedding(
    image,
    bbox=(x1, y1, x2, y2),
    model_path='outputs/reid/checkpoints/best_reid.pt'
)

# Subsequent calls (reuses embedder)
embedding2 = get_embedding(image2)
```

## Configuration

Edit `configs/example_config.yaml` to customize training:

```yaml
model:
  backbone: resnet50
  emb_dim: 256
  pretrained: true
  last_stride: 1

loss:
  ce_weight: 1.0
  triplet_weight: 1.0
  triplet_margin: 0.3
  label_smooth: 0.1

train:
  epochs: 80
  batch_size: 64
  num_instances: 4  # K in P×K sampling
  lr: 0.0003
  optimizer: adamw
  scheduler: warmup_cosine
  warmup_epochs: 10

data:
  root: data/reid
  height: 256
  width: 128

eval:
  metrics: [mAP, Rank1, Rank5, Rank10]
  max_rank: 50
```

## Performance Expectations

Based on standard ReID benchmarks, you can expect:

- **mAP**: 70-85% (depends on dataset quality)
- **Rank-1**: 85-95%
- **Rank-5**: 95-98%

**Note**: Performance heavily depends on:
- Quality and diversity of training data
- Number of person identities
- Lighting and viewing conditions
- Jersey similarity between teams

## Tips for Best Results

### Data Collection

1. **Diverse Viewpoints**: Collect crops from different camera angles
2. **Balanced Sampling**: Ensure each player has sufficient samples (≥10)
3. **Quality Filtering**: Remove blurry, occluded, or truncated crops
4. **Team Diversity**: Include multiple matches with different jerseys

### Training

1. **Augmentation**: Carefully tune augmentation (avoid excessive flipping)
2. **Learning Rate**: Start with 3e-4, reduce if loss doesn't converge
3. **Warmup**: Use warmup (10 epochs) for stable training
4. **Early Stopping**: Monitor validation mAP, save best model

### Inference

1. **Batch Processing**: Use `get_embeddings_batch()` for efficiency
2. **Cost Weights**: Tune α (IoU) and β (appearance) based on scenario
   - High α: Prioritize motion (fast movement)
   - High β: Prioritize appearance (crowded scenes)
3. **Distance Threshold**: Adjust `max_distance` to control false positives

## Acceptance Criteria

✅ **outputs/best_reid.pt** generated with training metrics logged

✅ **mAP and Rank-1** reported in evaluation results

✅ **integration/embedder_infer.py** provides single-function embedding extraction

✅ **cost_matrix.py** test verifies:
   - IoU=0.5, cos_sim=0.8 → **lower cost** than
   - IoU=0.1, cos_sim=0.2

✅ **Code** follows PEP8 standards

✅ **All paths** managed through configuration files

## Testing

Run module tests:

```bash
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
```

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution**: Reduce batch size in config:
```yaml
train:
  batch_size: 32  # Reduce from 64
  num_instances: 4
```

### Issue: Low mAP

**Possible Causes**:
1. Insufficient training data
2. Poor data quality (blurry crops, occlusions)
3. Imbalanced identities
4. Insufficient training epochs

**Solutions**:
- Collect more diverse data
- Increase training epochs
- Adjust learning rate
- Try different augmentation strategies

### Issue: Slow Training

**Solutions**:
- Enable CUDA: Set `device.use_cuda: true`
- Reduce image size: Try 224×112 instead of 256×128
- Increase `num_workers` for data loading
- Use mixed precision training

## Integration with Existing Tracking Pipeline

To integrate ReID with your tracking module:

1. **Update tracking config** (`model-training/tracking/configs/tracking.yaml`):

```yaml
tracking:
  reid:
    enabled: true
    weights: /path/to/reid/outputs/reid/checkpoints/best_reid.pt
    alpha: 0.6  # IoU weight
    beta: 0.4   # Appearance weight
    max_distance: 0.7
```

2. **Modify tracker** to use ReID embeddings in association step

3. **Extract embeddings** for each detection before association

4. **Use combined cost** instead of pure IoU-based cost

## Citation

If you use this ReID module, please cite relevant works:

```
@article{luo2019bag,
  title={Bag of tricks and a strong baseline for deep person re-identification},
  author={Luo, Hao and Gu, Youzhi and Liao, Xingyu and Lai, Shenqi and Jiang, Wei},
  journal={CVPR Workshops},
  year={2019}
}
```

## License

This module is part of the FoMAC project.

## Contact

For questions or issues, please open an issue on the repository.

---

**Last Updated**: December 9, 2025
