# ReID Module - Quick Reference

## Installation & Verification

```bash
cd /home/airborne/Desktop/eray/FoMAC/model-training/reid
pip install -r requirements.txt
python verify_installation.py
```

## Quick Start Commands

### 1. Extract Crops
```bash
# With tracking (recommended)
python scripts/make_crops_from_yolo.py \
    --video /path/to/match.mp4 \
    --out data/reid \
    --tracks tracking_results.txt

# Without tracking
python scripts/make_crops_from_yolo.py \
    --video /path/to/match.mp4 \
    --out data/reid/raw_crops
```

### 2. Train
```bash
python engine/train.py --cfg configs/example_config.yaml
```

### 3. Evaluate
```bash
python engine/evaluate.py --cfg configs/example_config.yaml
```

### 4. Export
```bash
python engine/export.py --cfg configs/example_config.yaml
```

## Python API

### Extract Embedding
```python
from integration.embedder_infer import ReIDEmbedder

embedder = ReIDEmbedder('outputs/reid/checkpoints/best_reid.pt')
embedding = embedder.get_embedding(player_crop)  # (256,)
```

### Build Cost Matrix
```python
from integration.cost_matrix import build_cost, compute_iou_matrix

iou_matrix = compute_iou_matrix(track_boxes, det_boxes)
cost = build_cost(iou_matrix, track_embs, det_embs, alpha=0.6, beta=0.4)
```

## Key Files

- **Config**: `configs/example_config.yaml`
- **Train**: `engine/train.py`
- **Evaluate**: `engine/evaluate.py`
- **Export**: `engine/export.py`
- **Crops**: `scripts/make_crops_from_yolo.py`
- **Embedder**: `integration/embedder_infer.py`
- **Cost**: `integration/cost_matrix.py`

## Model Architecture

```
Input (3, 256, 128)
    ↓
ResNet50 Backbone (pretrained)
    ↓
Global Average Pooling
    ↓
Bottleneck (2048 → 256)
    ↓
Batch Normalization
    ↓
L2 Normalization → Embedding (256,)
    ↓
Classifier → Logits (num_classes)
```

## Loss Function

```
Total Loss = CE Loss + Triplet Loss
           = 1.0 × CrossEntropy + 1.0 × TripletHardMining
```

## Cost Function

```
Cost = α × (1 - IoU) + β × (1 - cosine_similarity)
     = 0.6 × IoU_cost + 0.4 × appearance_cost
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 80 |
| Batch Size | 64 (P×K = 16×4) |
| Learning Rate | 3e-4 |
| Optimizer | AdamW |
| Scheduler | Warmup + Cosine |
| Embedding Dim | 256 |
| Image Size | 256×128 |
| Triplet Margin | 0.3 |

## Evaluation Metrics

- **mAP**: Mean Average Precision
- **Rank-1**: Top-1 accuracy
- **Rank-5**: Top-5 accuracy
- **Rank-10**: Top-10 accuracy

## Directory Structure

```
data/reid/
├── train/      # Training data
│   └── pid_XXXX/
├── query/      # Query data (10%)
│   └── pid_XXXX/
└── gallery/    # Gallery data (20%)
    └── pid_XXXX/

outputs/reid/
├── checkpoints/
│   ├── best_reid.pt      # Best model
│   └── latest.pt         # Latest checkpoint
└── logs/                 # TensorBoard logs
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of Memory | Reduce batch_size to 32 or 16 |
| Low mAP | Increase epochs, check data quality |
| Slow training | Enable CUDA, increase num_workers |
| Import errors | Run `pip install -r requirements.txt` |

## Testing

```bash
# Test individual modules
python models/backbone_resnet50.py
python models/head_bnneck.py
python losses/triplet.py
python integration/cost_matrix.py

# Verify installation
python verify_installation.py

# View examples
python examples.py
```

## Integration with Tracking

Update `model-training/tracking/configs/tracking.yaml`:

```yaml
tracking:
  reid:
    enabled: true
    weights: /path/to/reid/outputs/reid/checkpoints/best_reid.pt
    alpha: 0.6    # IoU weight
    beta: 0.4     # Appearance weight
    max_distance: 0.7
```

## Performance Expectations

| Metric | Expected Range |
|--------|----------------|
| mAP | 70-85% |
| Rank-1 | 85-95% |
| Rank-5 | 95-98% |

## Useful Links

- **README**: Full documentation
- **IMPLEMENTATION.md**: Implementation details
- **examples.py**: Usage examples
- **configs/example_config.yaml**: Configuration reference

## Support

For issues or questions:
1. Check README.md
2. Run verify_installation.py
3. Review IMPLEMENTATION.md
4. Check examples.py

---

**Last Updated**: December 9, 2025
