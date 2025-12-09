"""
Evaluation Engine for ReID

Computes mAP, Rank@K metrics for person re-identification.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from datasets.soccer_reid import build_reid_dataloader


def extract_features(model, dataloader, device):
    """
    Extract features from all images.
    
    Args:
        model: ReID model
        dataloader: DataLoader
        device: Compute device
        
    Returns:
        features: (N, emb_dim) numpy array
        pids: (N,) numpy array of person IDs
        camids: (N,) numpy array of camera IDs
    """
    model.eval()
    
    features_list = []
    pids_list = []
    camids_list = []
    
    with torch.no_grad():
        for imgs, pids, camids, _ in tqdm(dataloader, desc="Extracting"):
            imgs = imgs.to(device)
            
            # Extract embeddings
            embeddings = model(imgs, return_logits=False)
            
            features_list.append(embeddings.cpu().numpy())
            pids_list.append(pids.numpy())
            camids_list.append(camids.numpy())
    
    # Concatenate all batches
    features = np.concatenate(features_list, axis=0)
    pids = np.concatenate(pids_list, axis=0)
    camids = np.concatenate(camids_list, axis=0)
    
    return features, pids, camids


def compute_distance_matrix(query_features, gallery_features):
    """
    Compute pairwise cosine distance matrix.
    
    Args:
        query_features: (Nq, D) query embeddings (L2-normalized)
        gallery_features: (Ng, D) gallery embeddings (L2-normalized)
        
    Returns:
        dist_mat: (Nq, Ng) distance matrix
    """
    # Cosine distance = 1 - cosine_similarity
    # Since features are L2-normalized:
    # cosine_sim = query · gallery^T
    # cosine_dist = 1 - cosine_sim
    
    similarity = np.dot(query_features, gallery_features.T)
    dist_mat = 1 - similarity
    
    return dist_mat


def evaluate_rank(dist_mat, query_pids, gallery_pids,
                  query_camids, gallery_camids, max_rank=50):
    """
    Evaluate ranking metrics.
    
    Args:
        dist_mat: (Nq, Ng) distance matrix
        query_pids: (Nq,) query person IDs
        gallery_pids: (Ng,) gallery person IDs
        query_camids: (Nq,) query camera IDs
        gallery_camids: (Ng,) gallery camera IDs
        max_rank: Maximum rank to compute
        
    Returns:
        cmc: Cumulative Matching Characteristics
        mAP: Mean Average Precision
    """
    num_queries = dist_mat.shape[0]
    num_gallery = dist_mat.shape[1]
    
    if num_gallery < max_rank:
        max_rank = num_gallery
    
    # Sort gallery indices by distance
    indices = np.argsort(dist_mat, axis=1)
    
    # Initialize CMC and AP
    cmc = np.zeros(max_rank)
    all_AP = []
    all_INP = []
    
    for q_idx in range(num_queries):
        # Get query info
        q_pid = query_pids[q_idx]
        q_camid = query_camids[q_idx]
        
        # Sort gallery by distance
        order = indices[q_idx]
        
        # Remove gallery samples from same camera (if applicable)
        # For soccer, we typically have single camera, so this step
        # can be simplified
        remove = (gallery_pids[order] == q_pid) & \
                 (gallery_camids[order] == q_camid)
        keep = np.invert(remove)
        
        # Get match array
        orig_cmc = (gallery_pids[order] == q_pid)
        
        if not np.any(orig_cmc):
            # No valid matches for this query
            continue
        
        # Apply same-camera removal
        cmc_match = orig_cmc[keep]
        
        # Compute CMC
        if cmc_match.size == 0:
            # No gallery samples after filtering
            continue
        
        # Find first match
        first_match_idx = np.where(cmc_match)[0]
        
        if len(first_match_idx) == 0:
            continue
        
        first_match = first_match_idx[0]
        
        # Update CMC
        for k in range(first_match, max_rank):
            cmc[k] += 1
        
        # Compute Average Precision
        num_rel = np.sum(cmc_match)
        
        if num_rel == 0:
            continue
        
        mask = np.zeros(cmc_match.shape)
        mask[cmc_match] = 1
        
        # Compute precision at each position
        precision = np.cumsum(mask) / (np.arange(len(mask)) + 1)
        
        # Average precision
        AP = np.sum(precision[cmc_match]) / num_rel
        all_AP.append(AP)
    
    if len(all_AP) == 0:
        print("Warning: No valid queries found!")
        return np.zeros(max_rank), 0.0
    
    # Compute mAP
    mAP = np.mean(all_AP)
    
    # Normalize CMC
    cmc = cmc / num_queries
    
    return cmc, mAP


def evaluate(cfg, model=None, device=None):
    """
    Evaluate ReID model.
    
    Args:
        cfg: Configuration dictionary
        model: Optional model (if None, load from checkpoint)
        device: Compute device
        
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    # Setup device
    if device is None:
        if cfg['device']['use_cuda'] and torch.cuda.is_available():
            device = torch.device(f"cuda:{cfg['device']['gpu_ids'][0]}")
        else:
            device = torch.device('cpu')
    
    # Load model if not provided
    if model is None:
        from models.head_bnneck import build_reid_model
        
        # Load checkpoint
        checkpoint_path = Path(cfg['export']['path'])
        
        if not checkpoint_path.exists():
            # Try latest checkpoint
            checkpoint_path = (
                Path(cfg['paths']['output_root']) /
                'checkpoints' / 'latest.pt'
            )
        
        if not checkpoint_path.exists():
            print(f"Error: No checkpoint found at {checkpoint_path}")
            return None
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get number of classes from checkpoint config
        ckpt_cfg = checkpoint.get('cfg', cfg)
        
        # Build dataloaders to get num_classes
        train_loader = build_reid_dataloader(
            ckpt_cfg, split='train', is_train=False
        )
        num_classes = train_loader.dataset.num_pids
        
        # Build model
        model = build_reid_model(
            num_classes=num_classes,
            emb_dim=ckpt_cfg['model']['emb_dim'],
            pretrained=False,
            last_stride=ckpt_cfg['model']['last_stride']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
    
    model.eval()
    
    # Build dataloaders
    print("Loading query dataset...")
    query_loader = build_reid_dataloader(
        cfg, split='query', is_train=False
    )
    
    print("Loading gallery dataset...")
    gallery_loader = build_reid_dataloader(
        cfg, split='gallery', is_train=False
    )
    
    # Extract features
    print("Extracting query features...")
    query_features, query_pids, query_camids = extract_features(
        model, query_loader, device
    )
    
    print("Extracting gallery features...")
    gallery_features, gallery_pids, gallery_camids = extract_features(
        model, gallery_loader, device
    )
    
    print(f"Query: {len(query_pids)} samples")
    print(f"Gallery: {len(gallery_pids)} samples")
    
    # Compute distance matrix
    print("Computing distance matrix...")
    dist_mat = compute_distance_matrix(query_features, gallery_features)
    
    # Evaluate
    print("Computing metrics...")
    cmc, mAP = evaluate_rank(
        dist_mat,
        query_pids,
        gallery_pids,
        query_camids,
        gallery_camids,
        max_rank=cfg['eval']['max_rank']
    )
    
    # Collect metrics
    metrics = {
        'mAP': mAP,
        'Rank1': cmc[0],
        'Rank5': cmc[4] if len(cmc) > 4 else 0.0,
        'Rank10': cmc[9] if len(cmc) > 9 else 0.0,
    }
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(f"mAP:     {metrics['mAP']:.4f}")
    print(f"Rank-1:  {metrics['Rank1']:.4f}")
    print(f"Rank-5:  {metrics['Rank5']:.4f}")
    print(f"Rank-10: {metrics['Rank10']:.4f}")
    print("="*50)
    
    return metrics


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate ReID Model')
    parser.add_argument(
        '--cfg',
        type=str,
        default='configs/example_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint'
    )
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent.parent / args.cfg
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Override checkpoint path if provided
    if args.checkpoint is not None:
        cfg['export']['path'] = args.checkpoint
    
    # Run evaluation
    metrics = evaluate(cfg)
    
    if metrics is None:
        print("Evaluation failed!")
        return
    
    # Save results
    output_dir = Path(cfg['paths']['output_root'])
    results_file = output_dir / 'eval_results.txt'
    
    with open(results_file, 'w') as f:
        f.write("ReID Evaluation Results\n")
        f.write("=" * 50 + "\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value:.4f}\n")
    
    print(f"\n✓ Results saved to {results_file}")


if __name__ == "__main__":
    main()
