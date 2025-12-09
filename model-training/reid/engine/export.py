"""
Export Engine for ReID

Exports trained ReID model for inference.
"""

import argparse
from pathlib import Path

import torch
import yaml

import sys
sys.path.append(str(Path(__file__).parent.parent))


def export_model(cfg, checkpoint_path=None):
    """
    Export ReID model.
    
    Args:
        cfg: Configuration dictionary
        checkpoint_path: Optional checkpoint path to export
        
    Returns:
        export_path: Path to exported model
    """
    from models.head_bnneck import build_reid_model
    from datasets.soccer_reid import build_reid_dataloader
    
    # Determine checkpoint path
    if checkpoint_path is None:
        checkpoint_path = (
            Path(cfg['paths']['output_root']) /
            'checkpoints' / 'best_reid.pt'
        )
    else:
        checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return None
    
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get config from checkpoint
    ckpt_cfg = checkpoint.get('cfg', cfg)
    
    # Build dataloader to get num_classes
    train_loader = build_reid_dataloader(
        ckpt_cfg, split='train', is_train=False
    )
    num_classes = train_loader.dataset.num_pids
    
    print(f"Building model with {num_classes} classes...")
    
    # Build model
    model = build_reid_model(
        num_classes=num_classes,
        emb_dim=ckpt_cfg['model']['emb_dim'],
        pretrained=False,
        last_stride=ckpt_cfg['model']['last_stride']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Export path
    export_path = Path(cfg['export']['path'])
    export_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create export package
    export_data = {
        'model_state_dict': model.state_dict(),
        'cfg': ckpt_cfg,
        'num_classes': num_classes,
        'emb_dim': ckpt_cfg['model']['emb_dim'],
        'epoch': checkpoint.get('epoch', 0),
        'best_mAP': checkpoint.get('best_mAP', 0.0),
    }
    
    # Save
    torch.save(export_data, export_path)
    print(f"✓ Model exported to {export_path}")
    
    # Print model info
    print("\nModel Information:")
    print(f"  Classes: {num_classes}")
    print(f"  Embedding dim: {ckpt_cfg['model']['emb_dim']}")
    print(f"  Epoch: {export_data['epoch']}")
    print(f"  Best mAP: {export_data['best_mAP']:.4f}")
    
    return export_path


def load_exported_model(export_path, device='cpu'):
    """
    Load exported ReID model for inference.
    
    Args:
        export_path: Path to exported model
        device: Device to load model on
        
    Returns:
        model: Loaded model in eval mode
        cfg: Model configuration
    """
    from models.head_bnneck import build_reid_model
    
    export_path = Path(export_path)
    
    if not export_path.exists():
        raise FileNotFoundError(f"Exported model not found: {export_path}")
    
    # Load export package
    export_data = torch.load(export_path, map_location=device)
    
    # Build model
    model = build_reid_model(
        num_classes=export_data['num_classes'],
        emb_dim=export_data['emb_dim'],
        pretrained=False,
        last_stride=export_data['cfg']['model']['last_stride']
    )
    
    # Load weights
    model.load_state_dict(export_data['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, export_data['cfg']


def test_exported_model(export_path):
    """Test exported model."""
    print(f"Testing exported model: {export_path}")
    
    # Load model
    model, cfg = load_exported_model(export_path, device='cpu')
    
    print("✓ Model loaded successfully")
    
    # Test forward pass
    import torch
    
    batch_size = 2
    height = cfg['data']['height']
    width = cfg['data']['width']
    
    x = torch.randn(batch_size, 3, height, width)
    
    with torch.no_grad():
        embeddings = model(x, return_logits=False)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Embedding norm: {torch.norm(embeddings, p=2, dim=1)}")
    
    assert embeddings.shape == (batch_size, cfg['model']['emb_dim'])
    
    print("✓ All tests passed!")


def main():
    """Main export script."""
    parser = argparse.ArgumentParser(description='Export ReID Model')
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
        help='Path to checkpoint to export (default: best_reid.pt)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test exported model after export'
    )
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent.parent / args.cfg
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Export model
    export_path = export_model(cfg, checkpoint_path=args.checkpoint)
    
    if export_path is None:
        print("Export failed!")
        return
    
    # Test if requested
    if args.test:
        print("\n" + "="*50)
        test_exported_model(export_path)


if __name__ == "__main__":
    main()
