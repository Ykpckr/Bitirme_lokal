"""
Training Engine for ReID

Handles model training with CE + Triplet loss.
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from datasets.soccer_reid import build_reid_dataloader
from models.head_bnneck import build_reid_model
from losses.triplet import CombinedLoss


class ReIDTrainer:
    """ReID model trainer."""
    
    def __init__(self, cfg: dict):
        """
        Args:
            cfg: Configuration dictionary
        """
        self.cfg = cfg
        
        # Setup device
        self.device = self._setup_device()
        
        # Build model
        self.model = self._build_model()
        
        # Build loss
        self.criterion = self._build_loss()
        
        # Build optimizer
        self.optimizer = self._build_optimizer()
        
        # Build scheduler
        self.scheduler = self._build_scheduler()
        
        # Setup logging
        self.writer = None
        self.output_dir = Path(cfg['paths']['output_root'])
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if cfg['logging']['tensorboard']:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Training state
        self.current_epoch = 0
        self.best_mAP = 0.0
        self.global_step = 0
    
    def _setup_device(self):
        """Setup compute device."""
        if self.cfg['device']['use_cuda'] and torch.cuda.is_available():
            device = torch.device(
                f"cuda:{self.cfg['device']['gpu_ids'][0]}"
            )
            print(f"Using device: {device}")
        else:
            device = torch.device('cpu')
            print("Using device: CPU")
        return device
    
    def _build_model(self):
        """Build ReID model."""
        # Count number of classes from training data
        train_loader = build_reid_dataloader(
            self.cfg, split='train', is_train=True
        )
        num_classes = train_loader.dataset.num_pids
        
        print(f"Building model with {num_classes} classes...")
        
        model = build_reid_model(
            num_classes=num_classes,
            emb_dim=self.cfg['model']['emb_dim'],
            pretrained=self.cfg['model']['pretrained'],
            last_stride=self.cfg['model']['last_stride']
        )
        
        model = model.to(self.device)
        
        # Multi-GPU support
        if (self.cfg['device']['use_cuda'] and
            len(self.cfg['device']['gpu_ids']) > 1):
            model = nn.DataParallel(
                model, device_ids=self.cfg['device']['gpu_ids']
            )
            print(f"Using {len(self.cfg['device']['gpu_ids'])} GPUs")
        
        return model
    
    def _build_loss(self):
        """Build combined loss."""
        # Get number of classes
        train_loader = build_reid_dataloader(
            self.cfg, split='train', is_train=True
        )
        num_classes = train_loader.dataset.num_pids
        
        criterion = CombinedLoss(
            num_classes=num_classes,
            triplet_margin=self.cfg['loss']['triplet_margin'],
            ce_weight=self.cfg['loss']['ce_weight'],
            triplet_weight=self.cfg['loss']['triplet_weight'],
            label_smooth=self.cfg['loss']['label_smooth']
        )
        
        return criterion.to(self.device)
    
    def _build_optimizer(self):
        """Build optimizer."""
        train_cfg = self.cfg['train']
        
        if train_cfg['optimizer'].lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=train_cfg['lr'],
                weight_decay=train_cfg['weight_decay']
            )
        elif train_cfg['optimizer'].lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=train_cfg['lr'],
                weight_decay=train_cfg['weight_decay']
            )
        elif train_cfg['optimizer'].lower() == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=train_cfg['lr'],
                momentum=0.9,
                weight_decay=train_cfg['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {train_cfg['optimizer']}")
        
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        train_cfg = self.cfg['train']
        
        if train_cfg['scheduler'] == 'warmup_cosine':
            # Warmup + Cosine annealing
            def lr_lambda(epoch):
                warmup_epochs = train_cfg['warmup_epochs']
                total_epochs = train_cfg['epochs']
                warmup_factor = train_cfg['warmup_factor']
                
                if epoch < warmup_epochs:
                    # Linear warmup
                    alpha = epoch / warmup_epochs
                    return warmup_factor * (1 - alpha) + alpha
                else:
                    # Cosine annealing
                    progress = (epoch - warmup_epochs) / (
                        total_epochs - warmup_epochs
                    )
                    return 0.5 * (1 + torch.cos(torch.tensor(
                        progress * 3.14159
                    )))
            
            scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lr_lambda
            )
        elif train_cfg['scheduler'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        
        losses = []
        ce_losses = []
        triplet_losses = []
        
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{self.cfg['train']['epochs']}"
        )
        
        for batch_idx, (imgs, pids, _, _) in enumerate(pbar):
            # Move to device
            imgs = imgs.to(self.device)
            pids = pids.to(self.device)
            
            # Forward pass
            embeddings, logits = self.model(imgs, return_logits=True)
            
            # Compute loss
            loss, loss_dict = self.criterion(embeddings, logits, pids)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.cfg['train']['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg['train']['grad_clip']
                )
            
            self.optimizer.step()
            
            # Record losses
            losses.append(loss_dict['total_loss'])
            ce_losses.append(loss_dict['ce_loss'])
            triplet_losses.append(loss_dict['triplet_loss'])
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'ce': f"{loss_dict['ce_loss']:.4f}",
                'tri': f"{loss_dict['triplet_loss']:.4f}"
            })
            
            # Logging
            if (batch_idx % self.cfg['logging']['print_freq'] == 0 and
                self.writer is not None):
                self.writer.add_scalar(
                    'train/total_loss',
                    loss_dict['total_loss'],
                    self.global_step
                )
                self.writer.add_scalar(
                    'train/ce_loss',
                    loss_dict['ce_loss'],
                    self.global_step
                )
                self.writer.add_scalar(
                    'train/triplet_loss',
                    loss_dict['triplet_loss'],
                    self.global_step
                )
                self.writer.add_scalar(
                    'train/lr',
                    self.optimizer.param_groups[0]['lr'],
                    self.global_step
                )
            
            self.global_step += 1
        
        # Epoch statistics
        avg_loss = sum(losses) / len(losses)
        avg_ce = sum(ce_losses) / len(ce_losses)
        avg_tri = sum(triplet_losses) / len(triplet_losses)
        
        return {
            'loss': avg_loss,
            'ce_loss': avg_ce,
            'triplet_loss': avg_tri
        }
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_mAP': self.best_mAP,
            'cfg': self.cfg
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_reid.pt'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model to {best_path}")
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        print(f"Total epochs: {self.cfg['train']['epochs']}")
        print(f"Output directory: {self.output_dir}")
        
        # Build dataloaders
        train_loader = build_reid_dataloader(
            self.cfg, split='train', is_train=True
        )
        
        # Training loop
        for epoch in range(1, self.cfg['train']['epochs'] + 1):
            self.current_epoch = epoch
            
            # Train one epoch
            train_stats = self.train_epoch(train_loader, epoch)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Loss: {train_stats['loss']:.4f}")
            print(f"  CE Loss: {train_stats['ce_loss']:.4f}")
            print(f"  Triplet Loss: {train_stats['triplet_loss']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            if epoch % self.cfg['train']['save_period'] == 0:
                self.save_checkpoint(epoch)
            
            # Evaluate
            if epoch % self.cfg['train']['eval_period'] == 0:
                print("\nRunning evaluation...")
                # Import here to avoid circular dependency
                from .evaluate import evaluate
                
                metrics = evaluate(self.cfg, self.model, self.device)
                
                if metrics is not None:
                    mAP = metrics['mAP']
                    rank1 = metrics['Rank1']
                    
                    print(f"  mAP: {mAP:.4f}")
                    print(f"  Rank-1: {rank1:.4f}")
                    
                    # Log to tensorboard
                    if self.writer is not None:
                        self.writer.add_scalar('eval/mAP', mAP, epoch)
                        self.writer.add_scalar('eval/Rank1', rank1, epoch)
                    
                    # Save best model
                    if mAP > self.best_mAP:
                        self.best_mAP = mAP
                        self.save_checkpoint(epoch, is_best=True)
                        print(f"  ✓ New best mAP: {mAP:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint(self.cfg['train']['epochs'])
        
        print("\n✓ Training completed!")
        print(f"Best mAP: {self.best_mAP:.4f}")
        
        if self.writer is not None:
            self.writer.close()


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train ReID Model')
    parser.add_argument(
        '--cfg',
        type=str,
        default='configs/example_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent.parent / args.cfg
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print("Configuration:")
    print(yaml.dump(cfg, default_flow_style=False))
    
    # Create trainer
    trainer = ReIDTrainer(cfg)
    
    # Resume from checkpoint if specified
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and trainer.scheduler:
            trainer.scheduler.load_state_dict(
                checkpoint['scheduler_state_dict']
            )
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_mAP = checkpoint.get('best_mAP', 0.0)
        print(f"Resumed from epoch {trainer.current_epoch}")
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
