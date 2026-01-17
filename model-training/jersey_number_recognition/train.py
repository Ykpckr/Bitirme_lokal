import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score
from sklearn.preprocessing import label_binarize

from config import TrainingConfig
from dataset import JerseyNumberDataset
from model import create_model
from data_loader import prepare_dataset
from transforms import get_train_transforms, get_val_transforms
from train_utils import train_epoch, validate
from eval_utils import plot_training_history, plot_confusion_matrix


def main():
    config = TrainingConfig()
    
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    
    config.output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Jersey Number Recognition - Training")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Model: {config.model_name}")
    print(f"Max samples per class: {config.max_samples_per_class}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.num_epochs}")
    print("=" * 60)
    
    print("\nPreparing training dataset...")
    train_paths, train_labels_raw, idx_to_label = prepare_dataset(
        config.train_images_dir, config.train_gt_file, config
    )
    num_classes = len(idx_to_label)
    
    label_to_idx = {label: idx for idx, label in idx_to_label.items()}
    train_labels = [label for label in train_labels_raw]
    
    with open(config.output_dir / 'label_mapping.json', 'w') as f:
        json.dump({str(k): v for k, v in idx_to_label.items()}, f, indent=2)
    print(f"Saved label mapping to {config.output_dir / 'label_mapping.json'}")
    
    if config.use_test_as_val:
        print("\nPreparing validation dataset from test set...")
        valid_jersey_numbers = set(idx_to_label.values())
        val_paths, val_labels_raw, _ = prepare_dataset(
            config.test_images_dir, config.test_gt_file, config,
            valid_classes=valid_jersey_numbers,
            max_samples_override=config.max_val_samples_per_class
        )
        val_labels = [label for label in val_labels_raw]
    else:
        print("\nSplitting train set for validation...")
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels, test_size=config.test_size, 
            random_state=config.random_seed, stratify=train_labels
        )
    
    print(f"\nDataset split:")
    print(f"  Training samples: {len(train_paths)}")
    print(f"  Validation samples: {len(val_paths)}")
    
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)
    
    train_dataset = JerseyNumberDataset(train_paths, train_labels, train_transform)
    val_dataset = JerseyNumberDataset(val_paths, val_labels, val_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, 
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, 
        shuffle=False, num_workers=0
    )
    
    print(f"\nCreating {config.model_name} model...")
    model = create_model(num_classes, config.model_name, config.pretrained)
    model = model.to(config.device)
    
    class_counts = [train_labels.count(i) for i in range(num_classes)]
    total_samples = len(train_labels)
    class_weights = torch.FloatTensor([total_samples / (num_classes * count) if count > 0 else 1.0 
                                       for count in class_counts])
    class_weights = class_weights.to(config.device)
    print(f"\nClass weights applied (first 10): {class_weights[:10].tolist()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, 
                           weight_decay=config.weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_map': [],
        'val_loss': [],
        'val_acc': [],
        'val_map': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\nStarting training...")
    print("=" * 60)
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print("-" * 40)
        
        train_loss, train_acc, train_probs, train_labels_epoch = train_epoch(
            model, train_loader, criterion, optimizer, config.device
        )
        
        val_loss, val_acc, val_preds, val_labels_epoch, val_probs = validate(
            model, val_loader, criterion, config.device
        )
        
        # Calculate mAP
        try:
            train_labels_bin = label_binarize(train_labels_epoch, classes=range(num_classes))
            val_labels_bin = label_binarize(val_labels_epoch, classes=range(num_classes))
            
            # Handle binary case specifically if num_classes == 2 (though unlikely for jersey numbers)
            if num_classes == 2:
                # label_binarize returns (n_samples, 1) for binary, but probs might be (n_samples, 2)
                # average_precision_score expects (n_samples,) if y_true is (n_samples, 1) or vector
                # But we treat it as macro average over classes normally.
                # For safety in multiclass setting, use the matrix form.
                if train_labels_bin.shape[1] == 1:
                    train_labels_bin = np.hstack((1 - train_labels_bin, train_labels_bin))
                if val_labels_bin.shape[1] == 1:
                    val_labels_bin = np.hstack((1 - val_labels_bin, val_labels_bin))

            train_map = average_precision_score(train_labels_bin, train_probs, average="macro")
            val_map = average_precision_score(val_labels_bin, val_probs, average="macro")
        except Exception as e:
            print(f"Warning: Could not calculate mAP: {e}")
            train_map = 0.0
            val_map = 0.0
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_map'].append(train_map)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_map'].append(val_map)
        
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train mAP: {train_map:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val mAP:   {val_map:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            config_dict = {
                'MAX_SAMPLES_PER_CLASS': config.max_samples_per_class,
                'MIN_SAMPLES_PER_CLASS': config.min_samples_per_class,
                'BATCH_SIZE': config.batch_size,
                'NUM_EPOCHS': config.num_epochs,
                'LEARNING_RATE': config.learning_rate,
                'MODEL_NAME': config.model_name,
                'USE_ORIGINAL_SIZE': config.use_original_size,
            }
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'idx_to_label': idx_to_label,
                'config': config_dict
            }, config.output_dir / 'best_model.pth')
            print(f"  Saved best model (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  Early stopping patience: {patience_counter}/{config.early_stopping_patience}")
        
        if patience_counter >= config.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("=" * 60)
    
    plot_training_history(history, config.output_dir)
    
    checkpoint = torch.load(config.output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nFinal evaluation on validation set:")
    val_loss, val_acc, val_preds, val_labels_eval, _ = validate(
        model, val_loader, criterion, config.device
    )
    
    print(f"  Final Val Loss: {val_loss:.4f}")
    print(f"  Final Val Acc:  {val_acc:.4f}")
    
    print("\nClassification Report:")
    target_names = [str(idx_to_label[i]) for i in range(num_classes)]
    unique_labels = sorted(set(val_labels_eval) | set(val_preds))
    labels = list(range(len(unique_labels)))
    target_names_filtered = [str(idx_to_label[i]) for i in unique_labels if i in idx_to_label]
    print(classification_report(val_labels_eval, val_preds, labels=labels, target_names=target_names_filtered, zero_division=0))
    
    plot_confusion_matrix(val_labels_eval, val_preds, idx_to_label, config.output_dir)
    
    print(f"\nAll outputs saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
