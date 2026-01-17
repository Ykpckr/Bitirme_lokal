import json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_training_history(history: Dict, output_dir: Path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    if 'train_map' in history:
        ax3.plot(history['train_map'], label='Train mAP')
        ax3.plot(history['val_map'], label='Val mAP')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('mAP')
        ax3.set_title('Training and Validation mAP')
        ax3.legend()
        ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150)
    print(f"Saved training history plot to {output_dir / 'training_history.png'}")
    plt.close()


def plot_confusion_matrix(y_true: List, y_pred: List, idx_to_label: Dict, 
                         output_dir: Path, filename: str = 'confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    
    if len(idx_to_label) <= 30:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[idx_to_label[i] for i in range(len(idx_to_label))],
                   yticklabels=[idx_to_label[i] for i in range(len(idx_to_label))])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150)
        print(f"Saved confusion matrix to {output_dir / filename}")
        plt.close()
    else:
        print(f"Skipping confusion matrix plot (too many classes: {len(idx_to_label)})")


def save_error_analysis(y_true: List, y_pred: List, image_paths: List, 
                       idx_to_label: Dict, output_dir: Path):
    errors = []
    
    for i, (true_label, pred_label, img_path) in enumerate(zip(y_true, y_pred, image_paths)):
        if true_label != pred_label:
            errors.append({
                'image_path': img_path,
                'true_jersey': idx_to_label[true_label],
                'predicted_jersey': idx_to_label[pred_label]
            })
    
    with open(output_dir / 'misclassified_samples.json', 'w') as f:
        json.dump(errors, f, indent=2)
    
    print(f"Saved {len(errors)} misclassified samples to {output_dir / 'misclassified_samples.json'}")
    
    return errors
