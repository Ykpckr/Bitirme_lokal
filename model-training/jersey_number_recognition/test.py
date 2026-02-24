import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

from config import TestConfig
from dataset import JerseyNumberDatasetWithPath
from model import create_model
from data_loader import prepare_test_dataset
from transforms import get_test_transforms
from train_utils import test_model
from eval_utils import plot_confusion_matrix, save_error_analysis


def main():
    config = TestConfig()
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Jersey Number Recognition - Test Evaluation")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Model: {config.model_path}")
    print("=" * 60)
    
    if not config.model_path.exists():
        print(f"Error: Model not found at {config.model_path}")
        print("Please train the model first using train.py")
        return
    
    if not config.test_dir.exists():
        print(f"Error: Test directory not found at {config.test_dir}")
        print("Please add test data in the following structure:")
        print("test/")
        print("  test_gt.json")
        print("  images/")
        print("    0/")
        print("    10/")
        return
    
    print("\nLoading model checkpoint...")
    checkpoint = torch.load(config.model_path, map_location=config.device)
    idx_to_label = {int(k): v for k, v in checkpoint['idx_to_label'].items()}
    label_to_idx = {v: k for k, v in idx_to_label.items()}
    num_classes = len(idx_to_label)
    model_name = checkpoint['config']['MODEL_NAME']
    
    print(f"Model: {model_name}")
    print(f"Number of classes: {num_classes}")
    print(f"Training validation accuracy: {checkpoint['val_acc']:.4f}")
    print(f"Classes: {sorted(idx_to_label.values())}")
    
    test_image_paths, test_labels = prepare_test_dataset(config, label_to_idx)
    
    if len(test_image_paths) == 0:
        print("\nError: No valid test samples found!")
        return
    
    test_transform = get_test_transforms(config)
    
    test_dataset = JerseyNumberDatasetWithPath(test_image_paths, test_labels, test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, 
        shuffle=False, num_workers=0
    )
    
    print("\nCreating model...")
    model = create_model(num_classes, model_name, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    
    print("\nEvaluating on test set...")
    test_preds, test_labels_eval, test_paths = test_model(model, test_loader, config.device)
    
    test_acc = accuracy_score(test_labels_eval, test_preds)
    
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("=" * 60)
    
    print("\nDetailed Classification Report:")
    unique_test_labels = sorted(set(test_labels_eval) | set(test_preds))
    target_names = [str(idx_to_label[i]) for i in unique_test_labels]
    report = classification_report(test_labels_eval, test_preds, 
                                   labels=unique_test_labels,
                                   target_names=target_names, 
                                   zero_division=0)
    print(report)
    
    with open(config.output_dir / 'classification_report.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Jersey Number Recognition - Test Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n")
        f.write(f"Total test samples: {len(test_labels_eval)}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    
    print(f"\nSaved classification report to {config.output_dir / 'classification_report.txt'}")
    
    plot_confusion_matrix(test_labels_eval, test_preds, idx_to_label, 
                         config.output_dir, 'test_confusion_matrix.png')
    
    errors = save_error_analysis(test_labels_eval, test_preds, test_paths, 
                                 idx_to_label, config.output_dir)
    
    if len(errors) > 0:
        print(f"\nMisclassified samples: {len(errors)}/{len(test_labels_eval)}")
        print("\nTop 5 mistakes:")
        for i, error in enumerate(errors[:5]):
            print(f"  {i+1}. True: {error['true_jersey']} | Predicted: {error['predicted_jersey']} | {Path(error['image_path']).name}")
    else:
        print("\nPerfect! No misclassifications!")
    
    print(f"\nAll test results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
