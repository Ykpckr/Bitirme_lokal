import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from config import TrainingConfig, TestConfig


def load_ground_truth(gt_file: Path) -> Dict[str, int]:
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    return gt_data


def prepare_dataset(
    images_dir: Path, 
    gt_file: Path, 
    config: TrainingConfig, 
    valid_classes: Optional[set] = None, 
    max_samples_override: Optional[int] = None
) -> Tuple[List[str], List[int], Dict[int, str]]:
    print(f"Loading ground truth from {gt_file}...")
    gt_data = load_ground_truth(gt_file)
    
    class_to_samples = {}
    no_number_samples = []
    
    for dir_id, jersey_number in gt_data.items():
        if jersey_number == -1:
            if not config.include_no_number_class:
                continue
            dir_path = images_dir / str(dir_id)
            if dir_path.exists() and dir_path.is_dir():
                for img_file in dir_path.glob("*.jpg"):
                    no_number_samples.append((dir_id, str(img_file)))
                for img_file in dir_path.glob("*.png"):
                    no_number_samples.append((dir_id, str(img_file)))
            continue
        
        if valid_classes is not None and jersey_number not in valid_classes:
            continue
        
        dir_path = images_dir / str(dir_id)
        if not dir_path.exists() or not dir_path.is_dir():
            continue
        
        image_files = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png"))
        if not image_files:
            continue
        
        if jersey_number not in class_to_samples:
            class_to_samples[jersey_number] = []
        
        for img_file in image_files:
            class_to_samples[jersey_number].append((dir_id, str(img_file)))
    
    if config.min_samples_per_class is not None:
        valid_classes = {
            jersey_num: samples 
            for jersey_num, samples in class_to_samples.items() 
            if len(samples) >= config.min_samples_per_class
        }
        print(f"Found {len(valid_classes)} classes with at least {config.min_samples_per_class} samples")
    else:
        valid_classes = class_to_samples
        print(f"Found {len(valid_classes)} classes")
    
    if config.include_no_number_class and no_number_samples:
        random.shuffle(no_number_samples)
        if config.max_no_number_samples:
            no_number_samples = no_number_samples[:config.max_no_number_samples]
        valid_classes[-1] = no_number_samples
        print(f"Added 'no_number' class (-1) with {len(no_number_samples)} samples")
    
    image_paths = []
    labels = []
    
    max_samples = max_samples_override if max_samples_override is not None else config.max_samples_per_class
    
    for jersey_num in sorted(valid_classes.keys()):
        samples = valid_classes[jersey_num]
        
        random.shuffle(samples)
        if max_samples and jersey_num != -1:
            samples = samples[:max_samples]
        
        for image_id, img_path in samples:
            image_paths.append(img_path)
            labels.append(jersey_num)
    
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    labels_idx = [label_to_idx[label] for label in labels]
    
    print(f"Total samples: {len(image_paths)}")
    print(f"Number of classes: {len(unique_labels)}")
    print(f"Classes: {unique_labels[:10]}{'...' if len(unique_labels) > 10 else ''}")
    
    return image_paths, labels_idx, idx_to_label


def prepare_test_dataset(config: TestConfig, label_to_idx: Dict[int, int]) -> Tuple[List[str], List[int]]:
    print("Loading test ground truth...")
    gt_data = load_ground_truth(config.gt_file)
    
    valid_jersey_numbers = set(label_to_idx.keys())
    
    class_to_samples = {}
    skipped_unknown = 0
    skipped_invalid = 0
    
    for dir_id, jersey_number in gt_data.items():
        if jersey_number == -1:
            skipped_invalid += 1
            continue
        
        if jersey_number not in valid_jersey_numbers:
            skipped_unknown += 1
            continue
        
        dir_path = config.images_dir / str(dir_id)
        if not dir_path.exists() or not dir_path.is_dir():
            continue
        
        image_files = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png"))
        if not image_files:
            continue
        
        if jersey_number not in class_to_samples:
            class_to_samples[jersey_number] = []
        
        for img_file in image_files:
            class_to_samples[jersey_number].append((str(img_file), label_to_idx[jersey_number]))
    
    image_paths = []
    labels = []
    
    for jersey_number, samples in class_to_samples.items():
        random.shuffle(samples)
        
        if config.test_sample_ratio < 1.0:
            num_samples = max(1, int(len(samples) * config.test_sample_ratio))
            samples = samples[:num_samples]
        
        for img_path, label in samples:
            image_paths.append(img_path)
            labels.append(label)
    
    print(f"Test samples: {len(image_paths)} (sample ratio: {config.test_sample_ratio})")
    print(f"Skipped invalid (-1): {skipped_invalid}")
    print(f"Skipped unknown classes: {skipped_unknown}")
    
    return image_paths, labels
