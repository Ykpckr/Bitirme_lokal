from pathlib import Path
from dataclasses import dataclass
import torch


@dataclass
class TrainingConfig:
    train_dir: Path = Path("train")
    train_images_dir: Path = train_dir / "train" / "images"
    train_gt_file: Path = train_dir / "train" / "train_gt.json"
    
    test_dir: Path = Path("test")
    test_images_dir: Path = test_dir / "test" / "images"
    test_gt_file: Path = test_dir / "test" / "test_gt.json"
    
    output_dir: Path = Path("output")
    
    max_samples_per_class: int = None
    min_samples_per_class: int = 5
    max_val_samples_per_class: int = None
    include_no_number_class: bool = False
    max_no_number_samples: int = 70
    use_test_as_val: bool = False
    test_size: float = 0.2
    random_seed: int = 42
    
    batch_size: int = 64
    num_epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    
    model_name: str = "resnet18"
    pretrained: bool = True
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    use_original_size: bool = False
    image_size: int = 32
    
    early_stopping_patience: int = 10


@dataclass
class TestConfig:
    test_dir: Path = Path("test")
    images_dir: Path = test_dir / "test" / "images"
    gt_file: Path = test_dir / "test" / "test_gt.json"
    model_path: Path = Path("output/best_model.pth")
    output_dir: Path = Path("output/test_results")
    
    test_sample_ratio: float = 0.001
    batch_size: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
