"""
Training Module

Handles YOLO model training with configuration-driven parameters.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Optional: live class-wise reporter
try:
    from utils.classwise_reporter import ClasswiseReporter
except Exception:
    ClasswiseReporter = None


class YOLOTrainer:
    """Handles YOLO model training."""
    
    def __init__(
        self,
        yolo_config: Dict[str, Any],
        paths_config: Dict[str, Any],
        device: str,
        amp_enabled: bool = True
    ):
        """
        Initialize YOLO trainer.
        
        Args:
            yolo_config: YOLO training configuration
            paths_config: Path configuration
            device: Training device ('cuda:0', 'dml', 'cpu')
            amp_enabled: Whether to enable automatic mixed precision
        """
        self.yolo_config = yolo_config.copy()
        self.paths_config = paths_config
        self.device = device
        self.amp_enabled = amp_enabled
        self.model = None
        
    def train(self, dataset_yaml_path: Path) -> Optional[Path]:
        """
        Train YOLO model.
        
        Args:
            dataset_yaml_path: Path to dataset.yaml configuration
            
        Returns:
            Path to trained model or None if training failed
        """
        logger.info("🏋️ Starting YOLO training...")
        
        # Prepare training configuration
        training_config = self._prepare_training_config(dataset_yaml_path)
        
        # Log configuration
        self._log_training_config(training_config)
        
        # Create model
        model_arch = training_config.pop("model")
        self.model = YOLO(model_arch)
        # Optionally start classwise reporter to watch training outputs
        reporter = None
        try:
            models_root = self._get_models_root()
            project_name = self.yolo_config.get('project_name', 'football_detector')
            project_dir = models_root / project_name
            if ClasswiseReporter is not None:
                reporter = ClasswiseReporter(models_root, project_name, poll_interval=5.0)
                reporter.start()
        except Exception as e:
            logger.debug(f"Could not start ClasswiseReporter: {e}")

        # Start training
        try:
            results = self.model.train(**training_config)
            logger.info("✅ Training completed successfully")
            
            # Get model paths
            models_root = self._get_models_root()
            project_name = self.yolo_config.get('project_name', 'football_detector')
            project_dir = models_root / project_name
            best_model_path = project_dir / "weights" / "best.pt"
            
            # Generate training reports
            try:
                self._generate_training_reports(results, project_dir, best_model_path)
            except Exception as report_error:
                logger.warning(f"Failed to generate reports: {report_error}")
            
            if best_model_path.exists():
                if reporter is not None:
                    try:
                        reporter.stop()
                    except Exception:
                        pass
                return best_model_path
            else:
                logger.warning("Best model not found at expected location")
                if reporter is not None:
                    try:
                        reporter.stop()
                    except Exception:
                        pass
                return None
                
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            if reporter is not None:
                try:
                    reporter.stop()
                except Exception:
                    pass
            raise
    
    def _prepare_training_config(self, dataset_yaml_path: Path) -> Dict[str, Any]:
        """Prepare training configuration dictionary."""
        config = self.yolo_config.copy()
        
        # Set essential paths and device
        config["data"] = str(dataset_yaml_path)
        config["device"] = self.device
        config["amp"] = self.amp_enabled
        
        # Set project directory
        models_root = self._get_models_root()
        config["project"] = str(models_root)
        
        # Handle project name
        project_name = config.pop("project_name", "football_detector")
        config["name"] = project_name
        
        return config
    
    def _get_models_root(self) -> Path:
        """Get models root directory."""
        workspace_root = Path(self.paths_config['workspace_root']).expanduser()
        return workspace_root / self.paths_config['models_root']
    
    def _log_training_config(self, config: Dict[str, Any]):
        """Log training configuration."""
        logger.info("\nTraining Configuration:")
        logger.info("=" * 60)
        
        # Group related parameters for better readability
        param_groups = {
            "Model & Data": ["model", "data", "device", "amp"],
            "Training": ["epochs", "batch", "imgsz", "workers"],
            "Optimization": ["optimizer", "lr0", "lrf", "momentum", "weight_decay"],
            "Loss Weights": ["box", "cls", "dfl"],
            "Augmentation": ["hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale"],
            "Advanced": ["mosaic", "mixup", "copy_paste", "multi_scale", "cos_lr"],
            "Validation": ["val", "conf", "iou", "max_det"],
            "Output": ["project", "name", "save", "save_period"]
        }
        
        # Log grouped parameters
        for group_name, param_names in param_groups.items():
            logger.info(f"\n{group_name}:")
            for param in param_names:
                if param in config:
                    logger.info(f"  {param:15s}: {config[param]}")
        
        # Log remaining parameters
        logged_params = set()
        for params in param_groups.values():
            logged_params.update(params)
        
        remaining = {k: v for k, v in config.items() if k not in logged_params}
        if remaining:
            logger.info(f"\nOther Parameters:")
            for param, value in remaining.items():
                logger.info(f"  {param:15s}: {value}")
        
        logger.info("=" * 60)
    
    def get_model(self) -> Optional[YOLO]:
        """Get the trained model instance."""
        return self.model
    
    def evaluate(self, model_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            model_path: Path to model weights. If None, uses current model.
            
        Returns:
            Evaluation results
        """
        if model_path and model_path.exists():
            model = YOLO(str(model_path))
        elif self.model:
            model = self.model
        else:
            raise ValueError("No model available for evaluation")
        
        logger.info("📊 Evaluating model...")
        
        try:
            results = model.val()
            logger.info("✅ Evaluation completed")
            return results
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise
    
    def predict(
        self,
        source: str,
        model_path: Optional[Path] = None,
        save: bool = True,
        conf: Optional[float] = None
    ) -> Any:
        """
        Run inference with the trained model.
        
        Args:
            source: Input source (image, video, directory)
            model_path: Path to model weights. If None, uses current model.
            save: Whether to save results
            conf: Confidence threshold
            
        Returns:
            Prediction results
        """
        if model_path and model_path.exists():
            model = YOLO(str(model_path))
        elif self.model:
            model = self.model
        else:
            raise ValueError("No model available for prediction")
        
        logger.info(f"🔍 Running inference on: {source}")
        
        # Prepare prediction arguments
        predict_args = {
            "source": source,
            "save": save,
            "device": self.device
        }
        
        if conf is not None:
            predict_args["conf"] = conf
        else:
            predict_args["conf"] = self.yolo_config.get("conf", 0.01)
        
        try:
            results = model.predict(**predict_args)
            logger.info("✅ Inference completed")
            return results
        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            raise
    
    def _generate_training_reports(
        self,
        results: Any,
        project_dir: Path,
        model_path: Optional[Path]
    ):
        """Generate comprehensive training reports."""
        logger.info("📊 Generating training reports...")
        
        try:
            # Import report generator with absolute import
            try:
                from reports import create_report_generator
            except ImportError:
                # Try adding current directory to path
                import sys
                from pathlib import Path as PathLib
                current_dir = PathLib(__file__).parent.parent
                if str(current_dir) not in sys.path:
                    sys.path.insert(0, str(current_dir))
                from reports import create_report_generator
            
            # Get reports root from configuration
            workspace_root = Path(self.paths_config['workspace_root']).expanduser()
            reports_root = workspace_root / self.paths_config.get('reports_root', 'reports')
            
            # Create timestamped directory for this specific training run
            from datetime import datetime
            timestamp = datetime.now().strftime('%H%M-%d%m%Y')
            project_name = self.yolo_config.get('project_name', 'football_detector')
            
            # Create structure: reports/project_name/timestamp/
            training_reports_dir = reports_root / project_name / timestamp
            training_reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Also create reports in the model directory for easy access
            model_reports_dir = project_dir / "reports"
            model_reports_dir.mkdir(exist_ok=True)
            
            # Initialize report generator (use timestamped directory)
            report_gen = create_report_generator(training_reports_dir)
            
            # Collect dataset statistics
            dataset_stats = self._collect_dataset_stats()
            
            # Prepare configuration for report
            report_config = {
                'yolo_params': self.yolo_config,
                'paths': self.paths_config,
                'device': self.device,
                'amp_enabled': self.amp_enabled,
                'training_timestamp': timestamp,
            }
            
            # Generate reports
            report_paths = report_gen.generate_reports(
                training_results=results,
                config=report_config,
                model_path=model_path,
                dataset_stats=dataset_stats
            )
            
            # Log report locations
            logger.info(f"📁 Training reports directory: {training_reports_dir}")
            for report_type, path in report_paths.items():
                logger.info(f"📋 {report_type.upper()} report: {path}")
                
            # Create symlinks in model directory for convenience
            try:
                for report_type, source_path in report_paths.items():
                    link_path = model_reports_dir / source_path.name
                    if link_path.exists():
                        link_path.unlink()
                    link_path.symlink_to(source_path)
                logger.info(f"📎 Report links created in: {model_reports_dir}")
            except Exception as e:
                logger.debug(f"Could not create report symlinks: {e}")
                
        except ImportError as e:
            logger.warning(f"Report generation dependencies not available: {e}")
            logger.info("Install with: pip install matplotlib seaborn openpyxl")
        except Exception as e:
            logger.error(f"Failed to generate reports: {e}")
            raise
    
    def _collect_dataset_stats(self) -> Dict[str, Any]:
        """Collect basic dataset statistics."""
        stats = {}
        
        try:
            # Get dataset path from current training config
            data_path = self.yolo_config.get('data')
            if data_path:
                import yaml
                with open(data_path, 'r') as f:
                    dataset_config = yaml.safe_load(f)
                
                dataset_root = Path(dataset_config['path'])
                
                # Count images in each split
                for split in ['train', 'val', 'test']:
                    split_dir = dataset_root / split / 'images'
                    if split_dir.exists():
                        image_count = len(list(split_dir.glob('*.jpg')) + list(split_dir.glob('*.png')))
                        stats[f'{split}_images'] = image_count
                
                # Add class information
                stats['num_classes'] = dataset_config.get('nc', 0)
                stats['class_names'] = dataset_config.get('names', [])
                
        except Exception as e:
            logger.debug(f"Could not collect dataset stats: {e}")
        
        return stats


def create_trainer(
    yolo_config: Dict[str, Any],
    paths_config: Dict[str, Any],
    device: str,
    amp_enabled: bool = True
) -> YOLOTrainer:
    """
    Factory function to create a YOLO trainer.
    
    Args:
        yolo_config: YOLO training configuration
        paths_config: Path configuration
        device: Training device
        amp_enabled: Whether to enable automatic mixed precision
        
    Returns:
        Configured YOLOTrainer instance
    """
    return YOLOTrainer(yolo_config, paths_config, device, amp_enabled)