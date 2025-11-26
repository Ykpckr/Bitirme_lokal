#!/usr/bin/env python3
"""
YOLO Football Player Detection Training Pipeline

Modular training pipeline with configuration files and CLI support.
Refactored from the original training scripts to provide:
- Configuration-driven parameters
- Microservice-like module separation
- CLI argument support
- Better logging and error handling

Usage:
    python main.py                          # Use default configurations
    python main.py --run-extraction         # Force dataset extraction
    python main.py --epochs 200 --batch 32 # Override training parameters
    python main.py --extract-only           # Only extract dataset
    python main.py --train-only             # Skip extraction, only train
    python main.py --evaluate               # Evaluate trained model
    python main.py --predict /path/to/data  # Run inference
"""

import sys
import os
from pathlib import Path
import logging

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Module imports
from modules import create_device_manager, create_extractor, create_trainer
from utils import ConfigManager, setup_logging, validate_configurations

logger = logging.getLogger(__name__)


class YOLOTrainingPipeline:
    """Main orchestrator for the YOLO training pipeline."""
    
    def __init__(self):
        """Initialize the training pipeline."""
        self.config_manager = ConfigManager()
        self.configs = {}
        self.device_manager = None
        self.extractor = None
        self.trainer = None
        
    def run(self, args=None):
        """
        Run the complete training pipeline.
        
        Args:
            args: Command line arguments (None to parse from sys.argv)
        """
        try:
            # Parse arguments and load configurations
            parser = self.config_manager.create_cli_parser()
            parsed_args = parser.parse_args(args)
            
            # Setup logging
            setup_logging(
                level=parsed_args.log_level,
                log_file=parsed_args.log_file
            )
            
            logger.info("=" * 80)
            logger.info("YOLO FOOTBALL PLAYER DETECTION TRAINING PIPELINE")
            logger.info("=" * 80)
            
            # Load and validate configurations
            self._load_configurations(parsed_args)
            
            # Initialize components
            self._initialize_components()
            
            # Execute pipeline based on arguments
            self._execute_pipeline(parsed_args)
            
            logger.info("🎉 Pipeline completed successfully!")
            
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            sys.exit(1)
    
    def _load_configurations(self, args):
        """Load and validate all configurations."""
        logger.info("Loading configurations...")
        
        # Update config directory if specified
        if args.config_dir:
            self.config_manager = ConfigManager(args.config_dir)
        
        # Load all configuration files
        self.configs = self.config_manager.load_all_configs()
        
        # Merge with CLI arguments
        self.configs = self.config_manager.merge_configs_with_args(
            self.configs, args
        )
        
        # Validate configurations
        errors = validate_configurations(self.configs)
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            raise ValueError("Invalid configuration")
        
        logger.info("✅ Configurations loaded and validated")
        self._log_configuration_summary()
    
    def _log_configuration_summary(self):
        """Log a summary of the loaded configurations."""
        logger.info("\nConfiguration Summary:")
        logger.info("-" * 40)
        
        # Paths
        paths = self.configs.get('paths', {})
        extraction_cfg = self.configs.get('extraction', {})
        dataset_type = (extraction_cfg.get('dataset_type') or 'soccernet').lower()

        logger.info(f"Workspace Root: {paths.get('workspace_root')}")
        if dataset_type == 'snmot':
            logger.info(f"Dataset Type: SNMOT")
            logger.info(f"SNMOT Root: {paths.get('snmot_root')}")
        else:
            logger.info(f"Dataset Type: SoccerNet")
            logger.info(f"SoccerNet Root: {paths.get('soccernet_root')}")

        logger.info(f"Output Root: {paths.get('output_root')}")
        logger.info(f"Models Root: {paths.get('models_root')}")
        
        # Key training parameters
        yolo = self.configs.get('yolo_params', {})
        logger.info(f"Model: {yolo.get('model')}")
        logger.info(f"Epochs: {yolo.get('epochs')}")
        logger.info(f"Batch Size: {yolo.get('batch')}")
        logger.info(f"Image Size: {yolo.get('imgsz')}")
        
        # Extraction settings
        extraction = self.configs.get('extraction', {})
        logger.info(f"Run Extraction: {extraction.get('run_extraction')}")
        if extraction.get('max_samples_per_half'):
            logger.info(f"Max Samples/Half: {extraction.get('max_samples_per_half')}")
        
        logger.info("-" * 40)
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        logger.info("Initializing pipeline components...")
        
        # Initialize device manager
        device_config = self.configs.get('device', {})
        self.device_manager = create_device_manager(device_config)
        
        # Select training device
        device = self.device_manager.select_training_device()
        self.device_manager.log_device_info()
        
        # Initialize dataset extractor
        extraction_config = self.configs.get('extraction', {})
        paths_config = self.configs.get('paths', {})
        self.extractor = create_extractor(extraction_config, paths_config)
        
        # Initialize trainer
        yolo_config = self.configs.get('yolo_params', {})
        amp_enabled = self.device_manager.amp_enabled
        self.trainer = create_trainer(yolo_config, paths_config, device, amp_enabled)
        
        logger.info("✅ Components initialized")
    
    def _execute_pipeline(self, args):
        """Execute the pipeline based on command line arguments."""
        if args.extract_only:
            self._run_extraction_only()
        elif args.train_only:
            self._run_training_only()
        elif args.evaluate:
            self._run_evaluation()
        elif args.predict:
            self._run_prediction(args.predict)
        else:
            self._run_full_pipeline()
    
    def _run_extraction_only(self):
        """Run only dataset extraction."""
        logger.info("Running extraction-only pipeline...")
        output_root = self.extractor.extract_dataset()
        logger.info(f"✅ Dataset extracted to: {output_root}")
    
    def _run_training_only(self):
        """Run only training (skip extraction)."""
        logger.info("Running training-only pipeline...")
        
        # Ensure dataset exists
        dataset_yaml = self.extractor.ensure_dataset_yaml()
        logger.info(f"Using existing dataset: {dataset_yaml}")
        
        # Train model
        model_path = self.trainer.train(dataset_yaml)
        if model_path:
            logger.info(f"✅ Model trained successfully: {model_path}")
            
            # Reports are automatically generated in the configured reports directory
            workspace_root = Path(self.configs['paths']['workspace_root']).expanduser()
            reports_root = workspace_root / self.configs['paths'].get('reports_root', 'reports')
            project_name = self.configs['yolo_params'].get('project_name', 'football_detector')
            project_reports = reports_root / project_name
            
            if project_reports.exists():
                logger.info(f"📊 Training reports available in: {project_reports}")
        else:
            logger.warning("Training completed but best model path not found")
    
    def _run_full_pipeline(self):
        """Run the complete pipeline (extraction + training)."""
        logger.info("Running full pipeline...")
        
        # Extract dataset
        output_root = self.extractor.extract_dataset()
        
        # Get dataset configuration
        dataset_yaml = self.extractor.ensure_dataset_yaml()
        
        # Train model
        model_path = self.trainer.train(dataset_yaml)
        if model_path:
            logger.info(f"✅ Complete pipeline finished. Model: {model_path}")
            
            # Reports are automatically generated in the configured reports directory
            workspace_root = Path(self.configs['paths']['workspace_root']).expanduser()
            reports_root = workspace_root / self.configs['paths'].get('reports_root', 'reports')
            project_name = self.configs['yolo_params'].get('project_name', 'football_detector')
            project_reports = reports_root / project_name
            
            if project_reports.exists():
                logger.info(f"📊 Training reports available in: {project_reports}")
        else:
            logger.warning("Training completed but best model path not found")
    
    def _run_evaluation(self):
        """Run model evaluation."""
        logger.info("Running model evaluation...")
        
        # Ensure dataset exists for evaluation
        dataset_yaml = self.extractor.ensure_dataset_yaml()
        
        # Find latest model or use specified model
        models_root = Path(self.configs['paths']['workspace_root']).expanduser()
        models_root = models_root / self.configs['paths']['models_root']
        project_name = self.configs['yolo_params'].get('project_name', 'football_detector_optimized')
        model_path = models_root / project_name / "weights" / "best.pt"
        
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            logger.info("Please train a model first or specify the correct model path")
            return
        
        # Run evaluation
        results = self.trainer.evaluate(model_path)
        logger.info("✅ Evaluation completed")
    
    def _run_prediction(self, source: str):
        """Run model prediction/inference."""
        logger.info(f"Running inference on: {source}")
        
        # Find latest model
        models_root = Path(self.configs['paths']['workspace_root']).expanduser()
        models_root = models_root / self.configs['paths']['models_root']
        project_name = self.configs['yolo_params'].get('project_name', 'football_detector_optimized')
        model_path = models_root / project_name / "weights" / "best.pt"
        
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            logger.info("Please train a model first or specify the correct model path")
            return
        
        # Run prediction
        results = self.trainer.predict(source, model_path)
        logger.info("✅ Inference completed")


def main():
    """Main entry point."""
    pipeline = YOLOTrainingPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()