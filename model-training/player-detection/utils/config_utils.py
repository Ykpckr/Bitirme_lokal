"""
Configuration Utilities

Handles loading and merging of configuration files with CLI arguments.
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and CLI argument integration."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir is None:
            # Default to configs directory relative to this file
            config_dir = Path(__file__).parent.parent / "configs"
        
        self.config_dir = Path(config_dir)
        self._configs = {}
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a configuration file.
        
        Args:
            config_name: Name of config file (without .yaml extension)
            
        Returns:
            Configuration dictionary
        """
        if config_name in self._configs:
            return self._configs[config_name].copy()
        
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            self._configs[config_name] = config
            logger.debug(f"Loaded configuration: {config_name}")
            return config.copy()
            
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {config_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration {config_path}: {e}")
    
    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all available configuration files.
        
        Returns:
            Dictionary mapping config names to their contents
        """
        configs = {}
        
        # Define standard config files
        config_files = ['paths', 'yolo_params', 'extraction', 'device', 'visualization']
        
        for config_name in config_files:
            try:
                configs[config_name] = self.load_config(config_name)
            except FileNotFoundError:
                logger.warning(f"Optional config file not found: {config_name}.yaml")
                configs[config_name] = {}
        
        return configs
    
    def create_cli_parser(self) -> argparse.ArgumentParser:
        """
        Create command line argument parser with config options.
        
        Returns:
            Configured ArgumentParser
        """
        parser = argparse.ArgumentParser(
            description="YOLO Football Player Detection Training Pipeline",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Configuration files
        parser.add_argument(
            '--config-dir',
            type=Path,
            default=self.config_dir,
            help='Directory containing configuration files'
        )
        
        # Paths
        paths_group = parser.add_argument_group('Paths')
        paths_group.add_argument('--workspace-root', help='Workspace root directory')
        paths_group.add_argument('--soccernet-root', help='SoccerNet dataset root')
        paths_group.add_argument('--snmot-root', help='SNMOT dataset root directory')
        paths_group.add_argument('--output-root', help='Output dataset root')
        paths_group.add_argument('--models-root', help='Models output directory')
        paths_group.add_argument('--reports-root', help='Reports output directory')
        
        # Extraction options
        extraction_group = parser.add_argument_group('Dataset Extraction')
        extraction_group.add_argument(
            '--dataset-type',
            choices=['soccernet', 'snmot'],
            help='Dataset format to extract'
        )
        extraction_group.add_argument(
            '--run-extraction',
            action='store_true',
            help='Run dataset extraction'
        )
        extraction_group.add_argument(
            '--no-extraction',
            action='store_true',
            help='Skip dataset extraction'
        )
        extraction_group.add_argument(
            '--max-samples-per-half',
            type=int,
            help='Maximum samples per half'
        )
        extraction_group.add_argument(
            '--det-start-sec',
            type=float,
            help='Detection start time offset'
        )
        extraction_group.add_argument(
            '--frame-shift',
            type=int,
            help='Frame shift for alignment'
        )
        
        # Training options
        training_group = parser.add_argument_group('Training')
        training_group.add_argument('--model', help='YOLO model architecture')
        training_group.add_argument('--epochs', type=int, help='Number of training epochs')
        training_group.add_argument('--batch', type=int, help='Batch size')
        training_group.add_argument('--imgsz', type=int, help='Input image size')
        training_group.add_argument('--lr0', type=float, help='Initial learning rate')
        training_group.add_argument('--device', help='Training device (cuda:0, dml, cpu)')
        training_group.add_argument(
            '--project-name',
            help='Training project name'
        )

        # Visualization options
        viz_group = parser.add_argument_group('Visualization')
        viz_group.add_argument(
            '--tensorboard',
            action='store_true',
            help='Launch TensorBoard server during the run'
        )
        viz_group.add_argument(
            '--tensorboard-logdir',
            help='Directory for TensorBoard to watch'
        )
        viz_group.add_argument(
            '--tensorboard-host',
            help='TensorBoard host (default: 127.0.0.1)'
        )
        viz_group.add_argument(
            '--tensorboard-port',
            type=int,
            help='TensorBoard port (default: 6006)'
        )
        viz_group.add_argument(
            '--tensorboard-open-browser',
            action='store_true',
            help='Automatically open browser when TensorBoard starts'
        )
        
        # Logging
        parser.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default='INFO',
            help='Logging level'
        )
        parser.add_argument(
            '--log-file',
            type=Path,
            help='Log file path'
        )
        
        # Actions
        action_group = parser.add_argument_group('Actions')
        action_group.add_argument(
            '--extract-only',
            action='store_true',
            help='Only run dataset extraction'
        )
        action_group.add_argument(
            '--train-only',
            action='store_true',
            help='Only run training (skip extraction)'
        )
        action_group.add_argument(
            '--evaluate',
            action='store_true',
            help='Evaluate trained model'
        )
        action_group.add_argument(
            '--predict',
            help='Run inference on source (image/video/directory)'
        )
        
        return parser
    
    def merge_configs_with_args(
        self,
        configs: Dict[str, Dict[str, Any]],
        args: argparse.Namespace
    ) -> Dict[str, Dict[str, Any]]:
        """
        Merge configuration files with command line arguments.
        
        Args:
            configs: Loaded configuration dictionaries
            args: Parsed command line arguments
            
        Returns:
            Merged configuration dictionaries
        """
        merged = {}
        
        for config_name, config in configs.items():
            merged[config_name] = config.copy()
        
        # Override with command line arguments
        if args.workspace_root:
            merged['paths']['workspace_root'] = args.workspace_root
        if args.soccernet_root:
            merged['paths']['soccernet_root'] = args.soccernet_root
        if args.snmot_root:
            merged['paths']['snmot_root'] = args.snmot_root
        if args.output_root:
            merged['paths']['output_root'] = args.output_root
        if args.models_root:
            merged['paths']['models_root'] = args.models_root
        if args.reports_root:
            merged['paths']['reports_root'] = args.reports_root
        
        # Extraction overrides
        if args.dataset_type:
            merged['extraction']['dataset_type'] = args.dataset_type
        if args.run_extraction:
            merged['extraction']['run_extraction'] = True
        elif args.no_extraction:
            merged['extraction']['run_extraction'] = False
        if args.max_samples_per_half is not None:
            merged['extraction']['max_samples_per_half'] = args.max_samples_per_half
        if args.det_start_sec is not None:
            merged['extraction']['det_start_sec'] = args.det_start_sec
        if args.frame_shift is not None:
            merged['extraction']['frame_shift'] = args.frame_shift
        
        # Training overrides
        if args.model:
            merged['yolo_params']['model'] = args.model
        if args.epochs is not None:
            merged['yolo_params']['epochs'] = args.epochs
        if args.batch is not None:
            merged['yolo_params']['batch'] = args.batch
        if args.imgsz is not None:
            merged['yolo_params']['imgsz'] = args.imgsz
        if args.lr0 is not None:
            merged['yolo_params']['lr0'] = args.lr0
        if args.project_name:
            merged['yolo_params']['project_name'] = args.project_name

        # Visualization overrides
        viz_cfg = merged.setdefault('visualization', {}).setdefault('tensorboard', {})
        if args.tensorboard:
            viz_cfg['enabled'] = True
        if args.tensorboard_logdir:
            viz_cfg['logdir'] = args.tensorboard_logdir
        if args.tensorboard_host:
            viz_cfg['host'] = args.tensorboard_host
        if args.tensorboard_port is not None:
            viz_cfg['port'] = args.tensorboard_port
        if args.tensorboard_open_browser:
            viz_cfg['open_browser'] = True
        
        return merged


def setup_logging(level: str = 'INFO', log_file: Optional[Path] = None):
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")


def validate_configurations(configs: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Validate configuration dictionaries.
    
    Args:
        configs: Configuration dictionaries to validate
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    # Required configurations
    required_configs = ['paths', 'yolo_params', 'extraction', 'device']
    for config_name in required_configs:
        if config_name not in configs:
            errors.append(f"Missing required configuration: {config_name}")
    
    # Validate paths
    if 'paths' in configs:
        paths = configs['paths']
        dataset_type = configs.get('extraction', {}).get('dataset_type', 'soccernet')
        dataset_type = (dataset_type or 'soccernet').lower()

        if dataset_type not in {'soccernet', 'snmot'}:
            errors.append(f"Unsupported dataset_type: {dataset_type}")

        required_paths = ['workspace_root', 'output_root', 'models_root']
        if dataset_type == 'snmot':
            required_paths.append('snmot_root')
        else:
            required_paths.append('soccernet_root')

        for path_key in required_paths:
            if not paths.get(path_key):
                errors.append(f"Missing required path: {path_key}")
    
    # Validate YOLO parameters
    if 'yolo_params' in configs:
        yolo = configs['yolo_params']
        if not yolo.get('model'):
            errors.append("Missing YOLO model architecture")
        if not isinstance(yolo.get('epochs', 0), int) or yolo.get('epochs', 0) <= 0:
            errors.append("Invalid epochs value")
        if not isinstance(yolo.get('batch', 0), int) or yolo.get('batch', 0) <= 0:
            errors.append("Invalid batch size")
    
    return errors