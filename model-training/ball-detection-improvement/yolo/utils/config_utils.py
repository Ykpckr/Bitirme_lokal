"""
Configuration Utilities for the standalone ball-detection-improvement pipeline.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigManager:
    """Loads YAML configs and merges CLI overrides."""

    def __init__(self, config_dir: Optional[Path] = None):
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "configs"
        self.config_dir = Path(config_dir)
        self._configs: Dict[str, Dict[str, Any]] = {}

    def load_config(self, config_name: str) -> Dict[str, Any]:
        if config_name in self._configs:
            return self._configs[config_name].copy()

        config_path = self.config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        self._configs[config_name] = config
        return config.copy()

    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        configs: Dict[str, Dict[str, Any]] = {}
        for name in ["paths", "extraction", "yolo_params", "device"]:
            try:
                configs[name] = self.load_config(name)
            except FileNotFoundError:
                logger.warning("Optional config missing: %s.yaml", name)
                configs[name] = {}
        return configs

    def create_cli_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="YOLO Ball Detection Training Pipeline",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        parser.add_argument(
            "--config-dir",
            type=Path,
            default=self.config_dir,
            help="Directory containing configuration files",
        )

        paths_group = parser.add_argument_group("Paths")
        paths_group.add_argument("--workspace-root", help="Workspace root directory")
        paths_group.add_argument("--soccernet-root", help="SoccerNet dataset root")
        paths_group.add_argument("--output-root", help="Output dataset root")
        paths_group.add_argument("--models-root", help="Models directory")
        paths_group.add_argument("--reports-root", help="Reports directory")

        extraction_group = parser.add_argument_group("Dataset Extraction")
        extraction_group.add_argument("--run-extraction", action="store_true", help="Force extraction run")
        extraction_group.add_argument("--no-extraction", action="store_true", help="Skip extraction")
        extraction_group.add_argument("--max-samples-per-half", type=int)
        extraction_group.add_argument("--det-start-sec", type=float)
        extraction_group.add_argument("--frame-shift", type=int)

        training_group = parser.add_argument_group("Training")
        training_group.add_argument("--model", help="Pretrained weights path or architecture")
        training_group.add_argument("--epochs", type=int)
        training_group.add_argument("--batch", type=int)
        training_group.add_argument("--imgsz", type=int)
        training_group.add_argument("--lr0", type=float)
        training_group.add_argument("--project-name", help="Training project name")
        training_group.add_argument("--fraction", type=float)

        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            default="INFO",
        )
        parser.add_argument("--log-file", type=Path)

        actions = parser.add_argument_group("Actions")
        actions.add_argument("--extract-only", action="store_true")
        actions.add_argument("--train-only", action="store_true")
        actions.add_argument("--evaluate", action="store_true")
        actions.add_argument("--predict", help="Path for inference")

        return parser

    def merge_configs_with_args(
        self,
        configs: Dict[str, Dict[str, Any]],
        args: argparse.Namespace,
    ) -> Dict[str, Dict[str, Any]]:
        merged = {name: cfg.copy() for name, cfg in configs.items()}

        paths = merged.setdefault("paths", {})
        if args.workspace_root:
            paths["workspace_root"] = args.workspace_root
        if args.soccernet_root:
            paths["soccernet_root"] = args.soccernet_root
        if args.output_root:
            paths["output_root"] = args.output_root
        if args.models_root:
            paths["models_root"] = args.models_root
        if args.reports_root:
            paths["reports_root"] = args.reports_root

        extraction = merged.setdefault("extraction", {})
        if args.run_extraction:
            extraction["run_extraction"] = True
        elif args.no_extraction:
            extraction["run_extraction"] = False
        if args.max_samples_per_half is not None:
            extraction["max_samples_per_half"] = args.max_samples_per_half
        if args.det_start_sec is not None:
            extraction["det_start_sec"] = args.det_start_sec
        if args.frame_shift is not None:
            extraction["frame_shift"] = args.frame_shift

        yolo = merged.setdefault("yolo_params", {})
        if args.model:
            yolo["model"] = args.model
        if args.epochs is not None:
            yolo["epochs"] = args.epochs
        if args.batch is not None:
            yolo["batch"] = args.batch
        if args.imgsz is not None:
            yolo["imgsz"] = args.imgsz
        if args.lr0 is not None:
            yolo["lr0"] = args.lr0
        if args.project_name:
            yolo["project_name"] = args.project_name
        if args.fraction is not None:
            yolo["fraction"] = args.fraction

        return merged


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def validate_configurations(configs: Dict[str, Dict[str, Any]]) -> List[str]:
    errors: List[str] = []

    for name in ["paths", "yolo_params", "extraction", "device"]:
        if name not in configs:
            errors.append(f"Missing required configuration: {name}")

    paths = configs.get("paths", {})
    for key in ["workspace_root", "output_root"]:
        if not paths.get(key):
            errors.append(f"Missing required path: {key}")

    yolo = configs.get("yolo_params", {})
    if not yolo.get("model"):
        errors.append("yolo_params.model must point to pretrained weights")
    if not isinstance(yolo.get("epochs", 0), int) or yolo.get("epochs", 0) <= 0:
        errors.append("yolo_params.epochs must be positive integer")
    if not isinstance(yolo.get("batch", 0), int) or yolo.get("batch", 0) <= 0:
        errors.append("yolo_params.batch must be positive integer")

    return errors


__all__ = [
    "ConfigManager",
    "setup_logging",
    "validate_configurations",
]
