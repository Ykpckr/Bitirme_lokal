"""
YOLO Football Player Detection Training Pipeline

A modular, configuration-driven training pipeline for YOLO-based
football player detection using SoccerNet dataset.
"""

__version__ = "1.0.0"
__author__ = "YOLO Training Pipeline Team"
__description__ = "Modular YOLO training pipeline for football player detection"

# Main pipeline class
try:  # pragma: no cover - import convenience for direct execution contexts
	from .main import YOLOTrainingPipeline
except ImportError:  # pragma: no cover
	from main import YOLOTrainingPipeline

__all__ = ['YOLOTrainingPipeline']