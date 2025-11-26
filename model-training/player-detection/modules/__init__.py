"""
Module Initialization

Makes the modules directory a Python package.
"""

__version__ = "1.0.0"
__author__ = "YOLO Training Pipeline"

# Import main classes for convenience
from .device_manager import DeviceManager, create_device_manager
from .dataset_extractor import DatasetExtractor, create_extractor
from .trainer import YOLOTrainer, create_trainer
from .team_assigner import TeamAssigner, DetectionObservation, AssignedTeam

__all__ = [
    'DeviceManager',
    'create_device_manager',
    'DatasetExtractor', 
    'create_extractor',
    'YOLOTrainer',
    'create_trainer',
    'TeamAssigner',
    'DetectionObservation',
    'AssignedTeam',
]