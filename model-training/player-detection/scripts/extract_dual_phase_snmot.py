#!/usr/bin/env python3
"""SNMOT Dual-Phase Dataset Extraction.

This helper script builds two YOLO-ready datasets from the SNMOT tracking
format:

* Phase 1 (coarse): team_left / team_right / ball classes for generic
  player detection.
* Phase 2 (fine): separates goalkeepers and referees to support the second
  phase in the dual-stage training pipeline.

Both phases reuse the existing ``DatasetExtractor`` implementation and rely on
configuration snippets defined in ``configs/snmot_dual_phase.yaml``. Override
paths or run a subset of phases via the CLI if needed.
"""

from __future__ import annotations

import argparse
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from modules import create_extractor
from utils.config_utils import ConfigManager, setup_logging

logger = logging.getLogger(__name__)


def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``updates`` into ``target``."""
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def _ensure_str_paths(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ``Path`` instances in the mapping into string values."""
    for key, value in list(mapping.items()):
        if isinstance(value, Path):
            mapping[key] = str(value)
        elif isinstance(value, dict):
            _ensure_str_paths(value)
    return mapping


def _load_phase_config(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict) or 'phases' not in data:
        raise ValueError(f"Invalid phase configuration: {path}")

    phases = data['phases']
    if not isinstance(phases, dict) or not phases:
        raise ValueError("Phase configuration must define at least one phase")

    return phases


def _resolve_workspace_path(workspace_root: Path, maybe_relative: str) -> str:
    candidate = Path(maybe_relative)
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    return str(candidate.resolve())


def _select_phases(all_phases: Dict[str, Dict[str, Any]], selected: Optional[Iterable[str]]) -> Dict[str, Dict[str, Any]]:
    if not selected:
        return all_phases

    missing = [name for name in selected if name not in all_phases]
    if missing:
        raise ValueError(f"Unknown phase name(s): {', '.join(missing)}")

    return {name: all_phases[name] for name in selected}


def _prepare_phase_configs(
    phase_name: str,
    phase_cfg: Dict[str, Any],
    base_extraction: Dict[str, Any],
    base_paths: Dict[str, Any],
    workspace_root: Path,
    skip_clean: bool,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    extraction_cfg = deepcopy(base_extraction)
    paths_cfg = deepcopy(base_paths)

    extraction_cfg['dataset_type'] = 'snmot'
    extraction_cfg['run_extraction'] = True
    if skip_clean:
        extraction_cfg['clean_output'] = False
    elif 'clean_output' not in extraction_cfg:
        extraction_cfg['clean_output'] = True

    class_names = phase_cfg.get('class_names')
    if class_names is None:
        raise ValueError(f"Phase '{phase_name}' is missing 'class_names'")
    extraction_cfg['class_names'] = list(class_names)

    phase_snmot = phase_cfg.get('snmot_overrides') or {}
    _deep_update(extraction_cfg.setdefault('snmot', {}), phase_snmot)

    label_transform = phase_cfg.get('label_transform') or {}
    if label_transform:
        _deep_update(extraction_cfg.setdefault('snmot', {}).setdefault('label_transform', {}), label_transform)
        extraction_cfg['snmot']['label_transform'].setdefault('enabled', True)

    phase_paths = phase_cfg.get('paths') or {}
    _deep_update(paths_cfg, phase_paths)

    workspace_root = workspace_root.expanduser().resolve()
    paths_cfg['workspace_root'] = str(workspace_root)

    output_root = paths_cfg.get('output_root')
    if not output_root:
        raise ValueError(f"Phase '{phase_name}' must define an output_root")
    paths_cfg['output_root'] = _resolve_workspace_path(workspace_root, output_root)

    snmot_root = paths_cfg.get('snmot_root')
    if not snmot_root:
        raise ValueError("SNMOT root path is not configured")
    paths_cfg['snmot_root'] = _resolve_workspace_path(workspace_root, snmot_root)

    _ensure_str_paths(extraction_cfg)
    _ensure_str_paths(paths_cfg)
    return extraction_cfg, paths_cfg


def run_dual_phase_extraction(args: argparse.Namespace) -> None:
    config_manager = ConfigManager(args.config_dir)
    base_configs = config_manager.load_all_configs()

    extraction_cfg = base_configs.get('extraction') or {}
    paths_cfg = base_configs.get('paths') or {}

    if args.workspace_root:
        paths_cfg['workspace_root'] = str(args.workspace_root)
    if args.snmot_root:
        paths_cfg['snmot_root'] = str(args.snmot_root)

    workspace_root = Path(paths_cfg.get('workspace_root', '.'))

    phases = _load_phase_config(args.phase_config)
    phases_to_run = _select_phases(phases, args.phases)

    for phase_name, phase_cfg in phases_to_run.items():
        logger.info("🚀 Running SNMOT extraction phase '%s'", phase_name)
        description = phase_cfg.get('description')
        if description:
            logger.info("  • %s", description)
        extraction, paths = _prepare_phase_configs(
            phase_name,
            phase_cfg,
            extraction_cfg,
            paths_cfg,
            workspace_root,
            skip_clean=args.skip_clean,
        )

        output_root = paths['output_root']
        logger.info("  • Output root: %s", output_root)
        logger.info("  • Classes: %s", extraction.get('class_names'))

        extractor = create_extractor(extraction, paths)
        extractor.extract_dataset()

    logger.info("✅ Completed SNMOT dual-phase extraction")


def parse_args() -> argparse.Namespace:
    default_config_dir = Path(__file__).parent.parent / 'configs'
    default_phase_config = default_config_dir / 'snmot_dual_phase.yaml'

    parser = argparse.ArgumentParser(description="SNMOT dual-phase dataset extraction helper")
    parser.add_argument('--config-dir', type=Path, default=default_config_dir, help='Directory containing base YAML configs')
    parser.add_argument('--phase-config', type=Path, default=default_phase_config, help='Phase configuration YAML')
    parser.add_argument('--workspace-root', type=Path, help='Override workspace root directory')
    parser.add_argument('--snmot-root', type=Path, help='Override SNMOT dataset root directory')
    parser.add_argument('--phases', nargs='+', help='Specific phase names to run (default: all)')
    parser.add_argument('--skip-clean', action='store_true', help='Keep existing extracted datasets (no clean)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging verbosity')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)
    run_dual_phase_extraction(args)


if __name__ == '__main__':
    main()
