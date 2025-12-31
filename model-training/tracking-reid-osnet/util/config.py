"""YAML config loader for tracking runners.

This replaces the older JSON-based loader (`config_utils.py`) so the main runner can
run with zero CLI args using `config.yaml`.

Notes:
- Uses PyYAML (`pip install pyyaml`).
- Expands env vars and `~` and resolves relative paths against the config file dir.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

import os


@dataclass
class BoTSORTTeamReIDConfig:
    # IO
    video: str = ""
    detector_weights: str = r"C:\\Users\\Admin\\Desktop\\FoMAC\\FoMAC\\model-training\\ball-detection\\models\\player_ball_detector\\weights\\best.pt"
    reid_weights: str = r"C:\\Users\\Admin\\Desktop\\sn-reid\\sn-reid\\log\\model\\model.pth.tar-60"
    device: str = "cuda:0"
    save_video: str = ""
    save_txt: str = ""

    # Costs / gates
    w_iou: float = 0.6
    w_app: float = 0.35
    w_team: float = 0.05
    iou_gate: float = 0.05
    app_gate: float = 0.2
    team_strict: bool = False

    # Two-stage association / track spawning
    second_stage_iou: bool = True
    iou_gate_second: float = 0.2
    new_track_min_conf: float = 0.4

    # Track mgmt
    max_age: int = 30
    min_hits: int = 3
    alpha_embed: float = 0.9

    # ReID speed controls
    reid_batch_size: int = 64
    reid_fp16: bool = True
    reid_every_n: int = 1
    reid_min_conf: float = 0.25
    reid_topk: int = 60

    # Progress output
    progress_every_sec: float = 5.0

    # Referee rendering / labeling
    referee_team_id: int = 2

    # Re-linking (after leaving view)
    relink_enabled: bool = True
    relink_max_age: int = 1800
    relink_app_gate: float = 0.30
    relink_sim_margin: float = 0.05
    relink_team_strict: bool = True
    relink_only_player: bool = True

    # ReID embedding gallery (improves relink on re-entry)
    embed_gallery_size: int = 10

    # Team gating
    team_penalize_classes: str = "0"  # comma-separated class ids (default: player only)

    # Relink spatial constraint
    relink_max_center_dist_norm: float = 0.0  # 0 disables

    # Embedding drift safeguard (IoU-only association)
    embed_update_min_sim_iou_only: float = 0.30

    # Scene cuts / determinism
    cut_reset: bool = False
    cut_mode: str = "reset"  # reset | inactive
    seed: int = 0

    # Debug/iteration: stop early (0 = no limit)
    max_frames: int = 0

    # Any additional (nested) YAML keys are stored here.
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BoTSORTTeamReIDConfig":
        allowed = {f.name for f in fields(cls)}
        clean: Dict[str, Any] = {k: v for k, v in d.items() if k in allowed}
        extra: Dict[str, Any] = {k: v for k, v in d.items() if k not in allowed}
        cfg = cls(**clean)
        cfg.extra = extra
        return cfg

    def get(self, key_path: str, default: Any = None) -> Any:
        """Fetch a nested key from `extra` via dot-path, e.g. `reacquire.sim_gate`."""

        cur: Any = self.extra
        for part in str(key_path).split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def _expand_path(p: str, base_dir: Path) -> str:
    if not p:
        return p
    p2 = os.path.expandvars(os.path.expanduser(str(p)))
    path = Path(p2)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyYAML is required for config.yaml. Install with: pip install pyyaml") from e

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load_botsort_team_reid_config(config_path: str, base_dir: Optional[str] = None) -> BoTSORTTeamReIDConfig:
    p = Path(config_path)
    base = Path(base_dir) if base_dir else (p.parent if p.parent.exists() else Path.cwd())

    raw = _load_yaml(p)
    cfg = BoTSORTTeamReIDConfig.from_dict(raw)

    # Normalize/expand paths
    cfg.video = _expand_path(cfg.video, base)
    cfg.detector_weights = _expand_path(cfg.detector_weights, base)
    cfg.reid_weights = _expand_path(cfg.reid_weights, base)
    cfg.save_video = _expand_path(cfg.save_video, base) if cfg.save_video else ""
    cfg.save_txt = _expand_path(cfg.save_txt, base) if cfg.save_txt else ""

    if not cfg.device:
        cfg.device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"

    return cfg


def apply_overrides(cfg: BoTSORTTeamReIDConfig, overrides: Dict[str, Any], base_dir: str) -> BoTSORTTeamReIDConfig:
    base = Path(base_dir)

    for k, v in overrides.items():
        if v is None:
            continue
        if not hasattr(cfg, k):
            continue
        setattr(cfg, k, v)

    cfg.video = _expand_path(cfg.video, base)
    cfg.detector_weights = _expand_path(cfg.detector_weights, base)
    cfg.reid_weights = _expand_path(cfg.reid_weights, base)
    cfg.save_video = _expand_path(cfg.save_video, base) if cfg.save_video else ""
    cfg.save_txt = _expand_path(cfg.save_txt, base) if cfg.save_txt else ""

    return cfg


def parse_int_list(csv: str) -> tuple[int, ...]:
    s = str(csv or "").strip()
    if not s:
        return tuple()
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return tuple(out)
