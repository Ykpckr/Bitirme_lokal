"""TensorBoard logging helpers for Ultralytics YOLO training runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover - handle missing tensorboard
    SummaryWriter = None  # type: ignore[misc,assignment]


def _to_float(value: Any) -> Optional[float]:
    """Convert arbitrary numeric-like values to float when possible."""
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())  # type: ignore[call-arg]
        except Exception:  # pragma: no cover - defensive
            return None
    if hasattr(value, "tolist"):
        try:
            data = value.tolist()
            if isinstance(data, (list, tuple)) and data:
                return _to_float(data[0])
            return float(data)
        except Exception:  # pragma: no cover - defensive
            return None
    try:
        return float(value)
    except Exception:
        return None


class YOLOTensorBoardLogger:
    """Lightweight TensorBoard logger wired into Ultralytics callbacks."""

    def __init__(
        self,
        *,
        enabled: bool,
        logdir_path: Path,
        models_root: Path,
        project_name: str,
        hparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.requested = bool(enabled)
        self._enabled = bool(enabled) and SummaryWriter is not None
        self.logdir_path = Path(logdir_path).expanduser().resolve()
        self.models_root = Path(models_root).expanduser().resolve()
        self.project_name = project_name
        self.hparams = hparams or {}
        self.writer: Optional[SummaryWriter] = None  # type: ignore[assignment]
        self.current_run_dir: Optional[Path] = None
        self._attached = False

        if self.requested and SummaryWriter is None:
            logger.warning(
                "TensorBoard logging requested but torch.utils.tensorboard is "
                "unavailable. Install tensorboard to enable writer output."
            )

        self.logdir_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def attach(self, model: Any) -> None:
        """Register callbacks on the provided Ultralytics model."""
        if not self._enabled or self._attached:
            return

        try:
            model.add_callback("on_fit_start", self._on_fit_start)
            model.add_callback("on_fit_epoch_end", self._on_fit_epoch_end)
            model.add_callback("on_fit_end", self._on_fit_end)
            self._attached = True
            logger.info("TensorBoard callbacks attached to YOLO trainer")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to attach TensorBoard callbacks: %s", exc)
            self._attached = False

    def close(self) -> None:
        """Flush and close the underlying SummaryWriter."""
        if not self.writer:
            return
        try:
            self.writer.flush()
            self.writer.close()
            logger.info("TensorBoard writer closed: %s", self.current_run_dir)
        finally:
            self.writer = None
            self.current_run_dir = None

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------
    def _on_fit_start(self, trainer: Any) -> None:
        writer = self._ensure_writer(trainer)
        if not writer:
            return
        if self.hparams:
            try:
                writer.add_text("hparams", str(self.hparams))
            except Exception:  # pragma: no cover - defensive
                logger.debug("Failed to log hparams to TensorBoard")

    def _on_fit_epoch_end(self, trainer: Any) -> None:
        writer = self._ensure_writer(trainer)
        if not writer:
            return

        epoch = getattr(trainer, "epoch", 0)
        step = int(epoch) + 1

        metrics = self._normalize_mapping(getattr(trainer, "metrics", None))
        if metrics:
            self._log_scalars(metrics, step, "metrics", writer)

        losses = self._extract_losses(trainer)
        if losses:
            self._log_scalars(losses, step, "loss", writer)

        lrs = getattr(trainer, "lr", None)
        if isinstance(lrs, (list, tuple)):
            for idx, value in enumerate(lrs):
                val = _to_float(value)
                if val is not None:
                    writer.add_scalar(f"lr/group{idx}", val, step)
        else:
            val = _to_float(lrs)
            if val is not None:
                writer.add_scalar("lr/group0", val, step)

        writer.flush()

    def _on_fit_end(self, trainer: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_writer(self, trainer: Any) -> Optional[SummaryWriter]:  # type: ignore[override]
        if not self._enabled:
            return None

        if self.writer:
            return self.writer

        writer_dir = self._resolve_writer_dir(trainer)
        writer_dir.mkdir(parents=True, exist_ok=True)
        self.current_run_dir = writer_dir

        try:
            self.writer = SummaryWriter(log_dir=str(writer_dir))  # type: ignore[call-arg]
            logger.info("TensorBoard events will be stored in %s", writer_dir)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to initialize SummaryWriter: %s", exc)
            self.writer = None
        return self.writer

    def _resolve_writer_dir(self, trainer: Any) -> Path:
        run_dir = getattr(trainer, "save_dir", None)
        if run_dir:
            run_dir = Path(run_dir)
        else:
            run_dir = self.models_root / self.project_name

        try:
            relative = run_dir.resolve().relative_to(self.models_root)
        except Exception:
            relative = Path(self.project_name)

        return self.logdir_path / relative / "tensorboard"

    def _normalize_mapping(self, payload: Any) -> Dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if hasattr(payload, "results_dict"):
            try:
                data = payload.results_dict
                if isinstance(data, dict):
                    return data
            except Exception:  # pragma: no cover - defensive
                pass
        if hasattr(payload, "as_dict"):
            try:
                data = payload.as_dict()
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {}

    def _extract_losses(self, trainer: Any) -> Dict[str, Any]:
        losses = self._normalize_mapping(getattr(trainer, "label_loss_items", None))
        if losses:
            return losses

        raw_losses = getattr(trainer, "loss_items", None)
        if raw_losses is None:
            return {}

        if hasattr(trainer, "loss_names") and isinstance(trainer.loss_names, (list, tuple)):
            names = trainer.loss_names
        else:
            names = [f"loss_{i}" for i in range(len(raw_losses) if hasattr(raw_losses, "__len__") else 0)]

        values: Any
        if hasattr(raw_losses, "tolist"):
            try:
                values = raw_losses.tolist()
            except Exception:
                values = raw_losses
        else:
            values = raw_losses

        loss_dict: Dict[str, Any] = {}
        if isinstance(values, (list, tuple)):
            for name, value in zip(names, values):
                val = _to_float(value)
                if val is not None:
                    loss_dict[name] = val
        else:
            val = _to_float(values)
            if val is not None:
                loss_dict["loss"] = val

        return loss_dict

    def _log_scalars(
        self,
        scalars: Dict[str, Any],
        step: int,
        prefix: str,
        writer: SummaryWriter,  # type: ignore[type-arg]
    ) -> None:
        for key, value in scalars.items():
            val = _to_float(value)
            if val is None:
                continue
            tag = f"{prefix}/{key}" if prefix else key
            try:
                writer.add_scalar(tag, val, step)
            except Exception:  # pragma: no cover - defensive
                logger.debug("Failed to log %s to TensorBoard", tag)
