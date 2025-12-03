"""Lightweight training report generation for the new YOLO pipeline."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TrainingReportGenerator:
    """Produces simple CSV + HTML summaries after training."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_reports(
        self,
        training_results: Any,
        config: Dict[str, Any],
        model_path: Optional[Path] = None,
        dataset_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        timestamp = datetime.now().strftime("%H%M-%d%m%Y")
        summary = {
            "timestamp": timestamp,
            "model_path": str(model_path) if model_path else None,
            "config": config,
            "dataset_stats": dataset_stats or {},
        }

        excel_path = self._write_csv(summary)
        html_path = self._write_html(summary)
        meta_path = self.output_dir / f"training_report_{timestamp}.json"
        meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return {"excel": excel_path, "html": html_path, "meta": meta_path}

    def _write_csv(self, summary: Dict[str, Any]) -> Path:
        timestamp = summary["timestamp"]
        csv_path = self.output_dir / f"training_report_{timestamp}.csv"
        lines = ["section,key,value"]

        config = summary.get("config", {})
        yolo_params = config.get("yolo_params", {})
        for key, value in sorted(yolo_params.items()):
            lines.append(f"yolo_params,{key},{value}")

        dataset_stats = summary.get("dataset_stats", {})
        for key, value in sorted(dataset_stats.items()):
            lines.append(f"dataset_stats,{key},{value}")

        if summary.get("model_path"):
            lines.append(f"artifacts,best_model,{summary['model_path']}")

        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("CSV summary written: %s", csv_path)
        return csv_path

    def _write_html(self, summary: Dict[str, Any]) -> Path:
        timestamp = summary["timestamp"]
        html_path = self.output_dir / f"training_report_{timestamp}.html"
        config = summary.get("config", {})
        yolo_params = config.get("yolo_params", {})
        dataset_stats = summary.get("dataset_stats", {})

        def _rows(data: Dict[str, Any]) -> str:
            return "".join(
                f"<tr><td>{key}</td><td>{value}</td></tr>" for key, value in data.items()
            )

        html = f"""
        <!DOCTYPE html>
        <html lang=\"en\">
        <head>
            <meta charset=\"utf-8\" />
            <title>YOLO Training Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 2rem; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
                th, td {{ border: 1px solid #ddd; padding: 0.5rem; text-align: left; }}
                th {{ background: #004d8f; color: white; }}
            </style>
        </head>
        <body>
            <h1>YOLO Training Report</h1>
            <p>Generated at: {timestamp}</p>
            <h2>Model</h2>
            <p>{summary.get('model_path') or 'N/A'}</p>
            <h2>Training Parameters</h2>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                {_rows(yolo_params)}
            </table>
            <h2>Dataset Statistics</h2>
            <table>
                <tr><th>Statistic</th><th>Value</th></tr>
                {_rows(dataset_stats)}
            </table>
        </body>
        </html>
        """
        html_path.write_text(html, encoding="utf-8")
    logger.info("HTML report written: %s", html_path)
    return html_path

class EnhancedReportGenerator:
    """Creates timestamped directories and delegates to TrainingReportGenerator."""

    def __init__(self, reports_root: str, project_name: str):
        self.reports_root = Path(reports_root)
        self.project_dir = self.reports_root / project_name

    def generate_training_reports(
        self,
        training_data: Dict[str, Any],
        baseline_metrics: Optional[Dict[str, Any]] = None,
        final_metrics: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, str]:
        timestamp = datetime.now().strftime("%H%M-%d%m%Y")
        report_dir = self.project_dir / timestamp
        report_dir.mkdir(parents=True, exist_ok=True)

        merged_data = training_data.copy()
        if baseline_metrics:
            merged_data.setdefault("config", {})["baseline_metrics"] = baseline_metrics
        if final_metrics:
            merged_data["final_metrics"] = final_metrics

        generator = TrainingReportGenerator(report_dir)
        outputs = generator.generate_reports(
            training_results=None,
            config=merged_data.get("config", {}),
            model_path=Path(merged_data.get("model_path")) if merged_data.get("model_path") else None,
            dataset_stats=merged_data.get("dataset_stats", {}),
        )

    return str(outputs["excel"]), str(outputs["html"])


def create_report_generator(output_dir: Path) -> TrainingReportGenerator:
    return TrainingReportGenerator(output_dir)
