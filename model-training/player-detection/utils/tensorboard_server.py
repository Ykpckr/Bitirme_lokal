"""Utility helpers to launch and manage a TensorBoard server."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

import logging

logger = logging.getLogger(__name__)


class TensorBoardServer:
    """Small wrapper around the TensorBoard CLI process."""

    def __init__(
        self,
        logdir: Path,
        host: str = "127.0.0.1",
        port: int = 6006,
        open_browser: bool = False,
    ):
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.host = host
        self.port = port
        self.open_browser = open_browser
        self._process: Optional[subprocess.Popen[str]] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._url = f"http://{self.host}:{self.port}"

    @property
    def url(self) -> str:
        return self._url

    def start(self):
        if self._process is not None:
            logger.warning(
                "TensorBoard server already running at %s",
                self.url,
            )
            return

        command = [
            sys.executable,
            "-m",
            "tensorboard.main",
            f"--logdir={self.logdir}",
            f"--host={self.host}",
            f"--port={self.port}",
        ]

        logger.info("Starting TensorBoard server: %s", " ".join(command))
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self._stdout_thread = threading.Thread(
            target=self._pipe_logger,
            args=(self._process.stdout, logging.INFO),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._pipe_logger,
            args=(self._process.stderr, logging.ERROR),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

        time.sleep(1.0)
        logger.info("TensorBoard available at %s", self.url)
        if self.open_browser:
            try:
                webbrowser.open(self.url)
            except Exception as exc:
                logger.debug("Failed to open browser for TensorBoard: %s", exc)

    def stop(self):
        if self._process is None:
            return
        logger.info("Stopping TensorBoard server")
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning(
                "TensorBoard server did not terminate gracefully; killing"
            )
            self._process.kill()
        self._process = None

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _pipe_logger(self, pipe, level: int):
        if pipe is None:
            return
        for line in pipe:
            logger.log(level, "[TensorBoard] %s", line.rstrip())
        pipe.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False
