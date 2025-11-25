"""Utility helpers to launch and manage a TensorBoard server."""

from __future__ import annotations

import socket
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
        auto_port: bool = True,
        port_search: int = 10,
    ):
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.host = host
        self.port = port
        self.open_browser = open_browser
        self.auto_port = auto_port
        self.port_search = max(1, int(port_search))
        self._process: Optional[subprocess.Popen[str]] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._url: Optional[str] = None

    @property
    def url(self) -> str:
        if self._url:
            return self._url
        return f"http://{self.host}:{self.port}"

    def start(self):
        if self._process is not None:
            logger.warning(
                "TensorBoard server already running at %s",
                self.url,
            )
            return

        last_error: Optional[Exception] = None
        for candidate_port in self._candidate_ports():
            if not self._is_port_available(candidate_port):
                logger.info(
                    "TensorBoard port %s is busy, trying another port",
                    candidate_port,
                )
                continue

            self.port = candidate_port
            self._url = f"http://{self.host}:{self.port}"
            command = [
                sys.executable,
                "-m",
                "tensorboard.main",
                f"--logdir={self.logdir}",
                f"--host={self.host}",
                f"--port={self.port}",
            ]

            logger.info("Starting TensorBoard server: %s", " ".join(command))
            try:
                self._process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
            except Exception as exc:
                last_error = exc
                self._process = None
                continue

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
            if self._process.poll() is None:
                logger.info("TensorBoard available at %s", self.url)
                if self.open_browser:
                    try:
                        webbrowser.open(self.url)
                    except Exception as exc:
                        logger.debug("Failed to open browser for TensorBoard: %s", exc)
                return

            last_error = RuntimeError(
                f"TensorBoard exited early with code {self._process.returncode}"
            )
            self._cleanup_process()

        if last_error is None:
            last_error = RuntimeError("All candidate ports are busy")
        raise last_error

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
        self._cleanup_process()

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _pipe_logger(self, pipe, level: int):
        if pipe is None:
            return
        for line in pipe:
            if self._should_suppress_line(line):
                continue
            adjusted_level = self._classify_log_level(line, level)
            logger.log(adjusted_level, "[TensorBoard] %s", line.rstrip())
        pipe.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _candidate_ports(self):
        yield self.port
        if not self.auto_port:
            return
        for offset in range(1, self.port_search + 1):
            yield self.port + offset

    def _is_port_available(self, port: int) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, port))
        except OSError:
            return False
        finally:
            sock.close()
        return True

    def _cleanup_process(self):
        process = self._process
        self._process = None
        if process and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass
        self._stdout_thread = None
        self._stderr_thread = None
        self._url = None

    def _classify_log_level(self, line: str, default_level: int) -> int:
        text = line.lower()
        if "pkg_resources" in text:
            return logging.WARNING
        if "tensorflow installation not found" in text:
            return logging.WARNING
        if "userwarning" in text and default_level >= logging.WARNING:
            return logging.WARNING
        if "error" in text:
            return logging.ERROR
        return default_level

    def _should_suppress_line(self, line: str) -> bool:
        text = line.strip().lower()
        if text == "import pkg_resources":
            return True
        return False
