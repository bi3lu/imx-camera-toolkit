"""Small Jetson resource samplers used by opt-in hardware benchmarks."""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
from statistics import fmean


class TegrastatsSampler:
    """Collect average GR3D utilization from one owned ``tegrastats`` child."""

    _GPU_PATTERN = re.compile(r"\bGR3D_FREQ\s+(?P<percent>[0-9]+(?:[.][0-9]+)?)%")

    def __init__(self, interval_ms: int = 100) -> None:
        """Configure a sampler without requiring Jetson tools at import time."""
        if (
            isinstance(interval_ms, bool)
            or not isinstance(interval_ms, int)
            or interval_ms <= 0
        ):
            raise ValueError("interval_ms must be a positive integer")
        self._interval_ms = interval_ms
        self._lock = threading.Lock()
        self._samples: list[float] = []
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._started = False

    @property
    def sample_count(self) -> int:
        """Number of parsed GPU utilization samples."""
        with self._lock:
            return len(self._samples)

    @property
    def average_gpu_percent(self) -> float | None:
        """Average GR3D utilization, or ``None`` outside a Jetson runtime."""
        with self._lock:
            return None if not self._samples else fmean(self._samples)

    def start(self) -> None:
        """Start a private foreground tegrastats process when available."""
        if self._started:
            raise RuntimeError("tegrastats sampler cannot be restarted")

        self._started = True
        executable = shutil.which("tegrastats")

        if executable is None:
            return

        try:
            process = subprocess.Popen(
                [executable, "--interval", str(self._interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError:
            return
        self._process = process
        self._thread = threading.Thread(
            target=self._read_output,
            args=(process,),
            name="imx-tegrastats-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> float | None:
        """Stop only the owned process and return its average GPU usage."""
        process = self._process

        if process is not None and process.poll() is None:
            process.terminate()

            try:
                process.wait(timeout=1.0)

            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)

        thread = self._thread

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

        self._process = None
        self._thread = None
        return self.average_gpu_percent

    @classmethod
    def parse_gpu_percent(cls, line: str) -> float | None:
        """Parse one documented ``GR3D_FREQ X%`` tegrastats field."""
        match = cls._GPU_PATTERN.search(line)
        return None if match is None else float(match.group("percent"))

    def _read_output(self, process: subprocess.Popen[str]) -> None:
        """Consume child output without retaining complete telemetry lines."""
        if process.stdout is None:
            return

        for line in process.stdout:
            sample = self.parse_gpu_percent(line)

            if sample is not None:
                with self._lock:
                    self._samples.append(sample)


__all__ = ["TegrastatsSampler"]
