"""Run the local IMX camera preview server."""

from __future__ import annotations

import logging

from imx_camera_toolkit.api import create_app

HOST = "0.0.0.0"
PORT = 8000


class SnapshotAccessLogFilter(logging.Filter):
    """Exclude high-frequency camera snapshot requests from access logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Return whether an access log record should be emitted.

        Args:
            record: Logging record prepared by Uvicorn.

        Returns:
            ``False`` for snapshot access records and ``True`` otherwise.
        """
        return "/api/camera/snapshot" not in record.getMessage()


def configure_access_logging() -> None:
    """Suppress repetitive snapshot requests while preserving other access logs."""
    access_logger = logging.getLogger("uvicorn.access")

    if not any(
        isinstance(log_filter, SnapshotAccessLogFilter)
        for log_filter in access_logger.filters
    ):
        access_logger.addFilter(SnapshotAccessLogFilter())


def main() -> None:
    """Start the FastAPI application with Uvicorn.

    The camera is opened by the API lifespan handler after Uvicorn starts. Open
    the printed MJPEG URL in a browser to view the live camera feed.

    Raises:
        RuntimeError: If FastAPI or Uvicorn is unavailable.
    """
    app = create_app(view_mode="simple")

    try:
        import uvicorn

    except ImportError as error:
        raise RuntimeError(
            "Uvicorn is unavailable. Install the project dependencies with "
            "`uv sync` before starting the camera server."
        ) from error

    configure_access_logging()

    print(f"Camera preview: http://localhost:{PORT}/")
    print(f"API documentation: http://localhost:{PORT}/docs")

    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
