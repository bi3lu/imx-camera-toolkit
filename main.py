"""Run the local IMX camera preview server."""

from __future__ import annotations

from packages.api.api import app

HOST = "0.0.0.0"
PORT = 8000


def main() -> None:
    """Start the FastAPI application with Uvicorn.

    The camera is opened by the API lifespan handler after Uvicorn starts. Open
    the printed MJPEG URL in a browser to view the live camera feed.

    Raises:
        RuntimeError: If FastAPI or Uvicorn is unavailable.
    """
    if app is None:
        raise RuntimeError(
            "FastAPI is unavailable. Install the project dependencies with "
            "`uv sync` before starting the camera server."
        )

    try:
        import uvicorn

    except ImportError as error:
        raise RuntimeError(
            "Uvicorn is unavailable. Install the project dependencies with "
            "`uv sync` before starting the camera server."
        ) from error

    print(f"Camera preview: http://localhost:{PORT}/")
    print(f"API documentation: http://localhost:{PORT}/docs")
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
