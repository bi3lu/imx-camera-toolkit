"""Public namespace for the IMX Camera Toolkit library.

Applications should import stable library APIs from this namespace instead of
the repository-internal ``packages`` package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .camera import (
    Camera,
    CameraConfig,
    CameraDependencyError,
    CameraFrame,
    CameraProfile,
    CameraProfileStatus,
    CameraStats,
    Frame,
    get_camera_profile,
    list_camera_profiles,
)

if TYPE_CHECKING:
    from .preview import CameraPreview as CameraPreview
    from .preview import PreviewServer as PreviewServer
    from .preview import PreviewSource as PreviewSource
    from .preview import create_preview_app as create_preview_app
    from .preview import preview as preview

__version__ = "0.3.1"

__all__ = [
    "Camera",
    "CameraConfig",
    "CameraProfile",
    "CameraProfileStatus",
    "CameraStats",
    "CameraDependencyError",
    "CameraFrame",
    "Frame",
    "get_camera_profile",
    "list_camera_profiles",
    "__version__",
]


def __getattr__(name: str) -> Any:
    """Lazily import browser-preview helpers and their optional dependencies.

    Raises:
        AttributeError: If ``name`` is not a public namespace member.
        ImportError: If preview helpers are requested without the ``preview``
            optional dependency group.
    """
    if name not in {
        "CameraPreview",
        "PreviewServer",
        "PreviewSource",
        "create_preview_app",
        "preview",
    }:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        from .preview import (
            CameraPreview,
            PreviewServer,
            PreviewSource,
            create_preview_app,
            preview,
        )

    except ImportError as error:
        raise ImportError(
            "Browser preview support is optional. Install it with "
            '`uv add "imx-camera-toolkit[preview]"`. '
        ) from error

    globals().update(
        {
            "CameraPreview": CameraPreview,
            "PreviewServer": PreviewServer,
            "PreviewSource": PreviewSource,
            "create_preview_app": create_preview_app,
            "preview": preview,
        }
    )
    __all__.extend(
        (
            "CameraPreview",
            "PreviewServer",
            "PreviewSource",
            "create_preview_app",
            "preview",
        )
    )
    return globals()[name]
