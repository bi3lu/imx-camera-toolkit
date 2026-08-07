"""Exceptions raised by optional production preview transports."""


class ProductionPreviewError(RuntimeError):
    """Base error for production preview setup or operation."""


class ProductionPreviewDependencyError(ProductionPreviewError):
    """Required GStreamer transport plugins are unavailable."""


class ProductionPreviewConfigurationError(ValueError):
    """Production preview settings are invalid or incompatible."""


__all__ = [
    "ProductionPreviewConfigurationError",
    "ProductionPreviewDependencyError",
    "ProductionPreviewError",
]
