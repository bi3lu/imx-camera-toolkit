"""Exceptions raised when required camera runtime dependencies are absent."""


class CameraDependencyError(RuntimeError):
    """Raised when the local JetPack camera runtime is unavailable."""
