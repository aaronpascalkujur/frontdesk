from .server import DEFAULT_HOST, DEFAULT_PORT, UIServer

__all__ = ["UIServer", "DEFAULT_HOST", "DEFAULT_PORT", "run"]


def run(*args, **kwargs):
    """Start the browser frontend. Imported lazily — loading it pulls in the
    speech models, which console mode should not pay for."""
    from .launch import run as _run

    return _run(*args, **kwargs)
