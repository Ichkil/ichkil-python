"""Error types for the ichkil package."""

from __future__ import annotations

__all__ = ["IchkilError", "ChecksumError", "ModelLoadError", "InferenceError"]


class IchkilError(Exception):
    """Base class for all ichkil errors."""


class ChecksumError(IchkilError):
    """The downloaded model file does not match its expected SHA-256."""


class ModelLoadError(IchkilError):
    """The model artifact or its config is missing, invalid or unreadable."""


class InferenceError(IchkilError):
    """Inference failed (bad input shape, runtime error, ...)."""
