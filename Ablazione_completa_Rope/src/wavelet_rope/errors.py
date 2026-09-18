"""Errors raised by the standalone runtime infrastructure."""


class WaveletRopeError(RuntimeError):
    """Base error for runtime contract violations."""


class ConfigError(WaveletRopeError):
    """The operational configuration or architecture spec is invalid."""


class RequiredValueError(ConfigError):
    """One or more explicit REQUIRED placeholders are unresolved."""


class FingerprintMismatchError(WaveletRopeError):
    """A persisted artifact belongs to a different experiment."""


class CheckpointError(WaveletRopeError):
    """A checkpoint is incomplete or violates the resume contract."""


class ActiveRunError(WaveletRopeError):
    """The run directory already has an active lease."""

