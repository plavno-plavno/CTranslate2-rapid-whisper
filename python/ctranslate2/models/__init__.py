"""Whisper speech recognition model."""

try:
    from ctranslate2._ext import (
        Whisper,
        WhisperAlignmentResult,
        WhisperDecoderState,
        WhisperGenerationResult,
        WhisperGenerationResultAsync,
    )
except ImportError as e:
    # Allow using the Python package without the compiled extension.
    if "No module named" in str(e):
        pass
    else:
        raise
