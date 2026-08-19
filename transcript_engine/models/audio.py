from pathlib import Path

from pydantic import BaseModel


class PreparedAudio(BaseModel, frozen=True):
    """
    The normalized audio artifact produced by AudioPreprocessor.
    Always 16 kHz, mono, PCM WAV — the exact format Whisper expects.
    """

    path: Path
    duration: float
    original_path: Path
    original_format: str
    sample_rate: int = 16_000
    channels: int = 1
