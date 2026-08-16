"""
Transcript Engine — Production-quality local AI meeting transcription.

Public SDK surface:
    from transcript_engine import TranscriptEngine, Transcript
"""

from transcript_engine.models.transcript import Segment, Transcript, Word

__version__ = "0.1.0"
__all__ = ["Transcript", "Segment", "Word", "__version__"]
