from typing import Protocol, runtime_checkable

from transcript_engine.config.settings import TranscriptionConfig
from transcript_engine.models.audio import PreparedAudio
from transcript_engine.models.pipeline import RawTranscription
from transcript_engine.models.types import ProgressCallback


@runtime_checkable
class TranscriptionEngine(Protocol):
    """
    Protocol satisfied by any transcription backend.
    Implementations: WhisperXEngine.
    Future: WhisperCppEngine, ParakeetEngine, CloudWhisperEngine.
    """

    def transcribe(
        self,
        audio: PreparedAudio,
        config: TranscriptionConfig,
        on_progress: ProgressCallback | None = None,
    ) -> RawTranscription: ...
