from typing import Protocol, runtime_checkable

from transcript_engine.config.settings import DiarizationConfig
from transcript_engine.models.audio import PreparedAudio
from transcript_engine.models.pipeline import DiarizationResult
from transcript_engine.models.types import ProgressCallback


@runtime_checkable
class DiarizationEngine(Protocol):
    """
    Protocol satisfied by any speaker diarization backend.
    Implementations: PyannoteEngine.
    Future: NeMoDiarizationEngine, WhisperXDiarizationEngine.
    """

    def diarize(
        self,
        audio: PreparedAudio,
        config: DiarizationConfig,
        on_progress: ProgressCallback | None = None,
    ) -> DiarizationResult: ...
