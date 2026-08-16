from typing import ClassVar, Protocol, runtime_checkable

from transcript_engine.config.settings import ExportConfig
from transcript_engine.models.transcript import Transcript


@runtime_checkable
class Exporter(Protocol):
    """
    Protocol for all transcript exporters.

    Receives a Transcript, returns bytes.
    Never writes files — the caller is responsible for writing.
    Never mutates the Transcript.
    """

    format: ClassVar[str]
    extension: ClassVar[str]

    def export(self, transcript: Transcript, config: ExportConfig) -> bytes: ...
