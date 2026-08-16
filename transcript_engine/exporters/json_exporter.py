from typing import ClassVar

from transcript_engine.config.settings import ExportConfig
from transcript_engine.exporters.registry import register
from transcript_engine.models.transcript import Transcript


@register
class JSONExporter:
    """
    Full-fidelity export of the Transcript object.

    This is the most important exporter for production use.
    It allows re-export in any format without re-running inference,
    and enables speaker renaming, vocabulary updates, and corrections
    to be applied without touching the ML pipeline.
    """

    format: ClassVar[str] = "json"
    extension: ClassVar[str] = ".json"

    def export(self, transcript: Transcript, config: ExportConfig) -> bytes:
        return transcript.model_dump_json(indent=2).encode("utf-8")
