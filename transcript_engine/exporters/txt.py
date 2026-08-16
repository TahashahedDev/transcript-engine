from typing import ClassVar

from transcript_engine.config.settings import ExportConfig
from transcript_engine.exporters.registry import register
from transcript_engine.models.transcript import Transcript


@register
class TxtExporter:
    """Plain text export — speaker name followed by their text, one turn per paragraph."""

    format: ClassVar[str] = "txt"
    extension: ClassVar[str] = ".txt"

    def export(self, transcript: Transcript, config: ExportConfig) -> bytes:
        lines: list[str] = []
        for segment in transcript.segments:
            display_name = transcript.display_name(segment.speaker_id)
            lines.append(f"{display_name}: {segment.text}")
            lines.append("")
        content = "\n".join(lines).rstrip() + "\n"
        return content.encode("utf-8")
