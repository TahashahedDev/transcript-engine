from typing import ClassVar

from transcript_engine.config.settings import ExportConfig
from transcript_engine.exporters.registry import register
from transcript_engine.models.transcript import Transcript


@register
class SRTExporter:
    """SubRip subtitle format. Standard for video players and captioning tools."""

    format: ClassVar[str] = "srt"
    extension: ClassVar[str] = ".srt"

    def export(self, transcript: Transcript, config: ExportConfig) -> bytes:
        lines: list[str] = []
        for index, segment in enumerate(transcript.segments, start=1):
            display_name = transcript.display_name(segment.speaker_id)
            start_tc = _seconds_to_srt(segment.start)
            end_tc = _seconds_to_srt(segment.end)
            lines.append(str(index))
            lines.append(f"{start_tc} --> {end_tc}")
            lines.append(f"[{display_name}] {segment.text}")
            lines.append("")
        return "\n".join(lines).encode("utf-8")


def _seconds_to_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
