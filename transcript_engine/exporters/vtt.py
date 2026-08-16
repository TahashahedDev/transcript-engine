from typing import ClassVar

from transcript_engine.config.settings import ExportConfig
from transcript_engine.exporters.registry import register
from transcript_engine.models.transcript import Transcript


@register
class VTTExporter:
    """WebVTT format for HTML5 video and browser-based players."""

    format: ClassVar[str] = "vtt"
    extension: ClassVar[str] = ".vtt"

    def export(self, transcript: Transcript, config: ExportConfig) -> bytes:
        lines: list[str] = ["WEBVTT", ""]
        for index, segment in enumerate(transcript.segments, start=1):
            display_name = transcript.display_name(segment.speaker_id)
            start_tc = _seconds_to_vtt(segment.start)
            end_tc = _seconds_to_vtt(segment.end)
            lines.append(f"cue-{index}")
            lines.append(f"{start_tc} --> {end_tc}")
            lines.append(f"<v {display_name}>{segment.text}")
            lines.append("")
        return "\n".join(lines).encode("utf-8")


def _seconds_to_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
