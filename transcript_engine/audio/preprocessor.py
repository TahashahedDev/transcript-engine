from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from types import TracebackType

from transcript_engine.audio.exceptions import (
    AudioPreprocessingError,
    FFmpegNotFoundError,
)
from transcript_engine.config.settings import AudioConfig
from transcript_engine.logging import get_logger
from transcript_engine.models.audio import PreparedAudio

logger = get_logger(__name__)


class AudioPreprocessor:
    """
    Converts any audio/video file to 16 kHz mono WAV.
    Use as a context manager to ensure temp files are cleaned up.

    with AudioPreprocessor(config) as preprocessor:
        audio = preprocessor.prepare(Path("meeting.mp4"))
    # temp dir deleted here regardless of success or failure
    """

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._tmp_dir: Path | None = None
        self._validate_ffmpeg()

    def __enter__(self) -> AudioPreprocessor:
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="te_audio_"))
        logger.debug(f"Created temp audio dir: {self._tmp_dir}")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._tmp_dir and self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            logger.debug(f"Cleaned up temp audio dir: {self._tmp_dir}")

    def prepare(self, input_path: Path) -> PreparedAudio:
        """
        Convert input_path to 16 kHz mono WAV.
        Must be called inside a context manager.
        """
        if self._tmp_dir is None:
            raise RuntimeError("AudioPreprocessor must be used as a context manager.")

        if not input_path.exists():
            raise AudioPreprocessingError(f"Input file not found: {input_path}")

        output_path = self._tmp_dir / f"{input_path.stem}_prepared.wav"
        logger.info(f"Preprocessing audio: {input_path.name}")

        self._run_ffmpeg(input_path, output_path)
        duration = self._get_duration(output_path)

        logger.info(
            f"Audio prepared: {duration:.1f}s at {self._config.target_sample_rate} Hz"
        )

        return PreparedAudio(
            path=output_path,
            duration=duration,
            original_path=input_path,
            original_format=input_path.suffix.lstrip(".").lower(),
            sample_rate=self._config.target_sample_rate,
            channels=1,
        )

    def _run_ffmpeg(self, input_path: Path, output_path: Path) -> None:
        cmd = [
            self._ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-ar",
            str(self._config.target_sample_rate),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            "-vn",
            str(output_path),
        ]

        logger.debug(f"ffmpeg command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except FileNotFoundError as exc:
            raise FFmpegNotFoundError(self._ffmpeg) from exc
        except subprocess.TimeoutExpired as exc:
            raise AudioPreprocessingError("ffmpeg timed out after 10 minutes.") from exc

        if result.returncode != 0:
            raise AudioPreprocessingError(
                f"ffmpeg failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
            )

    def _get_duration(self, wav_path: Path) -> float:
        # Read the WAV header directly — no subprocess, no audio decode.
        # torchaudio.info() reads num_frames and sample_rate from the RIFF header.
        try:
            import torchaudio  # noqa: PLC0415

            info = torchaudio.info(str(wav_path))
            duration = float(info.num_frames) / float(info.sample_rate)
            if duration <= 0:
                raise AudioPreprocessingError(f"Audio has zero duration: {wav_path}")
            return duration
        except AudioPreprocessingError:
            raise
        except Exception as exc:
            raise AudioPreprocessingError(
                f"Could not read audio duration from {wav_path}: {exc}"
            ) from exc

    def _validate_ffmpeg(self) -> None:
        resolved = shutil.which(self._config.ffmpeg_path)
        if resolved is None:
            raise FFmpegNotFoundError(self._config.ffmpeg_path)
        # Pin the resolved absolute path so subprocess finds it regardless of PATH.
        self._ffmpeg_bin = resolved

    @property
    def _ffmpeg(self) -> str:
        return getattr(self, "_ffmpeg_bin", self._config.ffmpeg_path)
