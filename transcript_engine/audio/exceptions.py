class AudioPreprocessingError(Exception):
    """Raised when audio cannot be preprocessed."""


class FFmpegNotFoundError(AudioPreprocessingError):
    """Raised when ffmpeg binary is not found on PATH."""

    def __init__(self, ffmpeg_path: str = "ffmpeg") -> None:
        super().__init__(
            f"ffmpeg not found at '{ffmpeg_path}'.\n"
            "Install it with:\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html"
        )
