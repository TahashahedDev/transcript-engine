from __future__ import annotations

import sys
from pathlib import Path

import loguru
from loguru import logger

# Suppress loguru's default stderr sink until configure_logging is called.
# Without this, module-level @register decorators emit DEBUG noise on every
# `te --help` invocation.
logger.remove()
logger.add(sys.stderr, level="WARNING")


def configure_logging(log_level: str = "INFO", log_file: Path | None = None) -> None:
    """
    Configure Loguru for the process. Call once at startup from the interface layer.
    Removes the default handler and installs structured handlers.
    """
    logger.remove()

    logger.add(
        sys.stderr,
        level=log_level.upper(),
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            level=log_level.upper(),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
            rotation="10 MB",
            retention="30 days",
            encoding="utf-8",
        )


class _PercentStyleLogger:
    """
    Loguru logger that also understands stdlib %-style arguments.

    Loguru formats with str.format ("{}"), not "%s". A stdlib-style call like

        logger.info("chunk=%.0fs free=%.1f GB", chunk_s, vram)

    therefore logged the literal "chunk=%.0fs free=%.1f GB" and silently
    dropped both values — the codebase mixes %-style and f-string calls, and
    45 of the %-style ones were losing their data, concentrated in the GPU /
    OOM / chunking paths where those numbers matter most for diagnosis.

    Interpolating here fixes every such call at one point instead of rewriting
    (and re-breaking) each call site. f-string calls pass through untouched
    because they arrive with no extra args.

    opt(depth=1) keeps loguru's "{name}:{line}" pointing at the real caller
    rather than at this wrapper.
    """

    __slots__ = ("_log",)

    def __init__(self, bound: loguru.Logger) -> None:
        self._log = bound

    @staticmethod
    def _render(message: object, args: tuple[object, ...]) -> str:
        if not args:
            return str(message)
        try:
            return str(message) % args
        except (TypeError, ValueError):
            # Malformed format string: keep the data rather than raising from
            # a log call, which must never break the pipeline.
            return f"{message} {args!r}"

    def debug(self, message: object, *args: object, **kw: object) -> None:
        self._log.opt(depth=1).debug(self._render(message, args), **kw)

    def info(self, message: object, *args: object, **kw: object) -> None:
        self._log.opt(depth=1).info(self._render(message, args), **kw)

    def warning(self, message: object, *args: object, **kw: object) -> None:
        self._log.opt(depth=1).warning(self._render(message, args), **kw)

    def error(self, message: object, *args: object, **kw: object) -> None:
        self._log.opt(depth=1).error(self._render(message, args), **kw)

    def exception(self, message: object, *args: object, **kw: object) -> None:
        self._log.opt(depth=1).exception(self._render(message, args), **kw)

    def __getattr__(self, item: str) -> object:
        # Anything not wrapped above (bind, contextualize, ...) goes straight
        # to loguru unchanged.
        return getattr(self._log, item)


def get_logger(name: str) -> _PercentStyleLogger:
    """Return a logger bound to the given module name."""
    return _PercentStyleLogger(logger.bind(name=name))
