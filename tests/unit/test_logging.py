"""
Regression tests for %-style argument interpolation in the project logger.

The codebase mixes f-string and stdlib %-style logging calls. Loguru formats
with str.format, so %-style calls used to log the literal format string and
silently drop their arguments — losing exactly the VRAM / chunk-size / OOM
values the GPU paths log for diagnosis.
"""

from __future__ import annotations

from transcript_engine.logging import get_logger
from transcript_engine.logging.setup import _PercentStyleLogger


class TestPercentStyleInterpolation:
    def test_percent_args_are_interpolated(self) -> None:
        rendered = _PercentStyleLogger._render(
            "[GPU] %s | %.1f GB free | CC %d.%d", ("RTX 3060 Ti", 6.25, 8, 6)
        )
        assert rendered == "[GPU] RTX 3060 Ti | 6.2 GB free | CC 8.6"

    def test_message_without_args_passes_through_unchanged(self) -> None:
        # f-string call sites arrive already formatted, with no extra args.
        assert _PercentStyleLogger._render("already formatted: 42 words", ()) == (
            "already formatted: 42 words"
        )

    def test_literal_percent_without_args_is_not_mangled(self) -> None:
        assert _PercentStyleLogger._render("CPU at 95% utilization", ()) == (
            "CPU at 95% utilization"
        )

    def test_malformed_format_does_not_raise(self) -> None:
        """A logging call must never take down the pipeline it is reporting on."""
        rendered = _PercentStyleLogger._render("bad %d", ("not-an-int",))
        assert "bad %d" in rendered
        assert "not-an-int" in rendered

    def test_too_few_args_does_not_raise(self) -> None:
        rendered = _PercentStyleLogger._render("%s and %s", ("only-one",))
        assert "only-one" in rendered


class TestLoggerInterface:
    def test_get_logger_exposes_standard_levels(self) -> None:
        log = get_logger("test")
        for level in ("debug", "info", "warning", "error", "exception"):
            assert callable(getattr(log, level))

    def test_unwrapped_attributes_delegate_to_loguru(self) -> None:
        log = get_logger("test")
        # bind() is not wrapped explicitly; __getattr__ must forward it.
        assert callable(log.bind)

    def test_logging_calls_do_not_raise(self) -> None:
        log = get_logger("test")
        log.info("value=%s", 1)
        log.warning("value=%.2f", 3.14159)
        log.error("no args at all")
