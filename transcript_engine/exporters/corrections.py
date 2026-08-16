from __future__ import annotations

from transcript_engine.models.correction import CorrectionRecord


def export_corrections(
    corrections: list[CorrectionRecord],
    profile_name: str,
    audio_name: str,
) -> str:
    """Render a correction report as Markdown."""
    if not corrections:
        return (
            "# Correction Report\n\n"
            f"**Profile:** {profile_name}  \n"
            "**Total corrections:** 0\n\n"
            "No corrections applied.\n"
        )

    lines = [
        "# Correction Report",
        "",
        f"**Profile:** {profile_name}  ",
        f"**Audio:** {audio_name}  ",
        f"**Total corrections:** {len(corrections)}",
        "",
        "| # | Original | Replacement | Reason | Confidence |",
        "|---|----------|-------------|--------|------------|",
    ]
    for i, rec in enumerate(corrections, 1):
        lines.append(
            f"| {i} | {rec.original} | {rec.replacement} "
            f"| {rec.reason} | {rec.confidence:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)
