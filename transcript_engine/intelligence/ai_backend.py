"""Optional OpenAI-compatible LLM backend for intelligence extraction."""

from __future__ import annotations

import json
import os
from typing import Any

from transcript_engine.logging import get_logger

logger = get_logger(__name__)


class AIBackend:
    """
    Thin wrapper around any OpenAI-compatible chat endpoint.
    Requires TE_AI_BASE_URL and TE_AI_MODEL environment variables.
    All methods degrade gracefully — never raise; return empty results on failure.
    """

    def __init__(self) -> None:
        self._base_url = os.environ.get("TE_AI_BASE_URL")
        self._model = os.environ.get("TE_AI_MODEL")
        self.configured = bool(self._base_url and self._model)

        if not self.configured:
            logger.debug(
                "AI backend not configured (TE_AI_BASE_URL / TE_AI_MODEL not set)."
            )

    def extract_json(self, system_prompt: str, user_text: str) -> dict[str, Any]:
        """
        Call the LLM and parse the response as JSON.
        Returns {} on any failure (missing config, network error, invalid JSON).
        """
        if not self.configured:
            return {}

        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError:
            logger.warning("openai package not installed. Run: pip install openai")
            return {}

        try:
            client = OpenAI(
                base_url=self._base_url,
                api_key=os.environ.get("TE_AI_API_KEY", "sk-no-key"),
            )
            response = client.chat.completions.create(
                model=self._model or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)  # type: ignore[no-any-return]
        except Exception as exc:
            logger.warning(f"AI backend call failed: {exc}")
            return {}

    def complete(
        self, system_prompt: str, user_text: str, max_tokens: int = 1024
    ) -> str:
        """Return a plain-text completion, or '' on failure."""
        if not self.configured:
            return ""

        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError:
            return ""

        try:
            client = OpenAI(
                base_url=self._base_url,
                api_key=os.environ.get("TE_AI_API_KEY", "sk-no-key"),
            )
            response = client.chat.completions.create(
                model=self._model or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning(f"AI backend completion failed: {exc}")
            return ""
