class MissingHFTokenError(Exception):
    """Raised when pyannote requires a HuggingFace token that is not configured."""

    def __init__(self, model_id: str) -> None:
        super().__init__(
            f"HuggingFace token required to download '{model_id}'.\n\n"
            "1. Create a free token at: https://huggingface.co/settings/tokens\n"
            f"2. Accept the model terms at: https://huggingface.co/{model_id}\n"
            "3. Add to your .env file:\n"
            "       TE_HF_TOKEN=hf_your_token_here\n"
            "   or set the environment variable:\n"
            "       export TE_HF_TOKEN=hf_your_token_here"
        )


class DiarizationError(Exception):
    """Raised when diarization fails."""
