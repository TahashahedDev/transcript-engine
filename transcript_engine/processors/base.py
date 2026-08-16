from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from transcript_engine.models.correction import CorrectionRecord
    from transcript_engine.models.transcript import Transcript


@dataclass
class ProcessorContext:
    """
    Thin context object injected into every processor.
    Accumulates correction records produced during vocabulary processing.
    """

    config: dict[str, Any] = field(default_factory=dict)
    corrections: list[CorrectionRecord] = field(default_factory=list)
    profile_name: str = "generic"


@runtime_checkable
class Processor(Protocol):
    """
    Protocol for all transcript processors.

    Every processor:
    - receives an immutable Transcript
    - returns a new Transcript (never mutates the input)
    - is identified by a unique class-level name string
    - is stateless after __init__ (config loaded in __init__, not process())
    """

    name: ClassVar[str]

    def process(self, transcript: Transcript, ctx: ProcessorContext) -> Transcript: ...
