"""A decision model for checking that `decide` spans nest under `chat` through a durable engine's unit.

The model answers in memory, so the tests need no cassette: what they check is where the span lands and what
content policy it was given, not what a real backend answers.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from pydantic_ai.models.decision import (
    DecisionModel,
    DecisionModelSettings,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
)


class ShipIt(BaseModel):
    """Decide whether a change can ship."""

    ship: bool = Field(description='Can this change ship?')


class ShipItDecisionModel(DecisionModel[None]):
    """Answers `ShipIt`'s one question with a confident yes."""

    @property
    def model_name(self) -> str:
        return 'ship-it'

    @property
    def system(self) -> str:
        return 'test-decisions'

    async def decide(self, request: DecisionRequest, model_settings: DecisionModelSettings) -> DecisionResponse:
        return DecisionResponse(answers={'ship': NoulAnswer(noul=0.9)}, model_name='ship-it')


def decide_span_lineage(spans: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    """The one `decide` span's ancestors' names, innermost first, and its attributes."""
    by_id = {span['context']['span_id']: span for span in spans}
    [decide] = [span for span in spans if span['name'] == 'decide ship-it']
    lineage: list[str] = []
    span = decide
    while parent := span['parent']:
        span = by_id[parent['span_id']]
        lineage.append(span['name'])
    return lineage, decide['attributes']
