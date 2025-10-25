from __future__ import annotations as _annotations

from pydantic import BaseModel, Field


class EvalPlayer(BaseModel):
    """Player in a Bradley-Terry tournament."""
    idx: int = Field(..., description='unique identifier for the player')
    item: str = Field(..., description='item to be scored')
    score: float | None = Field(default=None, description='Bradley-Terry strength score for the item')
