from __future__ import annotations

from pydantic_ai.models import Model
from pydantic_ai.models.wrapper import WrapperModel


class HatchetModel(WrapperModel):
    """A wrapper for Model that integrates with DBOS, turning request and request_stream to DBOS steps."""

    def __init__(
        self,
        model: Model,
    ):
        super().__init__(model)

        pass
