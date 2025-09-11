from pydantic import BaseModel, Field

from pydantic_ai.messages import ModelMessage

MessageHistory = list[ModelMessage]


class OutlineNode(BaseModel):
    section_id: str = Field(repr=False)
    title: str
    description: str | None
    requires_research: bool
    children: list['OutlineNode'] = Field(default_factory=list)


OutlineNode.model_rebuild()


class Outline(BaseModel):
    """TODO: This should not involve a recursive type — some vendors don't do a good job generating recursive models."""

    root: OutlineNode
