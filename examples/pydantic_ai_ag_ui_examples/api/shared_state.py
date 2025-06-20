"""Shared State feature."""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated

from ag_ui.core import EventType, RunAgentInput, StateSnapshotEvent
from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_ai_ag_ui.consts import SSE_CONTENT_TYPE
from pydantic_ai_ag_ui.deps import StateDeps

from .agent import AGUIAgent

if TYPE_CHECKING:  # pragma: no cover
    from pydantic_ai import RunContext

_LOGGER: logging.Logger = logging.getLogger(__name__)


class SkillLevel(StrEnum):
    """The level of skill required for the recipe."""

    BEGINNER = 'Beginner'
    INTERMEDIATE = 'Intermediate'
    ADVANCED = 'Advanced'


class SpecialPreferences(StrEnum):
    """Special preferences for the recipe."""

    HIGH_PROTEIN = 'High Protein'
    LOW_CARB = 'Low Carb'
    SPICY = 'Spicy'
    BUDGET_FRIENDLY = 'Budget-Friendly'
    ONE_POT_MEAL = 'One-Pot Meal'
    VEGETARIAN = 'Vegetarian'
    VEGAN = 'Vegan'


class CookingTime(StrEnum):
    """The cooking time of the recipe."""

    FIVE_MIN = '5 min'
    FIFTEEN_MIN = '15 min'
    THIRTY_MIN = '30 min'
    FORTY_FIVE_MIN = '45 min'
    SIXTY_PLUS_MIN = '60+ min'


class Ingredient(BaseModel):
    """A class representing an ingredient in a recipe."""

    icon: str = Field(
        default='ingredient',
        description="The icon emoji (not emoji code like '\x1f35e', but the actual emoji like 🥕) of the ingredient",
    )
    name: str
    amount: str


class Recipe(BaseModel):
    """A class representing a recipe."""

    skill_level: SkillLevel = Field(
        description='The skill level required for the recipe'
    )
    special_preferences: list[SpecialPreferences] = Field(
        description='Any special preferences for the recipe'
    )
    cooking_time: CookingTime = Field(description='The cooking time of the recipe')
    ingredients: list[Ingredient] = Field(description='Ingredients for the recipe')
    instructions: list[str] = Field(description='Instructions for the recipe')


class RecipeSnapshot(BaseModel):
    """A class representing the state of the recipe."""

    recipe: Recipe = Field(description='The current state of the recipe')


router: APIRouter = APIRouter(prefix='/shared_state')
agui: AGUIAgent[StateDeps[RecipeSnapshot]] = AGUIAgent(
    deps_type=StateDeps[RecipeSnapshot]
)


@agui.agent.tool_plain
def display_recipe(recipe: Recipe) -> StateSnapshotEvent:
    """Display the recipe to the user.

    Args:
        recipe: The recipe to display.

    Returns:
        StateSnapshotEvent containing the recipe snapshot.
    """
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot={'recipe': recipe},
    )


@agui.agent.instructions
def recipe_instructions(ctx: RunContext[StateDeps[RecipeSnapshot]]) -> str:
    """Instructions for the recipe generation agent.

    Args:
        ctx: The run context containing recipe state information.

    Returns:
        Instructions string for the recipe generation agent.
    """
    _LOGGER.info('recipe instructions recipe=%s', ctx.deps.state.recipe)

    return f"""You are a helpful assistant for creating recipes.

IMPORTANT:
- Create a complete recipe using the existing ingredients
- Append new ingredients to the existing ones
- Use the `display_recipe` tool to present the recipe to the user
- Do NOT repeat the recipe in the message, use the tool instead

Once you have created the updated recipe and displayed it to the user,
summarise the changes in one sentence, don't describe the recipe in
detail or send it as a message to the user.

The structure of a recipe is as follows:

{json.dumps(Recipe.model_json_schema(), indent=2)}

The current state of the recipe is:

{ctx.deps.state.recipe.model_dump_json(indent=2)}
"""


@router.post('')
async def handler(
    input_data: RunAgentInput, accept: Annotated[str, Header()] = SSE_CONTENT_TYPE
) -> StreamingResponse:
    """Endpoint to handle AG-UI protocol requests and stream responses.

    Args:
        input_data: The AG-UI run input.
        accept: The Accept header to specify the response format.

    Returns:
        A streaming response with event-stream media type.
    """
    return StreamingResponse(
        agui.adapter.run(input_data, accept, deps=StateDeps(state_type=RecipeSnapshot)),
        media_type=SSE_CONTENT_TYPE,
    )
