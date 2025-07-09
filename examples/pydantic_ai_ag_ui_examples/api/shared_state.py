"""Shared State feature."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from ag_ui.core import EventType, StateSnapshotEvent
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.ag_ui import AGUIApp, StateDeps

if TYPE_CHECKING:  # pragma: no cover
    from pydantic_ai import RunContext


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
        default=SkillLevel.BEGINNER,
        description='The skill level required for the recipe',
    )
    special_preferences: list[SpecialPreferences] = Field(
        default_factory=lambda: list[SpecialPreferences](),
        description='Any special preferences for the recipe',
    )
    cooking_time: CookingTime = Field(
        default=CookingTime.FIVE_MIN, description='The cooking time of the recipe'
    )
    ingredients: list[Ingredient] = Field(
        default_factory=lambda: list[Ingredient](),
        description='Ingredients for the recipe',
    )
    instructions: list[str] = Field(
        default_factory=lambda: list[str](), description='Instructions for the recipe'
    )


class RecipeSnapshot(BaseModel):
    """A class representing the state of the recipe."""

    recipe: Recipe = Field(
        default_factory=Recipe, description='The current state of the recipe'
    )


# Ensure environment variables are loaded.
load_dotenv()

agent: Agent = Agent(
    'openai:gpt-4o-mini',
    output_type=str,
    deps_type=StateDeps[RecipeSnapshot],
)

app: AGUIApp = agent.to_ag_ui(deps=StateDeps(RecipeSnapshot()))


@agent.tool_plain
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


@agent.instructions
def recipe_instructions(ctx: RunContext[StateDeps[RecipeSnapshot]]) -> str:
    """Instructions for the recipe generation agent.

    Args:
        ctx: The run context containing recipe state information.

    Returns:
        Instructions string for the recipe generation agent.
    """
    return f"""You are a helpful assistant for creating recipes.

IMPORTANT:
- Create a complete recipe using the existing ingredients
- Append new ingredients to the existing ones
- Use the `display_recipe` tool to present the recipe to the user
- Do NOT repeat the recipe in the message, use the tool instead

Once you have created the updated recipe and displayed it to the user,
summarise the changes in one sentence, don't describe the recipe in
detail or send it as a message to the user.

The current state of the recipe is:

{ctx.deps.state.recipe.model_dump_json(indent=2)}
"""
