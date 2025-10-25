from __future__ import annotations as _annotations

import textwrap
from enum import Enum

from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

EVALUATION_INSTRUCTIONS = """
You are presented with a question and two possible answers A and B. Evaluate carefully whether answer A or answer B is the better reply.
You have got only these two options. Your evaluations contribute to Bradley-Terry scores across multiple items. Consistency and
objectivity are critical for reliable rankings. Each comparison should be independent but internally consistent.

<EXAMPLES>
Example 1:
<QUESTION> Which of the two ice cream flavours below is more creative? </QUESTION>
<A> Vanilla </A> 
<B> Pickled Citrus Ribbon </B>
Expected output:
{
    "result": "B",
}

Example 2:
<QUESTION> Which search query shows more genuine curiosity? </QUESTION>
<A> effect of ocean acidification feedback loops on Arctic methane release </A> 
<B> climate change effects </B>
Expected output:
{
    "result": "A",
}

Example 3:
<QUESTION> Which reply is more insulting? </QUESTION>
<A> Your argument lacks logical coherence and fails to address the core issue at hand. </A> 
<B> That's an interesting perspective, though I see it differently. </B>
Expected output:
{
    "result": "A",
}
</EXAMPLES>

<REQUIREMENTS>
1. Consider the question carefully. What aspects are important for the answer?
2. Think about answer A. Is it a good answer to the question? Why (not)?
3. Think about answer B. Is it a good answer to the question? Why (not)?
4. Make a decision based on your analysis.
</REQUIREMENTS>

<OUTPUT_FORMAT>
You must respond with valid JSON containing exactly one field called "response" with value "A" or "B":

{
    "response": "A",
}
or
{
    "response": "B",
}

Do NOT include explanations, reasoning, or any other fields.
</OUTPUT_FORMAT>
"""

class EvalPlayer(BaseModel):
    """Player in a Bradley-Terry tournament."""
    idx: int = Field(..., description='unique identifier for the player')
    item: str = Field(..., description='item to be scored')
    score: float | None = Field(default=None, description='Bradley-Terry strength score for the item')


class GameResult(str, Enum):
    """Possible results of an evaluation game."""
    A = 'A'
    B = 'B'


class EvalGame(BaseModel):
    """Represents a game between two players in the evaluation tournament."""
    criterion: str = Field(..., description='evaluation criterion on which players should be judged')

    async def run(self, players: tuple[EvalPlayer, EvalPlayer], agent: Agent[None, GameResult], model_settings: ModelSettings) -> tuple[int, int]:
        prompt = textwrap.dedent(f"""
            <QUESTION> {self.criterion} </QUESTION>
            <A> {players[0].item} </A>
            <B> {players[1].item} </B>
        """)

        async with agent:
            result = await agent.run(
                user_prompt=prompt,
                model_settings=model_settings,
            )

        if result.output == GameResult.A:
            return (players[0].idx, players[1].idx)
        else:
            return (players[1].idx, players[0].idx)
