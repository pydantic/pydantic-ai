"""PlanOutline subgraph.

state PlanOutline {
    [*]
    ClarifyRequest: Clarify user request & scope
    HumanFeedback: Human provides clarifications
    GenerateOutline: Draft initial outline
    ReviewOutline: Supervisor reviews outline

    [*] --> ClarifyRequest
    ClarifyRequest --> HumanFeedback: need more info
    HumanFeedback --> ClarifyRequest
    ClarifyRequest --> GenerateOutline: ready
    GenerateOutline --> ReviewOutline
    ReviewOutline --> GenerateOutline: revise
    ReviewOutline --> [*]: approve
}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from pydantic_graph.v2.graph_builder import GraphBuilder
from pydantic_graph.v2.step import StepContext
from pydantic_graph.v2.util import TypeExpression

from .nodes import Interruption, Prompt
from .shared_types import MessageHistory, Outline


# Types
## State
@dataclass
class State:
    chat: MessageHistory
    outline: Outline | None


## handle_user_message
class Clarify(BaseModel):
    """Ask some questions to clarify the user request."""

    choice: Literal['clarify']
    message: str


class Refuse(BaseModel):
    """Use this if you should not do research.

    This is the right choice if the user didn't ask for research, or if the user did but there was a safety concern.
    """

    choice: Literal['refuse']
    message: str  # message to show user


class Proceed(BaseModel):
    """There is enough information to proceed with handling the user's request."""

    choice: Literal['proceed']


## generate_outline
class ExistingOutlineFeedback(BaseModel):
    outline: Outline
    feedback: str


class GenerateOutlineInputs(BaseModel):
    chat: MessageHistory
    feedback: ExistingOutlineFeedback | None


## review_outline
class ReviewOutlineInputs(BaseModel):
    chat: MessageHistory
    outline: Outline

    def combine_with_choice(
        self, choice: ReviseOutlineChoice | ApproveOutlineChoice
    ) -> ReviseOutline | ApproveOutline:
        if isinstance(choice, ReviseOutlineChoice):
            return ReviseOutline(outline=self.outline, details=choice.details)
        else:
            return ApproveOutline(outline=self.outline, message=choice.message)


class ReviseOutlineChoice(BaseModel):
    choice: Literal['revise'] = 'revise'
    details: str


class ReviseOutline(ReviseOutlineChoice):
    outline: Outline


class ApproveOutlineChoice(BaseModel):
    choice: Literal['approve'] = 'approve'
    message: str  # message to user describing the research you are going to do


class ApproveOutline(ApproveOutlineChoice):
    outline: Outline


class OutlineStageOutput(BaseModel):
    """Use this if you have enough information to proceed."""

    outline: Outline  # outline of the research
    message: str  # message to show user before beginning research


# Node types
@dataclass
class YieldToHuman:
    message: str


# Transforms
async def transform_proceed(ctx: StepContext[State, object]) -> GenerateOutlineInputs:
    return GenerateOutlineInputs(chat=ctx.state.chat, feedback=None)


async def transform_clarify(
    ctx: StepContext[State, Clarify],
) -> Interruption[YieldToHuman, MessageHistory]:
    return Interruption[YieldToHuman, MessageHistory](
        YieldToHuman(ctx.inputs.message), handle_user_message.id
    )


async def transform_outline(ctx: StepContext[State, Outline]) -> ReviewOutlineInputs:
    return ReviewOutlineInputs(chat=ctx.state.chat, outline=ctx.inputs)


async def transform_revise_outline(
    ctx: StepContext[State, ReviseOutline],
) -> GenerateOutlineInputs:
    return GenerateOutlineInputs(
        chat=ctx.state.chat,
        feedback=ExistingOutlineFeedback(
            outline=ctx.inputs.outline, feedback=ctx.inputs.details
        ),
    )


async def transform_approve_outline(
    ctx: StepContext[State, ApproveOutline],
) -> OutlineStageOutput:
    return OutlineStageOutput(outline=ctx.inputs.outline, message=ctx.inputs.message)


# Graph builder
g = GraphBuilder(
    state_type=State,
    input_type=MessageHistory,
    output_type=TypeExpression[
        Refuse | OutlineStageOutput | Interruption[YieldToHuman, MessageHistory]
    ],
)

# Nodes
handle_user_message = g.step(
    Prompt(
        input_type=MessageHistory,
        output_type=TypeExpression[Refuse | Clarify | Proceed],
        prompt='Decide how to proceed from user message',  # prompt
    ),
    node_id='handle_user_message',
)

generate_outline = g.step(
    Prompt(
        input_type=GenerateOutlineInputs,
        output_type=Outline,
        prompt='Generate the outline',
    ),
    node_id='generate_outline',
)

review_outline = g.step(
    Prompt(
        input_type=ReviewOutlineInputs,
        output_type=TypeExpression[ReviseOutlineChoice | ApproveOutlineChoice],
        output_transform=ReviewOutlineInputs.combine_with_choice,
        prompt='Review the outline',
    ),
    node_id='review_outline',
)


# Edges:
g.add(
    g.edge_from(g.start_node).label('begin').to(handle_user_message),
    g.edge_from(handle_user_message).to(
        g.decision()
        .branch(g.match(Refuse).label('refuse').to(g.end_node))
        .branch(
            g.match(Clarify)
            .label('clarify')
            .transform(transform_clarify)
            .to(g.end_node)
        )
        .branch(
            g.match(Proceed)
            .label('proceed')
            .transform(transform_proceed)
            .to(generate_outline)
        )
    ),
    g.edge_from(generate_outline).transform(transform_outline).to(review_outline),
    g.edge_from(review_outline).to(
        g.decision()
        .branch(
            g.match(ReviseOutline)
            .transform(transform_revise_outline)
            .to(generate_outline)
        )
        .branch(
            g.match(ApproveOutline).transform(transform_approve_outline).to(g.end_node)
        )
    ),
)


graph = g.build()
