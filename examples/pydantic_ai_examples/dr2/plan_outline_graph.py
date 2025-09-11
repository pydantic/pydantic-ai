# """PlanOutline subgraph.
#
# state PlanOutline {
#     [*]
#     ClarifyRequest: Clarify user request & scope
#     HumanFeedback: Human provides clarifications
#     GenerateOutline: Draft initial outline
#     ReviewOutline: Supervisor reviews outline
#
#     [*] --> ClarifyRequest
#     ClarifyRequest --> HumanFeedback: need more info
#     HumanFeedback --> ClarifyRequest
#     ClarifyRequest --> GenerateOutline: ready
#     GenerateOutline --> ReviewOutline
#     ReviewOutline --> GenerateOutline: revise
#     ReviewOutline --> [*]: approve
# }
# """
#
# from __future__ import annotations
#
# from dataclasses import dataclass
# from typing import Literal
#
# from pydantic import BaseModel
# from pydantic_graph.v2.graph import GraphBuilder
# from pydantic_graph.v2.transform import TransformContext
# from pydantic_graph.v2.util import TypeExpression
#
# from .nodes import Interruption, Prompt
# from .shared_types import MessageHistory, Outline
#
#
# # Types
# ## State
# @dataclass
# class State:
#     chat: MessageHistory
#     outline: Outline | None
#
#
# @dataclass
# class Deps:
#     pass
#
#
# ## handle_user_message
# class Clarify(BaseModel):
#     """Ask some questions to clarify the user request."""
#
#     choice: Literal['clarify']
#     message: str
#
#
# class Refuse(BaseModel):
#     """Use this if you should not do research.
#
#     This is the right choice if the user didn't ask for research, or if the user did but there was a safety concern.
#     """
#
#     choice: Literal['refuse']
#     message: str  # message to show user
#
#
# class Proceed(BaseModel):
#     """There is enough information to proceed with handling the user's request."""
#
#     choice: Literal['proceed']
#
#
# ## generate_outline
# class ExistingOutlineFeedback(BaseModel):
#     outline: Outline
#     feedback: str
#
#
# class GenerateOutlineInputs(BaseModel):
#     chat: MessageHistory
#     feedback: ExistingOutlineFeedback | None
#
#
# ## review_outline
# class ReviewOutlineInputs(BaseModel):
#     chat: MessageHistory
#     outline: Outline
#
#     def combine_with_choice(
#         self, choice: ReviseOutlineChoice | ApproveOutlineChoice
#     ) -> ReviseOutline | ApproveOutline:
#         if isinstance(choice, ReviseOutlineChoice):
#             return ReviseOutline(outline=self.outline, details=choice.details)
#         else:
#             return ApproveOutline(outline=self.outline, message=choice.message)
#
#
# class ReviseOutlineChoice(BaseModel):
#     choice: Literal['revise'] = 'revise'
#     details: str
#
#
# class ReviseOutline(ReviseOutlineChoice):
#     outline: Outline
#
#
# class ApproveOutlineChoice(BaseModel):
#     choice: Literal['approve'] = 'approve'
#     message: str  # message to user describing the research you are going to do
#
#
# class ApproveOutline(ApproveOutlineChoice):
#     outline: Outline
#
#
# class OutlineStageOutput(BaseModel):
#     """Use this if you have enough information to proceed."""
#
#     outline: Outline  # outline of the research
#     message: str  # message to show user before beginning research
#
#
# # Node types
# @dataclass
# class YieldToHuman:
#     message: str
#
#
# # Transforms
# def transform_proceed(ctx: TransformContext[State, Deps, object]):
#     return GenerateOutlineInputs(chat=ctx.state.chat, feedback=None)
#
#
# def transform_clarify(ctx: TransformContext[State, Deps, Clarify]):
#     return Interruption[YieldToHuman, MessageHistory](
#         YieldToHuman(ctx.inputs.message), handle_user_message.id
#     )
#
#
# def transform_outline(ctx: TransformContext[State, Deps, Outline]):
#     return ReviewOutlineInputs(chat=ctx.state.chat, outline=ctx.inputs)
#
#
# def transform_revise_outline(
#     ctx: TransformContext[State, Deps, ReviseOutline],
# ) -> GenerateOutlineInputs:
#     return GenerateOutlineInputs(
#         chat=ctx.state.chat,
#         feedback=ExistingOutlineFeedback(
#             outline=ctx.inputs.outline, feedback=ctx.inputs.details
#         ),
#     )
#
#
# def transform_approve_outline(
#     ctx: TransformContext[State, Deps, ApproveOutline],
# ):
#     return OutlineStageOutput(outline=ctx.inputs.outline, message=ctx.inputs.message)
#
#
# # Graph builder
# g = GraphBuilder(
#     state_type=State,
#     deps_type=Deps,
#     input_type=MessageHistory,
#     output_type=TypeExpression[
#         Refuse | OutlineStageOutput | Interruption[YieldToHuman, MessageHistory]
#     ],
# )
#
# # Nodes
# handle_user_message = g.step(
#     Prompt(
#         input_type=MessageHistory,
#         output_type=TypeExpression[Refuse | Clarify | Proceed],
#         prompt='Decide how to proceed from user message',  # prompt
#     ),
#     node_id='handle_user_message',
# )
#
# generate_outline = g.step(
#     Prompt(
#         input_type=GenerateOutlineInputs,
#         output_type=Outline,
#         prompt='Generate the outline',
#     ),
#     node_id='generate_outline',
# )
#
# review_outline = g.step(
#     Prompt(
#         input_type=ReviewOutlineInputs,
#         output_type=TypeExpression[ReviseOutlineChoice | ApproveOutlineChoice],
#         output_transform=ReviewOutlineInputs.combine_with_choice,
#         prompt='Review the outline',
#     ),
#     node_id='review_outline',
# )
#
#
# # Edges:
# g.start_with(handle_user_message)
# g.add_edge(
#     handle_user_message,
#     destination=g.decision(node_id='handle_user_decision', note='Handle user decision')
#     .branch(g.handle(Refuse).end())
#     .branch(g.handle(Proceed).transform(transform_proceed).route_to(generate_outline))
#     .branch(g.handle(Clarify).transform(transform_clarify).end()),
# )
# g.add_edge(
#     generate_outline,
#     transform=transform_outline,
#     destination=review_outline,
# )
# g.add_edge(
#     review_outline,
#     g.decision(node_id='review_outline_decision')
#     .branch(
#         g.handle(ReviseOutline)
#         .transform(transform_revise_outline)
#         .route_to(generate_outline)
#     )
#     .branch(g.handle(ApproveOutline).transform(transform_approve_outline).end()),
# )
#
# graph = g.build()
