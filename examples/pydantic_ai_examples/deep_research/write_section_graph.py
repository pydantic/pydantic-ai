# """WriteSection subgraph
#
# state ...WriteSectionN {
#     [*]
#     BuildSectionTemplate: Outline sub‑headings / bullet points
#     WriteContents: Generate paragraph drafts
#     ReviewSectionWriting: Self / human review
#
#     [*] --> BuildSectionTemplate
#     BuildSectionTemplate --> WriteContents
#     WriteContents --> ReviewSectionWriting
#     ReviewSectionWriting --> BuildSectionTemplate: refine
#     ReviewSectionWriting --> [*]: complete
# }
# """
#
# from __future__ import annotations
#
# from pydantic import BaseModel
#
# from pydantic_ai.messages import ModelMessage
#
# from .shared_types import Outline
#
#
# # TODO: Move this into another file somewhere more generic
# class Interruption[StopT, ResumeT]:
#     pass  # need to implement
#
#
# # Aliases
# type MessageHistory = list[ModelMessage]
#
#
# # Types
# class OutlineNode(BaseModel):
#     section_id: str = Field(repr=False)
#     title: str
#     description: str | None
#     requires_research: bool
#     children: list[OutlineNode] = Field(default_factory=list)
#
#
# OutlineNode.model_rebuild()
#
#
# class Outline(BaseModel):
#     # TODO: Consider replacing this with a non-recursive model that is a list of sections with depth
#     #  to make it easier to generate
#     root: OutlineNode
#
#
# ## State
# @dataclass
# class State:
#     chat: MessageHistory
#     outline: Outline | None
#
#
# ## handle_user_message
# class Clarify(BaseModel):
#     """Ask some questions to clarify the user request."""
#
#     choice: Literal[clarify]
#     message: str
#
#
# class Refuse(BaseModel):
#     """Use this if you should not do research.
#
#     This is the right choice if the user didn't ask for research, or if the user did but there was a safety concern.
#     """
#
#     choice: Literal[refuse]
#     message: str  # message to show user
#
#
# class Proceed(BaseModel):
#     """There is enough information to proceed with handling the user's request"""
#
#     choice: Literal[proceed]
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
#
# class OutlineNeedsRevision(BaseModel):
#     choice: Literal[needs - revision]
#     details: str
#
#
# class OutlineApproved(BaseModel):
#     choice: Literal[approved]
#     message: str  # message to user describing the research you are going to do
#
#
# class OutlineStageOutput(BaseModel):
#     """Use this if you have enough information to proceed"""
#
#     outline: Outline  # outline of the research
#     message: str  # message to show user before beginning research
#
#
# # Node types
# @dataclass
# class YieldToHuman(Interruption[str, MessageHistory]):
#     # TODO: Implement handling with input message and user-response MessageHistory...
#     pass
#
#
# # Graph
# _g = Graph(
#     state_type=MessageHistory, output_type=Refuse | OutlineStageOutput | YieldToHuman
# )
#
# # Graph nodes
# handle_user_message = Prompt(
#     MessageHistory,  # input_type
#     'Decide how to proceed from user message',  # prompt
#     Refuse | Clarify | Proceed,  # output_type
# )
#
# generate_outline = Prompt(
#     GenerateOutlineInputs,
#     'Generate the outline',
#     Outline,
# )
#
# review_outline = Prompt(
#     ReviewOutlineInputs,
#     'Review the outline',
#     OutlineNeedsRevision | OutlineApproved,
# )
#
# # Graph edges
# _g.start_at(_g.handle(State).transform(lambda s: s.chat).route_to(handle_user_message))
# _g.add_decision(
#     handle_user_message,
#     Routing[
#         _g.handle(Refuse).end()
#         | _g.handle(Proceed)
#         .transform(
#             variant='state',
#             call=lambda s: GenerateOutlineInputs(chat=s.chat, feedback=None),
#         )
#         .route_to(generate_outline)
#         | _g.handle(Clarify)
#         .transform(lambda o: o.message)
#         .interrupt(YieldToHuman, handle_user_message)
#     ],
# )
# _g.add_edge(generate_outline, review_outline)
# _g.add_decision(
#     review_outline,
#     Routing[
#         _g.handle(OutlineNeedsRevision)
#         .transform(
#             variant='state-inputs-outputs',
#             call=lambda s, i, o: GenerateOutlineInputs(
#                 chat=s.chat,
#                 feedback=ExistingOutlineFeedback(outline=i.outline, feedback=o.details),
#             ),
#         )
#         .route_to(generate_outline)
#         | _g.handle(OutlineApproved)
#         .transform(
#             variant='inputs-output',
#             call=lambda i, o: OutlineStageOutput(outline=i.outline, message=o.message),
#         )
#         .end()
#     ],
# )
#
# plan_outline_graph = _g
