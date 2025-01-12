"""Example of a graph for asking and evaluating questions.

Run with:

    uv run -m pydantic_ai_examples.question_graph
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from typing import Annotated

import logfire
from devtools import debug
from pydantic_graph import BaseNode, Edge, End, Graph, GraphContext, HistoryStep

from pydantic_ai import Agent
from pydantic_ai.format_as_xml import format_as_xml
from pydantic_ai.messages import ModelMessage

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')

ask_agent = Agent('openai:gpt-4o', result_type=str)


@dataclass
class QuestionState:
    ask_agent_messages: list[ModelMessage] = field(default_factory=list)
    evaluate_agent_messages: list[ModelMessage] = field(default_factory=list)


@dataclass
class Ask(BaseNode[QuestionState]):
    """Generate a question to ask the user.

    Uses the GPT-4o model to generate the question.
    """

    async def run(
        self, ctx: GraphContext[QuestionState]
    ) -> Annotated[Answer, Edge(label='ask the question')]:
        result = await ask_agent.run(
            'Ask a simple question with a single correct answer.',
            message_history=ctx.state.ask_agent_messages,
        )
        ctx.state.ask_agent_messages += result.all_messages()
        return Answer(result.data)


@dataclass
class Answer(BaseNode[QuestionState]):
    """Get the answer to the question from the user.

    This node must be completed outside the graph run.
    """

    question: str
    answer: str | None = None

    async def run(
        self, ctx: GraphContext[QuestionState]
    ) -> Annotated[Evaluate, Edge(label='answer the question')]:
        assert self.answer is not None
        return Evaluate(self.question, self.answer)


@dataclass
class EvaluationResult:
    correct: bool
    comment: str


evaluate_agent = Agent(
    'openai:gpt-4o',
    result_type=EvaluationResult,
    system_prompt='Given a question and answer, evaluate if the answer is correct.',
    result_tool_name='evaluation',
)


@dataclass
class Evaluate(BaseNode[QuestionState]):
    question: str
    answer: str

    async def run(
        self,
        ctx: GraphContext[QuestionState],
    ) -> Congratulate | Castigate:
        result = await evaluate_agent.run(
            format_as_xml({'question': self.question, 'answer': self.answer}),
            message_history=ctx.state.evaluate_agent_messages,
        )
        ctx.state.evaluate_agent_messages += result.all_messages()
        if result.data.correct:
            return Congratulate(result.data.comment)
        else:
            return Castigate(result.data.comment)


@dataclass
class Congratulate(BaseNode[QuestionState, None]):
    """Congratulate the user and end."""

    comment: str

    async def run(
        self, ctx: GraphContext[QuestionState]
    ) -> Annotated[End, Edge(label='success')]:
        print(f'Correct answer! {self.comment}')
        return End(None)


@dataclass
class Castigate(BaseNode[QuestionState]):
    """Castigate the user, then ask another question."""

    comment: str

    async def run(
        self, ctx: GraphContext[QuestionState]
    ) -> Annotated[Ask, Edge(label='try again')]:
        print(f'Comment: {self.comment}')
        return Ask()


question_graph = Graph(nodes=(Ask, Answer, Evaluate, Congratulate, Castigate))
print(question_graph.mermaid_code(start_node=Ask, notes=False))


async def main():
    state = QuestionState()
    node = Ask()
    history: list[HistoryStep[QuestionState]] = []
    with logfire.span('run questions graph'):
        while True:
            node = await question_graph.next(state, node, history)
            if isinstance(node, End):
                debug([e.data_snapshot() for e in history])
                break
            elif isinstance(node, Answer):
                node.answer = input(f'{node.question} ')
            # otherwise just continue


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
