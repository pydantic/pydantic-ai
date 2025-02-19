"""Example of a graph for asking and evaluating questions.

Run with:

    uv run -m pydantic_ai_examples.question_graph
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Union, cast

import logfire
from devtools import debug
from pydantic_graph import BaseNode, Edge, End, Graph, GraphRunContext, HistoryStep

from pydantic_ai import Agent
from pydantic_ai.format_as_xml import format_as_xml
from pydantic_ai.messages import ModelMessage

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')

ask_agent = Agent('openai:gpt-4o')


@dataclass
class QuestionState:
    question: str | None = None
    ask_agent_messages: list[ModelMessage] = field(default_factory=list)
    evaluate_agent_messages: list[ModelMessage] = field(default_factory=list)


@dataclass
class Ask(BaseNode):
    state: QuestionState

    async def run(self, ctx: GraphRunContext) -> Answer:
        result = await ask_agent.run(
            'Ask a simple question with a single correct answer.',
            message_history=self.state.ask_agent_messages,
        )
        self.state.ask_agent_messages += result.all_messages()
        self.state.question = result.data
        return Answer(self.state)


@dataclass
class Answer(BaseNode):
    state: QuestionState
    answer: str | None = None

    async def run(self, ctx: GraphRunContext) -> Evaluate:
        assert self.answer is not None
        return Evaluate(self.state, self.answer)


@dataclass
class EvaluationResult:
    correct: bool
    comment: str


evaluate_agent = Agent(
    'openai:gpt-4o',
    result_type=EvaluationResult,
    system_prompt='Given a question and answer, evaluate if the answer is correct.',
)


@dataclass
class Evaluate(BaseNode):
    state: QuestionState
    answer: str

    async def run(
        self,
        ctx: GraphRunContext,
    ) -> Congratulate | Reprimand:
        assert self.state.question is not None
        result = await evaluate_agent.run(
            format_as_xml({'question': self.state.question, 'answer': self.answer}),
            message_history=self.state.evaluate_agent_messages,
        )
        self.state.evaluate_agent_messages += result.all_messages()
        if result.data.correct:
            return Congratulate(self.state, result.data.comment)
        else:
            return Reprimand(self.state, result.data.comment)


@dataclass
class Congratulate(BaseNode[None, None]):
    state: QuestionState
    comment: str

    async def run(self, ctx: GraphRunContext) -> Annotated[End, Edge(label='success')]:
        print(f'Correct answer! {self.comment}')
        return End(None)


@dataclass
class Reprimand(BaseNode):
    state: QuestionState
    comment: str

    async def run(self, ctx: GraphRunContext) -> Ask:
        print(f'Comment: {self.comment}')
        # > Comment: Vichy is no longer the capital of France.
        self.state.question = None
        return Ask(self.state)


question_graph = Graph(nodes=(Ask, Answer, Evaluate, Congratulate, Reprimand))


async def run_as_continuous():
    state = QuestionState()
    node = Ask(state)
    history: list[HistoryStep[None]] = []
    with logfire.span('run questions graph'):
        while True:
            node = await question_graph.next(node, history)
            if isinstance(node, End):
                debug([e.data for e in history])
                break
            elif isinstance(node, Answer):
                assert state.question
                node.answer = input(f'{state.question} ')
            # otherwise just continue


async def run_as_cli(answer: str | None):
    history_file = Path('question_graph_history.json')
    history = (
        question_graph.load_history(history_file.read_bytes())
        if history_file.exists()
        else []
    )

    if history:
        last = history[-1]
        assert last.kind == 'node', 'expected last step to be a node'
        last_node = cast(
            Union[Ask, Answer, Evaluate, Congratulate, Reprimand], last.node
        )
        state = last_node.state
        assert answer is not None, 'answer is required to continue from history'
        node = Answer(state, answer)
    else:
        state = QuestionState()
        node = Ask(state)
    debug(state, node)

    with logfire.span('run questions graph'):
        while True:
            node = await question_graph.next(node, history)
            if isinstance(node, End):
                debug([e.data for e in history])
                print('Finished!')
                break
            elif isinstance(node, Answer):
                print(state.question)
                break
            # otherwise just continue

    history_file.write_bytes(question_graph.dump_history(history, indent=2))


if __name__ == '__main__':
    import asyncio
    import sys

    try:
        sub_command = sys.argv[1]
        assert sub_command in ('continuous', 'cli', 'mermaid')
    except (IndexError, AssertionError):
        print(
            'Usage:\n'
            '  uv run -m pydantic_ai_examples.question_graph mermaid\n'
            'or:\n'
            '  uv run -m pydantic_ai_examples.question_graph continuous\n'
            'or:\n'
            '  uv run -m pydantic_ai_examples.question_graph cli [answer]',
            file=sys.stderr,
        )
        sys.exit(1)

    if sub_command == 'mermaid':
        print(question_graph.mermaid_code(start_node=Ask))
    elif sub_command == 'continuous':
        asyncio.run(run_as_continuous())
    else:
        a = sys.argv[2] if len(sys.argv) > 2 else None
        asyncio.run(run_as_cli(a))
