"""Dynamic system prompts with `on_change='append'`: what history records, and what the model is sent."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ReinjectSystemPrompt
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile


@dataclass
class Recorder:
    """A model that records, for each request, the system prompts and user prompts it was sent."""

    inline_system: bool = True
    requests: list[list[tuple[str, str]]] = field(default_factory=list[list[tuple[str, str]]])

    def model(self) -> FunctionModel:
        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            self.requests.append(
                [
                    (part.part_kind, part.content)
                    for message in messages
                    if isinstance(message, ModelRequest)
                    for part in message.parts
                    if isinstance(part, SystemPromptPart | UserPromptPart) and isinstance(part.content, str)
                ]
            )
            return ModelResponse(parts=[TextPart('Noted.')])

        return FunctionModel(respond, profile=ModelProfile(supports_inline_system_prompts=self.inline_system))


def _system_prompts(messages: list[ModelMessage]) -> list[list[SystemPromptPart]]:
    """Each request's system prompts, as history records them."""
    return [
        [part for part in message.parts if isinstance(part, SystemPromptPart)]
        for message in messages
        if isinstance(message, ModelRequest)
    ]


def _user_name_agent(recorder: Recorder, **kwargs: str) -> Agent[str | None, str]:
    agent = Agent(recorder.model(), deps_type=str | None)

    @agent.system_prompt(dynamic=True, on_change='append', **kwargs)
    def user_name(ctx: RunContext[str | None]) -> str | None:
        return f"The user's name is {ctx.deps}." if ctx.deps else None

    return agent


def _converse(agent: Agent[str | None, str], *names: str | None) -> list[ModelMessage]:
    history: list[ModelMessage] = []
    for name in names:
        history = agent.run_sync('Hi!', deps=name, message_history=history).all_messages()
    return history


def test_change_is_appended_and_sent_as_a_replacement():
    recorder = Recorder()
    agent = _user_name_agent(recorder)
    history = _converse(agent, 'Frank', 'Bob', 'Bob')

    # The first statement stays where it was, and the change opens the run that saw it.
    assert [[part.content for part in parts] for parts in _system_prompts(history)] == snapshot(
        [["The user's name is Frank."], ["The user's name is Bob."], []]
    )
    assert {(part.dynamic_ref, part.context_id) for parts in _system_prompts(history) for part in parts} == snapshot(
        {('_user_name_agent.<locals>.user_name', 'system_prompt:user_name')}
    )
    # Every request starts with what the one before it was sent.
    assert recorder.requests == snapshot(
        [
            [
                (
                    'system-prompt',
                    """\
<context id="system_prompt:user_name">
The user's name is Frank.
</context>
A later <context> element with the same id replaces the earlier one and stays in effect until replaced again.\
""",
                ),
                ('user-prompt', 'Hi!'),
            ],
            [
                (
                    'system-prompt',
                    """\
<context id="system_prompt:user_name">
The user's name is Frank.
</context>
A later <context> element with the same id replaces the earlier one and stays in effect until replaced again.\
""",
                ),
                ('user-prompt', 'Hi!'),
                (
                    'system-prompt',
                    """\
<context id="system_prompt:user_name">
The user's name is Bob.
</context>\
""",
                ),
                ('user-prompt', 'Hi!'),
            ],
            [
                (
                    'system-prompt',
                    """\
<context id="system_prompt:user_name">
The user's name is Frank.
</context>
A later <context> element with the same id replaces the earlier one and stays in effect until replaced again.\
""",
                ),
                ('user-prompt', 'Hi!'),
                (
                    'system-prompt',
                    """\
<context id="system_prompt:user_name">
The user's name is Bob.
</context>\
""",
                ),
                ('user-prompt', 'Hi!'),
                ('user-prompt', 'Hi!'),
            ],
        ]
    )


@pytest.mark.parametrize(
    'inline_system,expected',
    [
        pytest.param(
            True,
            snapshot(
                [
                    (
                        'system-prompt',
                        """\
<context id="system_prompt:user_name">
The user's name is Frank.
</context>
A later <context> element with the same id replaces the earlier one and stays in effect until replaced again.\
""",
                    ),
                    ('user-prompt', 'Hi!'),
                    (
                        'system-prompt',
                        """\
<context id="system_prompt:user_name">
This context has been withdrawn.
</context>\
""",
                    ),
                    ('user-prompt', 'Hi!'),
                    ('user-prompt', 'Hi!'),
                    (
                        'system-prompt',
                        """\
<context id="system_prompt:user_name">
The user's name is Ann.
</context>\
""",
                    ),
                    ('user-prompt', 'Hi!'),
                ]
            ),
            id='inline-system',
        ),
        pytest.param(
            False,
            snapshot(
                [
                    (
                        'system-prompt',
                        """\
<context id="system_prompt:user_name">
The user's name is Frank.
</context>
A later <context> element with the same id replaces the earlier one and stays in effect until replaced again.\
""",
                    ),
                    ('user-prompt', 'Hi!'),
                    (
                        'user-prompt',
                        """\
<system><context id="system_prompt:user_name">
This context has been withdrawn, and the withdrawal stays in effect until replaced again.
</context></system>\
""",
                    ),
                    ('user-prompt', 'Hi!'),
                    ('user-prompt', 'Hi!'),
                    (
                        'user-prompt',
                        """\
<system><context id="system_prompt:user_name">
The user's name is Ann.
</context></system>\
""",
                    ),
                    ('user-prompt', 'Hi!'),
                ]
            ),
            id='system-tagged-user-text',
        ),
    ],
)
def test_withdrawal_and_restatement(inline_system: bool, expected: list[tuple[str, str]]):
    recorder = Recorder(inline_system=inline_system)
    agent = _user_name_agent(recorder)
    history = _converse(agent, 'Frank', None, None, 'Ann')

    # Returning `None` records an empty statement once, which withdraws the prompt.
    assert [[part.content for part in parts] for parts in _system_prompts(history)] == snapshot(
        [["The user's name is Frank."], [''], [], ["The user's name is Ann."]]
    )
    assert recorder.requests[-1] == expected


def test_prompt_first_stated_after_an_empty_value():
    """An empty value that opens the conversation has nothing to withdraw, so the first real value opens the id."""
    recorder = Recorder()
    agent = _user_name_agent(recorder)
    _converse(agent, None, 'Frank')

    assert recorder.requests == snapshot(
        [
            [('user-prompt', 'Hi!')],
            [
                ('user-prompt', 'Hi!'),
                (
                    'system-prompt',
                    """\
<context id="system_prompt:user_name">
The user's name is Frank.
</context>
A later <context> element with the same id replaces the earlier one and stays in effect until replaced again.\
""",
                ),
                ('user-prompt', 'Hi!'),
            ],
        ]
    )


def test_change_opens_a_resumed_request():
    """A run that resumes a trailing request, with no new prompt, puts the change at the start of that request."""
    recorder = Recorder()
    agent = _user_name_agent(recorder)
    history = [*_converse(agent, 'Frank'), ModelRequest(parts=[UserPromptPart('Hi again!')])]

    result = agent.run_sync(deps='Bob', message_history=history)

    assert [[part.content for part in parts] for parts in _system_prompts(result.all_messages())] == snapshot(
        [["The user's name is Frank."], ["The user's name is Bob."]]
    )


def test_name_sets_the_id():
    recorder = Recorder()
    agent = _user_name_agent(recorder, name='user')
    _converse(agent, 'Frank')

    assert recorder.requests[0][0] == snapshot(
        (
            'system-prompt',
            """\
<context id="system_prompt:user">
The user's name is Frank.
</context>
A later <context> element with the same id replaces the earlier one and stays in effect until replaced again.\
""",
        )
    )


def test_another_agent_inherits_the_statements():
    """The statements are history, so an agent without the function sends them the same way."""
    first = Recorder()
    history = _converse(_user_name_agent(first), 'Frank', 'Bob')

    second = Recorder()
    other = Agent(second.model())
    other.run_sync(
        'Hi!', message_history=ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(history))
    )

    assert second.requests[0][:4] == first.requests[-1]


def test_reinject_restores_a_missing_prompt_as_its_first_statement():
    """History without its system prompts, as a UI frontend sends it, gets the current value back at its head."""
    recorder = Recorder()
    agent = _user_name_agent(recorder)
    history = _converse(agent, 'Frank', 'Bob')
    stripped = [
        ModelRequest(parts=[part for part in message.parts if not isinstance(part, SystemPromptPart)])
        if isinstance(message, ModelRequest)
        else message
        for message in history
    ]

    result = agent.run_sync('Hi!', deps='Ann', message_history=stripped, capabilities=[ReinjectSystemPrompt()])

    assert [[part.content for part in parts] for parts in _system_prompts(result.all_messages())] == snapshot(
        [["The user's name is Ann."], [], []]
    )
    assert recorder.requests[-1] == snapshot(
        [
            (
                'system-prompt',
                """\
<context id="system_prompt:user_name">
The user's name is Ann.
</context>
A later <context> element with the same id replaces the earlier one and stays in effect until replaced again.\
""",
            ),
            ('user-prompt', 'Hi!'),
            ('user-prompt', 'Hi!'),
            ('user-prompt', 'Hi!'),
        ]
    )


def test_reinject_replace_existing_restates_the_prompt_at_the_head():
    """`replace_existing=True` drops every statement of the prompt and puts the current value at the head."""
    recorder = Recorder()
    agent = _user_name_agent(recorder)
    history = _converse(agent, 'Frank', 'Bob')

    result = agent.run_sync(
        'Hi!', deps='Ann', message_history=history, capabilities=[ReinjectSystemPrompt(replace_existing=True)]
    )

    assert [[part.content for part in parts] for parts in _system_prompts(result.all_messages())] == snapshot(
        [["The user's name is Ann."], [], []]
    )
    assert recorder.requests[-1] == snapshot(
        [
            (
                'system-prompt',
                """\
<context id="system_prompt:user_name">
The user's name is Ann.
</context>
A later <context> element with the same id replaces the earlier one and stays in effect until replaced again.\
""",
            ),
            ('user-prompt', 'Hi!'),
            ('user-prompt', 'Hi!'),
            ('user-prompt', 'Hi!'),
        ]
    )


def test_append_requires_dynamic():
    agent = Agent('test')
    with pytest.raises(UserError, match="`on_change='append'` only applies to a dynamic system prompt"):
        agent.system_prompt(on_change='append')


def test_name_requires_append():
    agent = Agent('test')
    with pytest.raises(UserError, match="`name` only applies to a system prompt with `on_change='append'`"):
        agent.system_prompt(dynamic=True, name='user')


def test_names_are_unique():
    agent = Agent('test')

    @agent.system_prompt(dynamic=True, on_change='append', name='user')
    def first() -> str:
        return 'First.'  # pragma: no cover

    with pytest.raises(
        UserError, match="Another dynamic system prompt with `on_change='append'` is already named 'user'"
    ):

        @agent.system_prompt(dynamic=True, on_change='append', name='user')
        def second() -> str:
            return 'Second.'  # pragma: no cover


def test_name_cannot_contain_a_colon():
    agent = Agent('test')
    with pytest.raises(UserError, match="System prompt name 'user:name' cannot contain a colon"):

        @agent.system_prompt(dynamic=True, on_change='append', name='user:name')
        def user_name() -> str:
            return 'Name.'  # pragma: no cover
