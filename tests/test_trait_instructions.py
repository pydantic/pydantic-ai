"""Instructions gated on a model trait, resolved against the model handling each request (#12545).

Offline tests with `FunctionModel`: which model a request goes to, and what it's sent and records, doesn't depend
on the provider, and scripted models can fail on cue.
"""

from __future__ import annotations

from typing import Any

import pytest

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    TraitInstruction,
)
from pydantic_ai.capabilities import Capability, ModelFallback
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import InstructionBaselinePart, InstructionDeltaPart, SystemPromptPart, UserPromptPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.openai import openai_model_profile

from ._inline_snapshot import snapshot

DELEGATION = "Don't spawn sub-agents unless the user asks."

DELEGATES: ModelProfile = {'traits': {'delegates_unprompted': True}, 'supports_inline_system_prompts': True}
STEADY: ModelProfile = {'supports_inline_system_prompts': True}


def _house_rules() -> Capability[Any]:
    return Capability(
        id='house_rules',
        instructions=['Be brief.', TraitInstruction('delegates_unprompted', DELEGATION, name='delegation')],
    )


OPENED_WITH = (
    'Be brief.\n\n<pyai:context id="capability:house_rules:delegation">\n'
    "Don't spawn sub-agents unless the user asks.\n</pyai:context>\n"
    'A later <pyai:context> element with the same id replaces the earlier one and stays in effect until replaced again.'
)
"""The instructions a conversation started on a model with the trait opens with."""

WITHDRAWAL = {
    True: ['<pyai:context id="capability:house_rules:delegation">\nThis context has been withdrawn.\n</pyai:context>'],
    False: [
        '<system><pyai:context id="capability:house_rules:delegation">\nThis context has been withdrawn, '
        'and the withdrawal stays in effect until replaced again.\n</pyai:context></system>'
    ],
}
"""What the fallback model is sent after the user's prompt, with a native mid-conversation system role and without."""

Sent = list[tuple[str, str | None, list[str]]]


def _model(name: str, profile: ModelProfile, sent: Sent, *, fails: bool = False) -> FunctionModel:
    """A model that logs its instructions and what its last request says in system voice, then answers with its name."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        last = messages[-1]
        system_voice = [
            part.content
            for part in (last.parts if isinstance(last, ModelRequest) else [])
            if isinstance(part, SystemPromptPart)
            or (
                isinstance(part, UserPromptPart)
                and isinstance(part.content, str)
                and part.content.startswith('<system>')
            )
        ]
        sent.append((name, info.instructions, [str(content) for content in system_voice]))
        if fails:
            raise ModelAPIError(model_name=name, message='down')
        return ModelResponse(parts=[TextPart(name)])

    return FunctionModel(respond, model_name=name, profile=profile)


def _recorded_changes(messages: list[ModelMessage]) -> list[tuple[str, str | None]]:
    return [
        (str(part.id), part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, InstructionDeltaPart)
    ]


def test_openai_models_delegate_unprompted():
    assert openai_model_profile('gpt-5.2').get('traits') == {'delegates_unprompted': True}
    assert openai_model_profile('gpt-oss-120b').get('traits') is None
    assert anthropic_model_profile('claude-opus-5-5') is None or 'traits' not in (
        anthropic_model_profile('claude-opus-5-5') or {}
    )


def test_merge_profile_merges_traits_by_trait():
    """Overriding one trait keeps the others; a profile without `traits` keeps the base's."""
    base: ModelProfile = {'traits': {'delegates_unprompted': True}}
    assert merge_profile(base, {'supports_tools': True}) == {
        'traits': {'delegates_unprompted': True},
        'supports_tools': True,
    }
    assert merge_profile(base, {'traits': {'delegates_unprompted': False}}) == {
        'traits': {'delegates_unprompted': False}
    }
    assert merge_profile({'supports_tools': True}) == {'supports_tools': True}


async def test_only_models_with_the_trait_get_the_text():
    sent: Sent = []
    await Agent(_model('gpt', DELEGATES, sent), capabilities=[_house_rules()]).run('x')
    await Agent(_model('claude', STEADY, sent), capabilities=[_house_rules()]).run('x')
    assert [(name, instructions) for name, instructions, _ in sent] == snapshot(
        [
            (
                'gpt',
                """\
Be brief.

<pyai:context id="capability:house_rules:delegation">
Don't spawn sub-agents unless the user asks.
</pyai:context>
A later <pyai:context> element with the same id replaces the earlier one and stays in effect until replaced again.\
""",
            ),
            ('claude', 'Be brief.'),
        ]
    )


@pytest.mark.parametrize('inline_system', [True, False])
async def test_fallback_to_a_model_without_the_trait_withdraws_the_text(inline_system: bool):
    """A conversation started on a model with the trait, continued on a fallback without it, appends a withdrawal.

    The instructions the conversation opened with are sent unchanged, so the prefix holds; the change comes after.
    """
    sent: Sent = []
    first = await Agent(_model('gpt', DELEGATES, sent), capabilities=[_house_rules()]).run('x')
    agent = Agent(
        _model('gpt', DELEGATES, sent, fails=True),
        capabilities=[
            _house_rules(),
            ModelFallback(_model('claude', {'supports_inline_system_prompts': inline_system}, sent)),
        ],
    )
    second = await agent.run('y', message_history=first.all_messages())
    assert second.output == 'claude'
    # Every request opens with the same instructions, so the prefix holds; the withdrawal follows the new prompt.
    assert [(name, instructions) for name, instructions, _ in sent] == [
        ('gpt', OPENED_WITH),
        ('gpt', OPENED_WITH),
        ('claude', OPENED_WITH),
    ]
    assert [changes for *_, changes in sent] == [[], [], WITHDRAWAL[inline_system]]
    assert _recorded_changes(second.all_messages()) == snapshot([('capability:house_rules:delegation', None)])


async def test_switch_to_a_model_with_the_trait_appends_the_text():
    """A conversation started without the trait gets the text as a change, not as a rewritten prefix."""
    sent: Sent = []
    first = await Agent(_model('claude', STEADY, sent), capabilities=[_house_rules()]).run('x')
    second = await Agent(_model('gpt', DELEGATES, sent), capabilities=[_house_rules()]).run(
        'y', message_history=first.all_messages()
    )
    assert sent == snapshot(
        [
            ('claude', 'Be brief.', []),
            (
                'gpt',
                'Be brief.',
                [
                    """\
<pyai:context id="capability:house_rules:delegation">
Don't spawn sub-agents unless the user asks.
</pyai:context>\
"""
                ],
            ),
        ]
    )
    assert _recorded_changes(second.all_messages()) == snapshot(
        [('capability:house_rules:delegation', "Don't spawn sub-agents unless the user asks.")]
    )
    assert any(
        isinstance(part, InstructionBaselinePart)
        for message in second.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


async def test_a_fallback_model_has_no_traits():
    """`FallbackModel` has no profile of its own, so the text is left out rather than failing the run."""
    sent: Sent = []
    agent = Agent(
        FallbackModel(_model('gpt', DELEGATES, sent, fails=True), _model('claude', STEADY, sent)),
        capabilities=[_house_rules()],
    )
    assert (await agent.run('x')).output == 'claude'
    assert sent == snapshot([('gpt', 'Be brief.', []), ('claude', 'Be brief.', [])])


@pytest.mark.parametrize('sticky', [False, True])
async def test_fallback_within_a_run_is_resolved_for_each_model(sticky: bool):
    """Within one run, each attempt is sent the text as its own model's profile has it."""
    sent: Sent = []
    agent = Agent(
        _model('gpt', DELEGATES, sent, fails=True),
        capabilities=[_house_rules(), ModelFallback(_model('claude', STEADY, sent), sticky=sticky)],
    )
    result = await agent.run('x')
    assert result.output == 'claude'
    assert sent == snapshot(
        [
            (
                'gpt',
                """\
Be brief.

<pyai:context id="capability:house_rules:delegation">
Don't spawn sub-agents unless the user asks.
</pyai:context>
A later <pyai:context> element with the same id replaces the earlier one and stays in effect until replaced again.\
""",
                [],
            ),
            ('claude', 'Be brief.', []),
        ]
    )
    assert _recorded_changes(result.all_messages()) == snapshot([])
