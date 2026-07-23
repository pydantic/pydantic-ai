"""Foreign `ThinkingPart`s map to a noted `<thinking>` tag, not the model's own bare `<thinking>` tags.

Regression test for #5869: an unsigned or foreign-provider `ThinkingPart` in history (round-tripped
through storage, rebuilt by a history processor, or produced by another model in a `FallbackModel`
chain) can't be sent through the model's own native reasoning channel. Anthropic documents that bare
`<thinking>` tags in the prompt get generalized into the model's own output, so re-rendering such a
part in that native format teaches the model to leak it into user-visible answers. The three providers
with a native thinking-block concept (Anthropic, xAI, Bedrock) instead wrap it in a `<thinking>` tag
carrying an explicit note that the reasoning is carried over from another model in the conversation. The
source provider is deliberately not named.

The outbound request body is asserted directly via each model's `_map_message`/`_map_messages`:
provider cassettes match on method+URI only, so a VCR test would play back green regardless of how the
part is rendered.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import pytest
from inline_snapshot import snapshot

from pydantic_ai._thinking_part import FOREIGN_THINKING_NOTE
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters

from .conftest import try_import

with try_import() as anthropic_imports:
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as xai_imports:
    from google.protobuf.json_format import MessageToDict

    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.providers.xai import XaiProvider

with try_import() as bedrock_imports:
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

pytestmark = pytest.mark.anyio

_QUESTION = 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?'
_REASONING = (
    'Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves '
    'more per unit change in rates.'
)
_ANSWER = 'The 10-year Treasury has more interest-rate risk.'

# An unsigned `ThinkingPart` (no signature, no provider_name) is the exact #5869 trigger — the same shape
# whether it came from storage, a history processor, or another model in a `FallbackModel` chain.
_FOREIGN_HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
    ModelResponse(parts=[ThinkingPart(content=_REASONING), TextPart(content=_ANSWER)]),
]
# A *signed* but foreign-provider `ThinkingPart` is #5869's primary `FallbackModel` trigger: the signature
# was minted by another model, so it can't ride the serving provider's native channel and must fall to the
# marker. This pins the provider half of each native gate as load-bearing — a gate loosened to check only
# `signature is not None` would wrongly send this as a native block and get a provider 400.
_SIGNED_FOREIGN_HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
    ModelResponse(
        parts=[
            ThinkingPart(content=_REASONING, signature='sig-from-openai', provider_name='openai'),
            TextPart(content=_ANSWER),
        ]
    ),
]
_NO_THINKING_HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
    ModelResponse(parts=[TextPart(content=_ANSWER)]),
]


async def _anthropic_outbound(history: list[ModelMessage]) -> object:
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='x'))
    _, messages = await model._map_message(  # pyright: ignore[reportPrivateUsage]
        history, ModelRequestParameters(), AnthropicModelSettings()
    )
    return messages


async def _xai_outbound(history: list[ModelMessage]) -> object:
    model = XaiModel('grok-4-fast-reasoning', provider=XaiProvider(api_key='x'))
    messages = await model._map_messages(history, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    return [MessageToDict(m, preserving_proto_field_name=True) for m in messages]


async def _bedrock_outbound(history: list[ModelMessage]) -> object:
    model = BedrockConverseModel(
        'us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=BedrockProvider(api_key='x', region_name='us-east-1')
    )
    _, messages = await model._map_messages(history, ModelRequestParameters(), None)  # pyright: ignore[reportPrivateUsage]
    return messages


@dataclass
class Case:
    id: str
    outbound: Callable[[list[ModelMessage]], Awaitable[object]]
    history: list[ModelMessage]
    expect_marker: bool
    expected: object
    marks: tuple[pytest.MarkDecorator, ...] = field(default_factory=tuple)


CASES = [
    Case(
        'anthropic-foreign',
        _anthropic_outbound,
        _FOREIGN_HISTORY,
        expect_marker=True,
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': [
                        {
                            'text': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?',
                            'type': 'text',
                        }
                    ],
                },
                {
                    'role': 'assistant',
                    'content': [
                        {
                            'text': """\
<thinking note="continued from another model in this conversation">
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</thinking>\
""",
                            'type': 'text',
                        },
                        {'text': 'The 10-year Treasury has more interest-rate risk.', 'type': 'text'},
                    ],
                },
            ]
        ),
        marks=(pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed'),),
    ),
    Case(
        'anthropic-no-thinking',
        _anthropic_outbound,
        _NO_THINKING_HISTORY,
        expect_marker=False,
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': [
                        {
                            'text': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?',
                            'type': 'text',
                        }
                    ],
                },
                {
                    'role': 'assistant',
                    'content': [{'text': 'The 10-year Treasury has more interest-rate risk.', 'type': 'text'}],
                },
            ]
        ),
        marks=(pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed'),),
    ),
    Case(
        'xai-foreign',
        _xai_outbound,
        _FOREIGN_HISTORY,
        expect_marker=True,
        expected=snapshot(
            [
                {
                    'content': [
                        {'text': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?'}
                    ],
                    'role': 'ROLE_USER',
                },
                {
                    'content': [
                        {
                            'text': """\
<thinking note="continued from another model in this conversation">
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</thinking>\
"""
                        }
                    ],
                    'role': 'ROLE_ASSISTANT',
                },
                {'content': [{'text': 'The 10-year Treasury has more interest-rate risk.'}], 'role': 'ROLE_ASSISTANT'},
            ]
        ),
        marks=(pytest.mark.skipif(not xai_imports(), reason='xai not installed'),),
    ),
    Case(
        'xai-no-thinking',
        _xai_outbound,
        _NO_THINKING_HISTORY,
        expect_marker=False,
        expected=snapshot(
            [
                {
                    'content': [
                        {'text': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?'}
                    ],
                    'role': 'ROLE_USER',
                },
                {'content': [{'text': 'The 10-year Treasury has more interest-rate risk.'}], 'role': 'ROLE_ASSISTANT'},
            ]
        ),
        marks=(pytest.mark.skipif(not xai_imports(), reason='xai not installed'),),
    ),
    Case(
        'bedrock-foreign',
        _bedrock_outbound,
        _FOREIGN_HISTORY,
        expect_marker=True,
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': [
                        {'text': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?'}
                    ],
                },
                {
                    'role': 'assistant',
                    'content': [
                        {
                            'text': """\
<thinking note="continued from another model in this conversation">
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</thinking>\
"""
                        },
                        {'text': 'The 10-year Treasury has more interest-rate risk.'},
                    ],
                },
            ]
        ),
        marks=(pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed'),),
    ),
    Case(
        'bedrock-no-thinking',
        _bedrock_outbound,
        _NO_THINKING_HISTORY,
        expect_marker=False,
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': [
                        {'text': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?'}
                    ],
                },
                {'role': 'assistant', 'content': [{'text': 'The 10-year Treasury has more interest-rate risk.'}]},
            ]
        ),
        marks=(pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed'),),
    ),
    Case(
        'anthropic-signed-foreign',
        _anthropic_outbound,
        _SIGNED_FOREIGN_HISTORY,
        expect_marker=True,
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': [
                        {
                            'text': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?',
                            'type': 'text',
                        }
                    ],
                },
                {
                    'role': 'assistant',
                    'content': [
                        {
                            'text': """\
<thinking note="continued from another model in this conversation">
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</thinking>\
""",
                            'type': 'text',
                        },
                        {'text': 'The 10-year Treasury has more interest-rate risk.', 'type': 'text'},
                    ],
                },
            ]
        ),
        marks=(pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed'),),
    ),
    Case(
        'xai-signed-foreign',
        _xai_outbound,
        _SIGNED_FOREIGN_HISTORY,
        expect_marker=True,
        expected=snapshot(
            [
                {
                    'content': [
                        {'text': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?'}
                    ],
                    'role': 'ROLE_USER',
                },
                {
                    'content': [
                        {
                            'text': """\
<thinking note="continued from another model in this conversation">
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</thinking>\
"""
                        }
                    ],
                    'role': 'ROLE_ASSISTANT',
                },
                {'content': [{'text': 'The 10-year Treasury has more interest-rate risk.'}], 'role': 'ROLE_ASSISTANT'},
            ]
        ),
        marks=(pytest.mark.skipif(not xai_imports(), reason='xai not installed'),),
    ),
    Case(
        'bedrock-signed-foreign',
        _bedrock_outbound,
        _SIGNED_FOREIGN_HISTORY,
        expect_marker=True,
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': [
                        {'text': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?'}
                    ],
                },
                {
                    'role': 'assistant',
                    'content': [
                        {
                            'text': """\
<thinking note="continued from another model in this conversation">
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</thinking>\
"""
                        },
                        {'text': 'The 10-year Treasury has more interest-rate risk.'},
                    ],
                },
            ]
        ),
        marks=(pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed'),),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id, marks=c.marks) for c in CASES])
async def test_foreign_thinking_marker(case: Case):
    """A non-native `ThinkingPart` — unsigned, or signed by another provider — is re-rendered inside a
    noted `<thinking>` tag, never the model's own bare `<thinking>` tags; a history without thinking is
    unaffected."""
    body = await case.outbound(case.history)
    assert body == case.expected

    serialized = json.dumps(body, default=str)
    assert (FOREIGN_THINKING_NOTE in serialized) is case.expect_marker
    assert '<thinking>' not in serialized
