"""A `ThinkingPart` that can't round-trip natively goes in the user turn for models that imitate formatting.

Regression test for [#5869](https://github.com/pydantic/pydantic-ai/issues/5869): an unsigned or
foreign-provider `ThinkingPart` in history (round-tripped through storage, rebuilt by a history processor,
or produced by another model in a `FallbackModel` chain) can't be sent through the model's own native
reasoning channel, so it falls back to text. Claude reads the assistant turns of a history as examples of
how it is supposed to write, so putting that text in the assistant turn teaches it to emit `<thinking>`
tags in the answers the user reads — measured live in
`tests/models/anthropic/test_thinking_tag_mimicry.py`. Models carrying
`mimics_assistant_message_formatting` therefore carry it in the preceding user message instead.

The outbound request body is asserted directly via each model's `_map_message`/`_map_messages`: provider
cassettes match on method and URI only, so a VCR test would play back green regardless of which turn the
reasoning ends up in.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import pytest
from inline_snapshot import snapshot

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.profiles import ModelProfile

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
# was minted by another model, so it can't ride the serving provider's native channel and falls back to
# text. This pins the provider half of each native gate as load-bearing — a gate loosened to check only
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
# A history opening with a `ModelResponse` has no user turn to carry the reasoning, so one is created.
_LEADING_RESPONSE_HISTORY: list[ModelMessage] = [
    ModelResponse(parts=[ThinkingPart(content=_REASONING), TextPart(content=_ANSWER)]),
    ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
]
_NO_THINKING_HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
    ModelResponse(parts=[TextPart(content=_ANSWER)]),
]


async def _anthropic_outbound(history: list[ModelMessage]) -> list[dict[str, object]]:
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='x'))
    _, messages = await model._map_message(  # pyright: ignore[reportPrivateUsage]
        history, ModelRequestParameters(), AnthropicModelSettings()
    )
    return [dict(message) for message in messages]


async def _anthropic_opted_out_outbound(history: list[ModelMessage]) -> list[dict[str, object]]:
    """The documented opt-out: with the flag off, the reasoning goes back in the assistant turn."""
    model = AnthropicModel(
        'claude-sonnet-4-5',
        provider=AnthropicProvider(api_key='x'),
        profile=ModelProfile(mimics_assistant_message_formatting=False),
    )
    _, messages = await model._map_message(  # pyright: ignore[reportPrivateUsage]
        history, ModelRequestParameters(), AnthropicModelSettings()
    )
    return [dict(message) for message in messages]


async def _xai_outbound(history: list[ModelMessage]) -> list[dict[str, object]]:
    model = XaiModel('grok-4-fast-reasoning', provider=XaiProvider(api_key='x'))
    messages = await model._map_messages(history, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    return [MessageToDict(m, preserving_proto_field_name=True) for m in messages]


async def _bedrock_outbound(history: list[ModelMessage]) -> list[dict[str, object]]:
    model = BedrockConverseModel(
        'us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=BedrockProvider(api_key='x', region_name='us-east-1')
    )
    _, messages = await model._map_messages(history, ModelRequestParameters(), None)  # pyright: ignore[reportPrivateUsage]
    return [dict(message) for message in messages]


@dataclass
class Case:
    id: str
    outbound: Callable[[list[ModelMessage]], Awaitable[list[dict[str, object]]]]
    history: list[ModelMessage]
    carrying_roles: set[str]
    """The roles of the outbound messages expected to carry the reasoning."""
    expected: object
    marks: tuple[pytest.MarkDecorator, ...] = field(default_factory=tuple)


CASES = [
    Case(
        'anthropic-foreign',
        _anthropic_outbound,
        _FOREIGN_HISTORY,
        carrying_roles={'user'},
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
                    'role': 'user',
                    'content': [
                        {
                            'text': """\
<thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</thinking>\
""",
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
        'anthropic-signed-foreign',
        _anthropic_outbound,
        _SIGNED_FOREIGN_HISTORY,
        carrying_roles={'user'},
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
                    'role': 'user',
                    'content': [
                        {
                            'text': """\
<thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</thinking>\
""",
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
        'anthropic-leading-response',
        _anthropic_outbound,
        _LEADING_RESPONSE_HISTORY,
        carrying_roles={'user'},
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': [
                        {
                            'text': """\
<thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</thinking>\
""",
                            'type': 'text',
                        }
                    ],
                },
                {
                    'role': 'assistant',
                    'content': [{'text': 'The 10-year Treasury has more interest-rate risk.', 'type': 'text'}],
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            'text': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?',
                            'type': 'text',
                        }
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
        carrying_roles=set(),
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
        'anthropic-foreign-opted-out',
        _anthropic_opted_out_outbound,
        _FOREIGN_HISTORY,
        carrying_roles={'assistant'},
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
<thinking>
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
        'bedrock-leading-response',
        _bedrock_outbound,
        _LEADING_RESPONSE_HISTORY,
        carrying_roles={'user'},
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': [
                        {
                            'text': """\
<thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</thinking>\
"""
                        }
                    ],
                },
                {'role': 'assistant', 'content': [{'text': 'The 10-year Treasury has more interest-rate risk.'}]},
                {
                    'role': 'user',
                    'content': [
                        {'text': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?'}
                    ],
                },
            ]
        ),
        marks=(pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed'),),
    ),
    Case(
        'bedrock-foreign',
        _bedrock_outbound,
        _FOREIGN_HISTORY,
        carrying_roles={'user'},
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': [
                        {'text': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?'},
                        {
                            'text': """\
<thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</thinking>\
"""
                        },
                    ],
                },
                {'role': 'assistant', 'content': [{'text': 'The 10-year Treasury has more interest-rate risk.'}]},
            ]
        ),
        marks=(pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed'),),
    ),
    Case(
        'bedrock-signed-foreign',
        _bedrock_outbound,
        _SIGNED_FOREIGN_HISTORY,
        carrying_roles={'user'},
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': [
                        {'text': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?'},
                        {
                            'text': """\
<thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</thinking>\
"""
                        },
                    ],
                },
                {'role': 'assistant', 'content': [{'text': 'The 10-year Treasury has more interest-rate risk.'}]},
            ]
        ),
        marks=(pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed'),),
    ),
    Case(
        'bedrock-no-thinking',
        _bedrock_outbound,
        _NO_THINKING_HISTORY,
        carrying_roles=set(),
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
    # Grok showed no imitation when reasoning was replayed in its assistant turn, so its profile doesn't
    # carry the flag and its rendering is unchanged: thinking tags, in the assistant turn.
    Case(
        'xai-foreign-stays-in-assistant-turn',
        _xai_outbound,
        _FOREIGN_HISTORY,
        carrying_roles={'assistant'},
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
<think>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</think>\
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
        'xai-signed-foreign-stays-in-assistant-turn',
        _xai_outbound,
        _SIGNED_FOREIGN_HISTORY,
        carrying_roles={'assistant'},
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
<think>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</think>\
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
        carrying_roles=set(),
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
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id, marks=c.marks) for c in CASES])
async def test_foreign_thinking_placement(case: Case):
    """Reasoning that can't ride the native channel reaches models that imitate formatting as user content,
    and every other model as assistant content, in both cases wrapped in the profile's thinking tags."""
    body = await case.outbound(case.history)
    assert body == case.expected

    # `ROLE_USER`/`ROLE_ASSISTANT` on the xAI wire, `user`/`assistant` on the other two.
    assert {
        str(message['role']).lower().removeprefix('role_')
        for message in body
        if _REASONING in json.dumps(message, default=str)
    } == case.carrying_roles
