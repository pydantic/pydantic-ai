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

from pydantic_ai.messages import (
    CachePoint,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.profiles import ModelProfile

from ._inline_snapshot import snapshot
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
_FOLLOW_UP = 'And a 30-year?'

# An unsigned `ThinkingPart` (no signature, no provider_name) is the exact trigger — the same shape
# whether it came from storage, a history processor, or another model in a `FallbackModel` chain.
_FOREIGN_HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
    ModelResponse(parts=[ThinkingPart(content=_REASONING), TextPart(content=_ANSWER)]),
]
# A *signed* but foreign-provider `ThinkingPart` is the primary `FallbackModel` trigger: the signature
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
# The turn ahead of the reasoning carries tool results, which is what a tool-using agent hits on every
# step after the first. Merging the reasoning into that turn puts text after the `tool_result` blocks,
# which Anthropic rejects with a 400 when the same response also called a server tool it never resolved,
# so the reasoning has to keep a turn of its own. On Bedrock the equivalent merge is governed by
# `bedrock_tool_result_colocatable_content`
# (https://github.com/pydantic/pydantic-ai/issues/6081), which only the trailing merge pass consults.
_TOOL_RESULT_HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
    ModelResponse(parts=[ToolCallPart(tool_name='lookup', args={'tenor': 10}, tool_call_id='call-1')]),
    ModelRequest(parts=[ToolReturnPart(tool_name='lookup', content='8.1', tool_call_id='call-1')]),
    ModelResponse(parts=[ThinkingPart(content=_REASONING), TextPart(content=_ANSWER)]),
]
# A mid-conversation `SystemPromptPart` renders a `system` entry behind the user turn, so a placement
# that searched backwards for the user turn missed it. The entry still has to end up in front of the
# generation it governs.
_MID_CONVERSATION_SYSTEM_HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
    ModelResponse(parts=[TextPart(content=_ANSWER)]),
    ModelRequest(parts=[SystemPromptPart(content='Be terse.'), UserPromptPart(content=_FOLLOW_UP)]),
    ModelResponse(parts=[ThinkingPart(content=_REASONING), TextPart(content=_ANSWER)]),
]
# A response whose only part is the `ThinkingPart` renders no assistant turn at all.
_THINKING_ONLY_HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
    ModelResponse(parts=[ThinkingPart(content=_REASONING)]),
]
# The response ahead of the reasoning called a server tool it never resolved. Anthropic rejects a
# continued conversation carrying that call at all, whatever else is in it, so the mapper drops the call
# before it reaches the wire — which leaves the reasoning free to take its usual user turn.
_UNRESOLVED_SERVER_TOOL_HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
    ModelResponse(
        parts=[
            NativeToolCallPart(
                tool_name='web_search',
                args={'query': '10-year Treasury duration'},
                tool_call_id='srvtoolu_01EoSNE7k4dUJyGatASCV5qs',
                provider_name='anthropic',
            ),
            ToolCallPart(tool_name='lookup', args={'tenor': 10}, tool_call_id='call-1'),
        ]
    ),
    ModelRequest(parts=[ToolReturnPart(tool_name='lookup', content='8.1', tool_call_id='call-1')]),
    ModelResponse(parts=[ThinkingPart(content=_REASONING), TextPart(content=_ANSWER)]),
]
# A `CachePoint` marks the end of the cacheable prefix, and the `system` entry the same request renders sits
# behind it. The carried reasoning has to keep that boundary where the user authored it instead of pushing
# the `cache_control` onto an earlier message.
_CACHE_POINT_HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content='Q1')]),
    ModelResponse(parts=[TextPart(content='A1')]),
    ModelRequest(parts=[SystemPromptPart(content='Be terse.'), UserPromptPart(content=['Q2', CachePoint()])]),
    ModelResponse(parts=[ThinkingPart(content=_REASONING), TextPart(content='A2')]),
]
# Back-to-back `ModelResponse`s: the second one's reasoning has no request of its own to sit behind.
_BACK_TO_BACK_RESPONSE_HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
    ModelResponse(parts=[TextPart(content=_ANSWER)]),
    ModelResponse(parts=[ThinkingPart(content=_REASONING), TextPart(content=_ANSWER)]),
]


@dataclass
class Case:
    id: str
    outbound: Callable[[Case], Awaitable[list[dict[str, object]]]]
    history: list[ModelMessage]
    carrying_roles: set[str]
    """The roles of the outbound messages expected to carry the reasoning."""
    expected: object
    model_name: str | None = None
    """The model to build; unset takes the provider builder's own default."""
    profile: ModelProfile | None = None
    """The profile to build the model with; unset takes the one the provider picks for `model_name`."""
    marks: tuple[pytest.MarkDecorator, ...] = field(default_factory=tuple)


async def _anthropic_outbound(case: Case) -> list[dict[str, object]]:
    model = AnthropicModel(
        case.model_name or 'claude-sonnet-4-5', provider=AnthropicProvider(api_key='x'), profile=case.profile
    )
    _, messages = await model._map_message(  # pyright: ignore[reportPrivateUsage]
        case.history, ModelRequestParameters(), AnthropicModelSettings()
    )
    return [dict(message) for message in messages]


async def _xai_outbound(case: Case) -> list[dict[str, object]]:
    model = XaiModel(
        case.model_name or 'grok-4-fast-reasoning', provider=XaiProvider(api_key='x'), profile=case.profile
    )
    messages = await model._map_messages(case.history, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    return [MessageToDict(m, preserving_proto_field_name=True) for m in messages]


async def _bedrock_outbound(case: Case) -> list[dict[str, object]]:
    model = BedrockConverseModel(
        case.model_name or 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
        provider=BedrockProvider(api_key='x', region_name='us-east-1'),
        profile=case.profile,
    )
    _, messages = await model._map_messages(case.history, ModelRequestParameters(), None)  # pyright: ignore[reportPrivateUsage]
    return [dict(message) for message in messages]


# With the flag off, the reasoning goes back in the assistant turn — the documented opt-out.
_OPTED_OUT_PROFILE = ModelProfile(mimics_assistant_message_formatting=False)
# Only a model whose profile takes mid-conversation system prompts inline grows a `system` entry on the wire.
_INLINE_SYSTEM_MODEL = 'claude-opus-4-8'


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
        _anthropic_outbound,
        _FOREIGN_HISTORY,
        carrying_roles={'assistant'},
        profile=_OPTED_OUT_PROFILE,
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
    Case(
        'anthropic-tool-result-turn',
        _anthropic_outbound,
        _TOOL_RESULT_HISTORY,
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
                    'role': 'assistant',
                    'content': [{'id': 'call-1', 'type': 'tool_use', 'name': 'lookup', 'input': {'tenor': 10}}],
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            'tool_use_id': 'call-1',
                            'type': 'tool_result',
                            'content': [{'text': '8.1', 'type': 'text'}],
                            'is_error': False,
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
        'anthropic-mid-conversation-system',
        _anthropic_outbound,
        _MID_CONVERSATION_SYSTEM_HISTORY,
        carrying_roles={'user'},
        model_name=_INLINE_SYSTEM_MODEL,
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
                {'role': 'user', 'content': [{'text': 'And a 30-year?', 'type': 'text'}]},
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
                {'role': 'system', 'content': [{'text': 'Be terse.', 'type': 'text'}]},
                {
                    'role': 'assistant',
                    'content': [{'text': 'The 10-year Treasury has more interest-rate risk.', 'type': 'text'}],
                },
            ]
        ),
        marks=(pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed'),),
    ),
    Case(
        'anthropic-thinking-only-response',
        _anthropic_outbound,
        _THINKING_ONLY_HISTORY,
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
            ]
        ),
        marks=(pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed'),),
    ),
    Case(
        'anthropic-back-to-back-responses',
        _anthropic_outbound,
        _BACK_TO_BACK_RESPONSE_HISTORY,
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
                    'role': 'assistant',
                    'content': [{'text': 'The 10-year Treasury has more interest-rate risk.', 'type': 'text'}],
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
        'bedrock-tool-result-turn',
        _bedrock_outbound,
        _TOOL_RESULT_HISTORY,
        carrying_roles={'user'},
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
                    'content': [{'toolUse': {'toolUseId': 'call-1', 'name': 'lookup', 'input': {'tenor': 10}}}],
                },
                {
                    'role': 'user',
                    'content': [
                        {'toolResult': {'toolUseId': 'call-1', 'content': [{'text': '8.1'}], 'status': 'success'}},
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
        'anthropic-unresolved-server-tool-still-carries-in-user-turn',
        _anthropic_outbound,
        _UNRESOLVED_SERVER_TOOL_HISTORY,
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
                    'role': 'assistant',
                    'content': [{'id': 'call-1', 'type': 'tool_use', 'name': 'lookup', 'input': {'tenor': 10}}],
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            'tool_use_id': 'call-1',
                            'type': 'tool_result',
                            'content': [{'text': '8.1', 'type': 'text'}],
                            'is_error': False,
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
        'anthropic-cache-point-keeps-boundary',
        _anthropic_outbound,
        _CACHE_POINT_HISTORY,
        carrying_roles={'user'},
        model_name=_INLINE_SYSTEM_MODEL,
        expected=snapshot(
            [
                {'role': 'user', 'content': [{'text': 'Q1', 'type': 'text'}]},
                {'role': 'assistant', 'content': [{'text': 'A1', 'type': 'text'}]},
                {'role': 'user', 'content': [{'text': 'Q2', 'type': 'text'}]},
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
                    'role': 'system',
                    'content': [
                        {'text': 'Be terse.', 'type': 'text', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}}
                    ],
                },
                {'role': 'assistant', 'content': [{'text': 'A2', 'type': 'text'}]},
            ]
        ),
        marks=(pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed'),),
    ),
    Case(
        'bedrock-foreign-opted-out',
        _bedrock_outbound,
        _FOREIGN_HISTORY,
        carrying_roles={'assistant'},
        profile=_OPTED_OUT_PROFILE,
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
<thinking>
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
async def test_foreign_thinking_placement(case: Case):
    """Reasoning that can't ride the native channel reaches models that imitate formatting as user content,
    and every other model as assistant content, in both cases wrapped in the profile's thinking tags."""
    body = await case.outbound(case)
    assert body == case.expected

    # `ROLE_USER`/`ROLE_ASSISTANT` on the xAI wire, `user`/`assistant` on the other two.
    assert {
        str(message['role']).lower().removeprefix('role_')
        for message in body
        if _REASONING in json.dumps(message, default=str)
    } == case.carrying_roles
