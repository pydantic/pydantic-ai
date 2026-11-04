"""A `ThinkingPart` that can't round-trip natively goes in the user turn for models that imitate formatting.

Regression test for [#5869](https://github.com/pydantic/pydantic-ai/issues/5869): an unsigned or
foreign-provider `ThinkingPart` in history (round-tripped through storage, rebuilt by a history processor,
or produced by another model in a `FallbackModel` chain) can't be sent through the model's own native
reasoning channel, so it falls back to text. Claude reads the assistant turns of a history as examples of
how it is supposed to write, so putting that text in the assistant turn teaches it to emit `<thinking>`
tags in the answers the user reads — measured live in
`tests/models/anthropic/test_thinking_tag_mimicry.py`. Models carrying
`mimics_assistant_message_formatting` therefore take it in a user turn instead.

The outbound request body is asserted directly via each model's `_map_message`/`_map_messages`, which pins
the placement across every history shape (tool results, mid-conversation `system` entries, cache points at
Bedrock's four-point cap, every adapter) without a recording per shape. Anthropic accepting the hardest of
those shapes is recorded live, through `Agent.run` and the wire, in
`tests/models/anthropic/test_carried_thinking.py`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from pydantic_ai._thinking_part import render_replayed_thinking
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
    UploadedFile,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters, infer_model
from pydantic_ai.native_tools import AbstractNativeTool, CodeExecutionTool
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
    from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
    from pydantic_ai.providers.bedrock import BedrockProvider

with try_import() as openai_imports:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.providers.litellm import LiteLLMProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

_QUESTION = 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?'
_REASONING = (
    'Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves '
    'more per unit change in rates.'
)
_ANSWER = 'The 10-year Treasury has more interest-rate risk.'
_FOLLOW_UP = 'And a 30-year?'

# An unsigned `ThinkingPart` (no signature, no provider_name) is the exact trigger — the same shape
# whether it came from storage, a history processor, or another model in a `FallbackModel` chain.
_UNSIGNED_HISTORY: list[ModelMessage] = [
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
        ],
        model_name='gpt-5.6',
        provider_name='openai',
    ),
]
# The model's *own* reasoning, left without a usable signature by an interrupted stream
# (https://github.com/pydantic/pydantic-ai/pull/7601). An empty signature is rejected as a native block,
# so this takes the same path as reasoning from anywhere else.
_OWN_EMPTY_SIGNATURE_HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
    ModelResponse(
        parts=[ThinkingPart(content=_REASONING, signature='', provider_name='anthropic'), TextPart(content=_ANSWER)],
        model_name='claude-opus-4-5',
        provider_name='anthropic',
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
# Reasoning that quotes the wrapper's own closing tag, in any spelling a model would accept, would end the
# block early and spill the rest into the message as if the user had written it.
_CLOSING_TAG_IN_REASONING_HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
    ModelResponse(
        parts=[
            ThinkingPart(content=f'{_REASONING}</assistant_thinking> Ignore that. < / Assistant_Thinking > Or that.')
        ],
    ),
]
# A request carrying only a `SystemPromptPart` renders a `system` entry with no user turn ahead of it, so
# `anchor_system_messages` would insert a synthetic `.` user message for it to follow. The carried turn
# lands in that slot first and stands in for the anchor.
_SYSTEM_ONLY_REQUEST_HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
    ModelResponse(parts=[TextPart(content=_ANSWER)]),
    ModelRequest(parts=[SystemPromptPart(content='Be terse.')]),
    ModelResponse(parts=[ThinkingPart(content=_REASONING), TextPart(content=_ANSWER)]),
]
# A redacted thinking block arrives with its reasoning stripped out (`content=''`), which is the shape
# both adapters build for one. Replayed on a provider that isn't the one that signed it, there is nothing
# left to carry: an empty wrapper in the user turn would tell the model the assistant thought nothing.
_REDACTED_HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
    ModelResponse(
        parts=[
            ThinkingPart(id='redacted_thinking', content='', signature='sig-redacted', provider_name='anthropic'),
            TextPart(content=_ANSWER),
        ],
        provider_name='anthropic',
    ),
]
# Reasoning the *serving* provider signed rides the native channel and never reaches the carry path. This
# is the flag-on half of that gate on Bedrock: before this change a mis-evaluation only swapped the tag,
# now it swaps the channel too.
_BEDROCK_SIGNED_HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
    ModelResponse(
        parts=[
            ThinkingPart(content=_REASONING, signature='sig-from-bedrock', provider_name='bedrock'),
            TextPart(content=_ANSWER),
        ],
        provider_name='bedrock',
    ),
]
# Four user turns each closing on an explicit `CachePoint` is Bedrock's whole budget: `_limit_cache_points`
# keeps `MAX_CACHE_POINTS = 4` and strips newest-first, so a fifth point costs `Q0` its marker — the one
# anchoring the longest prefix. Carried reasoning merged *behind* a marker makes the turn end in text
# again, which is what earns the automatic fifth point from `bedrock_cache_messages`.
_BEDROCK_CACHE_SATURATED_HISTORY: list[ModelMessage] = [
    message
    for turn in range(4)
    for message in (
        ModelRequest(parts=[UserPromptPart(content=[f'Q{turn}', CachePoint()])]),
        ModelResponse(parts=[ThinkingPart(content=f'{_REASONING} ({turn})'), TextPart(content=f'A{turn}')]),
    )
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
    expected: list[dict[str, object]]
    model_name: str | None = None
    """The model to build; unset takes the provider builder's own default."""
    profile: ModelProfile | None = None
    """The profile to build the model with; unset takes the one the provider picks for `model_name`."""
    native_tools: list[AbstractNativeTool] = field(default_factory=list[AbstractNativeTool])
    """Native tools the request is sent with; only `CodeExecutionTool.files` changes the placement."""
    bedrock_settings: BedrockModelSettings | None = None
    """Settings for the Bedrock builder; the cache-point cases need `bedrock_cache_messages`."""
    env: dict[str, str] = field(default_factory=dict[str, str])
    """Environment variables the OpenAI Chat builder's provider needs to be built from `model_name`."""
    marks: tuple[pytest.MarkDecorator, ...] = field(default_factory=tuple)


async def _anthropic_outbound(case: Case) -> list[dict[str, object]]:
    model = AnthropicModel(
        case.model_name or 'claude-sonnet-4-5', provider=AnthropicProvider(api_key='x'), profile=case.profile
    )
    _, messages = await model._map_message(  # pyright: ignore[reportPrivateUsage]
        case.history, ModelRequestParameters(native_tools=case.native_tools), AnthropicModelSettings()
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
    _, messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        case.history, ModelRequestParameters(), case.bedrock_settings
    )
    # `_messages_create` runs this over the mapper's output before sending, and it is where an extra
    # cache point turns into a lost one: the cap is four and the newest win, so the block dropped is the
    # oldest explicit marker. Applying it here makes the snapshot the set that actually reaches Bedrock.
    model._limit_cache_points([], messages, [])  # pyright: ignore[reportPrivateUsage]
    return [dict(message) for message in messages]


# With the flag off, the reasoning goes back in the assistant turn — the documented opt-out.
_OPTED_OUT_PROFILE = ModelProfile(mimics_assistant_message_formatting=False)
# Only a model whose profile takes mid-conversation system prompts inline grows a `system` entry on the wire.
_INLINE_SYSTEM_MODEL = 'claude-opus-4-8'


# Claude served through an OpenAI-compatible provider resolves `anthropic_model_profile`, flag included, but
# reaches the wire through `OpenAIChatModel` (https://github.com/pydantic/pydantic-ai/issues/7419).
_OPENAI_CHAT_CLAUDE_MODEL = 'litellm:anthropic/claude-sonnet-4-5'
# Two reasoning blocks in one response, one of them redacted to nothing: the rest share one user message.
_TWO_THINKINGS_ONE_EMPTY_HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
    ModelResponse(
        parts=[
            ThinkingPart(content=_REASONING),
            ThinkingPart(content=''),
            ThinkingPart(content='Longer maturity, longer duration.'),
            TextPart(content=_ANSWER),
        ]
    ),
]


async def _openai_chat_outbound(case: Case) -> list[dict[str, object]]:
    # Each route is built the way a user names it, so the profile is the one that route really resolves.
    # The opt-out needs a profile of its own, which `infer_model` can't take, so it is built on LiteLLM.
    with patch.dict(os.environ, case.env):
        model = (
            infer_model(case.model_name or _OPENAI_CHAT_CLAUDE_MODEL)
            if case.profile is None
            else OpenAIChatModel('anthropic/claude-sonnet-4-5', provider=LiteLLMProvider(), profile=case.profile)
        )
    assert isinstance(model, OpenAIChatModel)
    messages = await model._map_messages(case.history, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    return [dict(message) for message in messages]


async def _openai_responses_outbound(case: Case) -> list[dict[str, object]]:
    # Claude through OpenRouter's Responses API, the combination `docs/models/openrouter.md` shows.
    model = OpenAIResponsesModel(
        case.model_name or 'anthropic/claude-sonnet-4-5', provider=OpenRouterProvider(api_key='x'), profile=case.profile
    )
    _, items = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        case.history, OpenAIResponsesModelSettings(), ModelRequestParameters()
    )
    return [dict(item) for item in items]


_OPENAI_MARKS = (pytest.mark.skipif(not openai_imports(), reason='openai not installed'),)
_OPENAI_CHAT_UNSIGNED_EXPECTED = snapshot(
    [
        {'role': 'user', 'content': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?'},
        {
            'role': 'user',
            'content': """\
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>\
""",
        },
        {'role': 'assistant', 'content': 'The 10-year Treasury has more interest-rate risk.'},
    ]
)
_OPENAI_CASES = [
    *(
        Case(
            f'openai-chat-{route}-unsigned',
            _openai_chat_outbound,
            _UNSIGNED_HISTORY,
            carrying_roles={'user'},
            expected=_OPENAI_CHAT_UNSIGNED_EXPECTED,
            model_name=model_name,
            env=env,
            marks=_OPENAI_MARKS,
        )
        for route, model_name, env in [
            ('litellm', _OPENAI_CHAT_CLAUDE_MODEL, {}),
            ('vercel', 'vercel:anthropic/claude-sonnet-4-5', {'VERCEL_AI_GATEWAY_API_KEY': 'x'}),
            ('heroku', 'heroku:claude-4-5-sonnet', {'HEROKU_INFERENCE_KEY': 'x'}),
            ('snowflake', 'snowflake:claude-sonnet-4-5', {'SNOWFLAKE_TOKEN': 'x', 'SNOWFLAKE_ACCOUNT': 'acct'}),
            ('github-copilot', 'github-copilot:claude-sonnet-4.5', {'GITHUB_COPILOT_API_KEY': 'x'}),
        ]
    ),
    Case(
        'openai-chat-signed-foreign',
        _openai_chat_outbound,
        _SIGNED_FOREIGN_HISTORY,
        carrying_roles={'user'},
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?',
                },
                {
                    'role': 'user',
                    'content': """\
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>\
""",
                },
                {'role': 'assistant', 'content': 'The 10-year Treasury has more interest-rate risk.'},
            ]
        ),
        marks=_OPENAI_MARKS,
    ),
    Case(
        'openai-chat-tool-result-turn',
        _openai_chat_outbound,
        _TOOL_RESULT_HISTORY,
        carrying_roles={'user'},
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?',
                },
                {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [
                        {
                            'id': 'call-1',
                            'type': 'function',
                            'function': {'name': 'lookup', 'arguments': '{"tenor":10}'},
                        }
                    ],
                },
                {'role': 'tool', 'tool_call_id': 'call-1', 'content': '8.1'},
                {
                    'role': 'user',
                    'content': """\
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>\
""",
                },
                {'role': 'assistant', 'content': 'The 10-year Treasury has more interest-rate risk.'},
            ]
        ),
        marks=_OPENAI_MARKS,
    ),
    Case(
        'openai-chat-thinking-only-response',
        _openai_chat_outbound,
        _THINKING_ONLY_HISTORY,
        carrying_roles={'user'},
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?',
                },
                {
                    'role': 'user',
                    'content': """\
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>\
""",
                },
            ]
        ),
        marks=_OPENAI_MARKS,
    ),
    Case(
        'openai-chat-two-thinkings-one-empty',
        _openai_chat_outbound,
        _TWO_THINKINGS_ONE_EMPTY_HISTORY,
        carrying_roles={'user'},
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?',
                },
                {
                    'role': 'user',
                    'content': """\
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>

<assistant_thinking>
Longer maturity, longer duration.
</assistant_thinking>\
""",
                },
                {'role': 'assistant', 'content': 'The 10-year Treasury has more interest-rate risk.'},
            ]
        ),
        marks=_OPENAI_MARKS,
    ),
    Case(
        'openai-chat-unsigned-opted-out',
        _openai_chat_outbound,
        _UNSIGNED_HISTORY,
        carrying_roles={'assistant'},
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?',
                },
                {
                    'role': 'assistant',
                    'content': """\
<thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</thinking>

The 10-year Treasury has more interest-rate risk.\
""",
                },
            ]
        ),
        profile=_OPTED_OUT_PROFILE,
        marks=_OPENAI_MARKS,
    ),
    # OpenRouter's profile sends reasoning back in its `reasoning` field, so nothing is rendered as text.
    Case(
        'openrouter-chat-reasoning-field',
        _openai_chat_outbound,
        _UNSIGNED_HISTORY,
        carrying_roles={'assistant'},
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?',
                },
                {
                    'role': 'assistant',
                    'reasoning': 'Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.',
                    'content': 'The 10-year Treasury has more interest-rate risk.',
                },
            ]
        ),
        model_name='openrouter:anthropic/claude-sonnet-4-5',
        env={'OPENROUTER_API_KEY': 'x'},
        marks=_OPENAI_MARKS,
    ),
    Case(
        'openai-responses-unsigned',
        _openai_responses_outbound,
        _UNSIGNED_HISTORY,
        carrying_roles={'user'},
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?',
                },
                {
                    'role': 'user',
                    'content': """\
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>\
""",
                },
                {'role': 'assistant', 'content': 'The 10-year Treasury has more interest-rate risk.'},
            ]
        ),
        marks=_OPENAI_MARKS,
    ),
    Case(
        'openai-responses-tool-result-turn',
        _openai_responses_outbound,
        _TOOL_RESULT_HISTORY,
        carrying_roles={'user'},
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?',
                },
                {'name': 'lookup', 'arguments': '{"tenor":10}', 'call_id': 'call-1', 'type': 'function_call'},
                {'type': 'function_call_output', 'call_id': 'call-1', 'output': '8.1'},
                {
                    'role': 'user',
                    'content': """\
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>\
""",
                },
                {'role': 'assistant', 'content': 'The 10-year Treasury has more interest-rate risk.'},
            ]
        ),
        marks=_OPENAI_MARKS,
    ),
    Case(
        'openai-responses-thinking-only-response',
        _openai_responses_outbound,
        _THINKING_ONLY_HISTORY,
        carrying_roles={'user'},
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?',
                },
                {
                    'role': 'user',
                    'content': """\
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>\
""",
                },
            ]
        ),
        marks=_OPENAI_MARKS,
    ),
    Case(
        'openai-responses-two-thinkings-one-empty',
        _openai_responses_outbound,
        _TWO_THINKINGS_ONE_EMPTY_HISTORY,
        carrying_roles={'user'},
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?',
                },
                {
                    'role': 'user',
                    'content': """\
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>

<assistant_thinking>
Longer maturity, longer duration.
</assistant_thinking>\
""",
                },
                {'role': 'assistant', 'content': 'The 10-year Treasury has more interest-rate risk.'},
            ]
        ),
        marks=_OPENAI_MARKS,
    ),
    Case(
        'openai-responses-unsigned-opted-out',
        _openai_responses_outbound,
        _UNSIGNED_HISTORY,
        carrying_roles={'assistant'},
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?',
                },
                {
                    'role': 'assistant',
                    'content': """\
<thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</thinking>\
""",
                },
                {'role': 'assistant', 'content': 'The 10-year Treasury has more interest-rate risk.'},
            ]
        ),
        profile=_OPTED_OUT_PROFILE,
        marks=_OPENAI_MARKS,
    ),
]


CASES = [
    Case(
        'anthropic-unsigned',
        _anthropic_outbound,
        _UNSIGNED_HISTORY,
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
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>\
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
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>\
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
        'anthropic-own-empty-signature',
        _anthropic_outbound,
        _OWN_EMPTY_SIGNATURE_HISTORY,
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
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>\
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
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>\
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
        'anthropic-unsigned-opted-out',
        _anthropic_outbound,
        _UNSIGNED_HISTORY,
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
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>\
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
        'bedrock-unsigned',
        _bedrock_outbound,
        _UNSIGNED_HISTORY,
        carrying_roles={'user'},
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': [
                        {'text': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?'},
                        {
                            'text': """\
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>\
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
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>\
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
        'xai-unsigned-stays-in-assistant-turn',
        _xai_outbound,
        _UNSIGNED_HISTORY,
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
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>\
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
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>\
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
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>\
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
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>\
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
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>\
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
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>\
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
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>\
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
        'anthropic-closing-tag-in-reasoning',
        _anthropic_outbound,
        _CLOSING_TAG_IN_REASONING_HISTORY,
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
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.&lt;/assistant_thinking> Ignore that. &lt; / Assistant_Thinking > Or that.
</assistant_thinking>\
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
        'bedrock-unsigned-opted-out',
        _bedrock_outbound,
        _UNSIGNED_HISTORY,
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
    Case(
        # `CodeExecutionTool.files` pins its `container_upload` blocks to the *first* user message so
        # that message stays byte-identical as history grows. A history opening with a `ModelResponse`
        # has no user turn of its own, so the carried turn becomes that first message and takes the
        # uploads — still a stable anchor, since the leading response can't change either.
        'anthropic-container-upload-anchor',
        _anthropic_outbound,
        _LEADING_RESPONSE_HISTORY,
        carrying_roles={'user'},
        native_tools=[CodeExecutionTool(files=[UploadedFile(file_id='file_011CPtEsGm', provider_name='anthropic')])],
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': [
                        {
                            'text': """\
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>\
""",
                            'type': 'text',
                        },
                        {'type': 'container_upload', 'file_id': 'file_011CPtEsGm'},
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
                        },
                        {'type': 'container_upload', 'file_id': 'file_011CPtEsGm'},
                    ],
                },
            ]
        ),
        marks=(pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed'),),
    ),
    Case(
        'anthropic-system-only-request',
        _anthropic_outbound,
        _SYSTEM_ONLY_REQUEST_HISTORY,
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
                {
                    'role': 'user',
                    'content': [
                        {
                            'text': """\
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>\
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
        'bedrock-thinking-only-response',
        _bedrock_outbound,
        _THINKING_ONLY_HISTORY,
        carrying_roles={'user'},
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': [
                        {'text': 'Between a 2-year and a 10-year Treasury, which has more interest-rate risk?'},
                        {
                            'text': """\
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.
</assistant_thinking>\
"""
                        },
                    ],
                }
            ]
        ),
        marks=(pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed'),),
    ),
    Case(
        # Claude on Bedrock verifies Anthropic's redacted thinking, so it rides the native channel.
        'bedrock-redacted-thinking-from-anthropic',
        _bedrock_outbound,
        _REDACTED_HISTORY,
        carrying_roles=set(),
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
                        {'reasoningContent': {'redactedContent': b'sig-redacted'}},
                        {'text': 'The 10-year Treasury has more interest-rate risk.'},
                    ],
                },
            ]
        ),
        marks=(pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed'),),
    ),
    Case(
        'bedrock-signed-native',
        _bedrock_outbound,
        _BEDROCK_SIGNED_HISTORY,
        carrying_roles={'assistant'},
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
                            'reasoningContent': {
                                'reasoningText': {
                                    'text': 'Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates.',
                                    'signature': 'sig-from-bedrock',
                                }
                            }
                        },
                        {'text': 'The 10-year Treasury has more interest-rate risk.'},
                    ],
                },
            ]
        ),
        marks=(pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed'),),
    ),
    Case(
        'bedrock-cache-points-survive-carry',
        _bedrock_outbound,
        _BEDROCK_CACHE_SATURATED_HISTORY,
        carrying_roles={'user'},
        bedrock_settings={'bedrock_cache_messages': True},
        expected=snapshot(
            [
                {
                    'role': 'user',
                    'content': [
                        {'text': 'Q0'},
                        {
                            'text': """\
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates. (0)
</assistant_thinking>\
"""
                        },
                        {'cachePoint': {'type': 'default', 'ttl': '5m'}},
                    ],
                },
                {'role': 'assistant', 'content': [{'text': 'A0'}]},
                {
                    'role': 'user',
                    'content': [
                        {'text': 'Q1'},
                        {
                            'text': """\
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates. (1)
</assistant_thinking>\
"""
                        },
                        {'cachePoint': {'type': 'default', 'ttl': '5m'}},
                    ],
                },
                {'role': 'assistant', 'content': [{'text': 'A1'}]},
                {
                    'role': 'user',
                    'content': [
                        {'text': 'Q2'},
                        {
                            'text': """\
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates. (2)
</assistant_thinking>\
"""
                        },
                        {'cachePoint': {'type': 'default', 'ttl': '5m'}},
                    ],
                },
                {'role': 'assistant', 'content': [{'text': 'A2'}]},
                {
                    'role': 'user',
                    'content': [
                        {'text': 'Q3'},
                        {
                            'text': """\
<assistant_thinking>
Interest-rate risk scales with duration, and duration rises with maturity, so the 10-year moves more per unit change in rates. (3)
</assistant_thinking>\
"""
                        },
                        {'cachePoint': {'type': 'default', 'ttl': '5m'}},
                    ],
                },
                {'role': 'assistant', 'content': [{'text': 'A3'}]},
            ]
        ),
        marks=(pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed'),),
    ),
    *_OPENAI_CASES,
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id, marks=c.marks) for c in CASES])
async def test_thinking_placement(case: Case):
    """Reasoning that can't ride the native channel reaches models that imitate formatting as user content
    wrapped in an `assistant_thinking` tag, and every other model as assistant content in its own thinking tags."""
    body = await case.outbound(case)
    assert body == case.expected

    # `ROLE_USER`/`ROLE_ASSISTANT` on the xAI wire, `user`/`assistant` on the others.
    assert {
        str(message['role']).lower().removeprefix('role_')
        for message in body
        if _REASONING in json.dumps(message, default=str)
    } == case.carrying_roles


def test_carried_thinking_ending_in_a_closing_tag_cannot_close_the_wrapper() -> None:
    """Reasoning that ends mid-tag would otherwise meet the newline ahead of the real closing tag."""
    text, carried = render_replayed_thinking(
        'thinking about </assistant_thinking',
        ModelProfile(mimics_assistant_message_formatting=True),
    )

    assert carried
    assert text == snapshot("""\
<assistant_thinking>
thinking about &lt;/assistant_thinking
</assistant_thinking>\
""")
