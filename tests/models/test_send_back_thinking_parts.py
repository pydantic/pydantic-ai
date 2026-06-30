"""Cross-provider `*_send_back_thinking_parts` history mapping for xai / groq / huggingface.

Companion to `test_anthropic_send_back_thinking_parts` (in `test_anthropic.py`) and
`test_bedrock_send_back_thinking_parts` (in `test_bedrock.py`), which cover the signature-bearing
providers. These three have no native thinking round-trip, so the default `'auto'` drops every
`ThinkingPart` from history and `'tags'` re-renders it as `thinking_tags` text.

The outbound request body is asserted directly via each model's `_map_messages`: provider cassettes
match on method+URI only, so a VCR test would play back green whether the part is dropped or re-rendered.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal

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
from pydantic_ai.profiles import merge_profile

from ..conftest import try_import

with try_import() as xai_imports:
    from google.protobuf.json_format import MessageToDict

    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.profiles.grok import GrokModelProfile
    from pydantic_ai.providers.xai import XaiProvider

with try_import() as groq_imports:
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.profiles.groq import GroqModelProfile
    from pydantic_ai.providers.groq import GroqProvider

with try_import() as huggingface_imports:
    from pydantic_ai.models.huggingface import HuggingFaceModel
    from pydantic_ai.profiles.huggingface import HuggingFaceModelProfile
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

pytestmark = pytest.mark.anyio

# A foreign-provider (`anthropic`) part stands in for the unsigned/foreign case these providers can't send
# natively — the same shape whether it came from a `FallbackModel` chain or a history processor.
_HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content='question')]),
    ModelResponse(parts=[ThinkingPart(content='reasoning', provider_name='anthropic'), TextPart(content='answer')]),
]


_SendBack = Literal['auto', 'tags']


async def _xai_outbound(send_back: _SendBack | None) -> list[dict[str, Any]]:
    profile = XaiModel('grok-4-fast-reasoning', provider=XaiProvider(api_key='x')).profile
    if send_back is not None:
        profile = merge_profile(profile, GrokModelProfile(grok_send_back_thinking_parts=send_back))
    model = XaiModel('grok-4-fast-reasoning', provider=XaiProvider(api_key='x'), profile=profile)
    messages = await model._map_messages(_HISTORY, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    return [MessageToDict(m, preserving_proto_field_name=True) for m in messages]


async def _groq_outbound(send_back: _SendBack | None) -> list[dict[str, Any]]:
    profile = GroqModel('qwen/qwen3-32b', provider=GroqProvider(api_key='x')).profile
    if send_back is not None:
        profile = merge_profile(profile, GroqModelProfile(groq_send_back_thinking_parts=send_back))
    model = GroqModel('qwen/qwen3-32b', provider=GroqProvider(api_key='x'), profile=profile)
    messages = await model._map_messages(_HISTORY, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    return [dict(m) for m in messages]


async def _huggingface_outbound(send_back: _SendBack | None) -> list[dict[str, Any]]:
    profile = HuggingFaceModel('Qwen/Qwen3-235B-A22B', provider=HuggingFaceProvider(api_key='x')).profile
    if send_back is not None:
        profile = merge_profile(profile, HuggingFaceModelProfile(huggingface_send_back_thinking_parts=send_back))
    model = HuggingFaceModel('Qwen/Qwen3-235B-A22B', provider=HuggingFaceProvider(api_key='x'), profile=profile)
    messages = await model._map_messages(_HISTORY, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    return [{k: v for k, v in asdict(m).items() if v is not None} for m in messages]


@dataclass
class _Case:
    id: str
    outbound: Callable[[_SendBack | None], Awaitable[list[dict[str, Any]]]]
    send_back: _SendBack | None
    expected: list[dict[str, Any]]
    marks: tuple[pytest.MarkDecorator, ...] = ()


_xai_mark = pytest.mark.skipif(not xai_imports(), reason='xai not installed')
_groq_mark = pytest.mark.skipif(not groq_imports(), reason='groq not installed')
_huggingface_mark = pytest.mark.skipif(not huggingface_imports(), reason='huggingface not installed')

_CASES = [
    # Default (unset) resolves to `'auto'`: the foreign part is dropped, leaving only the answer.
    _Case(
        'xai-default-drops',
        _xai_outbound,
        None,
        snapshot(
            [
                {'content': [{'text': 'question'}], 'role': 'ROLE_USER'},
                {'content': [{'text': 'answer'}], 'role': 'ROLE_ASSISTANT'},
            ]
        ),
        marks=(_xai_mark,),
    ),
    _Case(
        'xai-auto-drops',
        _xai_outbound,
        'auto',
        snapshot(
            [
                {'content': [{'text': 'question'}], 'role': 'ROLE_USER'},
                {'content': [{'text': 'answer'}], 'role': 'ROLE_ASSISTANT'},
            ]
        ),
        marks=(_xai_mark,),
    ),
    _Case(
        'xai-tags-rerenders',
        _xai_outbound,
        'tags',
        snapshot(
            [
                {'content': [{'text': 'question'}], 'role': 'ROLE_USER'},
                {
                    'content': [
                        {
                            'text': """\
<think>
reasoning
</think>\
"""
                        }
                    ],
                    'role': 'ROLE_ASSISTANT',
                },
                {'content': [{'text': 'answer'}], 'role': 'ROLE_ASSISTANT'},
            ]
        ),
        marks=(_xai_mark,),
    ),
    _Case(
        'groq-default-drops',
        _groq_outbound,
        None,
        snapshot([{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'answer'}]),
        marks=(_groq_mark,),
    ),
    _Case(
        'groq-auto-drops',
        _groq_outbound,
        'auto',
        snapshot([{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'answer'}]),
        marks=(_groq_mark,),
    ),
    _Case(
        'groq-tags-rerenders',
        _groq_outbound,
        'tags',
        snapshot(
            [
                {'role': 'user', 'content': 'question'},
                {
                    'role': 'assistant',
                    'content': """\
<think>
reasoning
</think>

answer\
""",
                },
            ]
        ),
        marks=(_groq_mark,),
    ),
    _Case(
        'hf-default-drops',
        _huggingface_outbound,
        None,
        snapshot([{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'answer'}]),
        marks=(_huggingface_mark,),
    ),
    _Case(
        'hf-auto-drops',
        _huggingface_outbound,
        'auto',
        snapshot([{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'answer'}]),
        marks=(_huggingface_mark,),
    ),
    _Case(
        'hf-tags-rerenders',
        _huggingface_outbound,
        'tags',
        snapshot(
            [
                {'role': 'user', 'content': 'question'},
                {
                    'role': 'assistant',
                    'content': """\
<think>
reasoning
</think>

answer\
""",
                },
            ]
        ),
        marks=(_huggingface_mark,),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id, marks=c.marks) for c in _CASES])
async def test_send_back_thinking_parts(case: _Case):
    outbound = await case.outbound(case.send_back)
    assert outbound == case.expected
