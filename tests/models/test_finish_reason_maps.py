"""Every finish-reason map has a case, and every finish reason a provider SDK declares is mapped or exempt.

A model reads a provider's finish reason through its map with `.get()`, so a value the SDK adds and the map
lacks leaves the response without a finish reason, and no type checker notices. Each case compares a map with
the type its SDK declares: a new SDK value fails until the map translates it or the case says why it doesn't.

These are unit tests, not VCR tests: they compare the maps with the SDKs' declared types, and no recorded
response carries every value an SDK declares.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import UnionType
from typing import Literal, Union, get_args, get_origin

import pytest

import pydantic_ai
from pydantic_ai.messages import FinishReason
from pydantic_ai.ui.vercel_ai._event_stream import (
    _FINISH_REASON_MAP as vercel_ai_map,  # pyright: ignore[reportPrivateUsage]
)

from ..conftest import try_import
from ..manifests import Exempt, assert_classified

with try_import() as anthropic_available:
    from anthropic.types.beta import BetaStopReason

    from pydantic_ai.models.anthropic._response import FINISH_REASON_MAP as anthropic_map

with try_import() as bedrock_available:
    from mypy_boto3_bedrock_runtime.literals import StopReasonType

    from pydantic_ai.models.bedrock import _FINISH_REASON_MAP as bedrock_map  # pyright: ignore[reportPrivateUsage]

with try_import() as cohere_available:
    from cohere import ChatFinishReason

    from pydantic_ai.models.cohere import _FINISH_REASON_MAP as cohere_map  # pyright: ignore[reportPrivateUsage]

with try_import() as google_available:
    from google.genai.types import FinishReason as GoogleFinishReason

    from pydantic_ai.models.google import _FINISH_REASON_MAP as google_map  # pyright: ignore[reportPrivateUsage]

with try_import() as groq_available:
    from groq.types.chat import chat_completion as groq_completion, chat_completion_chunk as groq_chunk

    from pydantic_ai.models.groq import _FINISH_REASON_MAP as groq_map  # pyright: ignore[reportPrivateUsage]

with try_import() as huggingface_available:
    from huggingface_hub import TextGenerationOutputFinishReason

    from pydantic_ai.models.huggingface import (
        _FINISH_REASON_MAP as huggingface_map,  # pyright: ignore[reportPrivateUsage]
    )

with try_import() as mistral_available:
    from mistralai.client.models import ChatCompletionChoiceFinishReason

    from pydantic_ai.models.mistral import _FINISH_REASON_MAP as mistral_map  # pyright: ignore[reportPrivateUsage]

with try_import() as openai_available:
    from openai.types.chat import chat_completion, chat_completion_chunk
    from openai.types.responses.response import IncompleteDetails
    from openai.types.responses.response_status import ResponseStatus

    from pydantic_ai.models.openai._chat_completions import CHAT_FINISH_REASON_MAP
    from pydantic_ai.models.openai._response_fields import RESPONSES_FINISH_REASON_MAP

with try_import() as xai_available:
    from xai_sdk.proto import sample_pb2

    from pydantic_ai.models.xai import _FINISH_REASON_PROTO_MAP as xai_map  # pyright: ignore[reportPrivateUsage]


def literal_values(annotation: object) -> set[str]:
    """The values of the `Literal`s in `annotation`, looking through unions.

    The other members of a union, such as cohere's `| Any` or mistral's `UnrecognizedStr`, accept any string, so
    they add no value to check.
    """
    origin = get_origin(annotation)
    if origin is Literal:
        return set(get_args(annotation))
    if origin is Union or origin is UnionType:
        return set[str]().union(*(literal_values(arg) for arg in get_args(annotation)))
    return set()


def requires(available: Callable[[], bool], dependency: str) -> tuple[pytest.MarkDecorator, ...]:
    return (pytest.mark.skipif(not available(), reason=f'{dependency} not installed'),)


@dataclass(frozen=True)
class Mapped:
    """An SDK value the map translates."""

    target: str | None

    @property
    def reason(self) -> str:
        return f'maps to {self.target!r}'


@dataclass(frozen=True)
class Case:
    """Checks a map against the type its SDK declares."""

    source: str
    """The SDK type the map's keys come from."""
    sdk_values: Callable[[], Iterable[str]]
    map_items: Callable[[], Iterable[tuple[str, str | None]]]
    unmapped: dict[str, Exempt] = field(default_factory=dict[str, Exempt])
    """SDK values the map leaves out on purpose."""
    unlisted: dict[str, Exempt] = field(default_factory=dict[str, Exempt])
    """Map keys the SDK type doesn't list."""
    marks: tuple[pytest.MarkDecorator, ...] = ()

    @property
    def reason(self) -> str:
        return f'checked against {self.source}'


_PENDING = Exempt('A pending status: the response has no finish reason yet, and `response_status_to_state` carries it.')
_OPENAI_STYLE = Exempt(
    'The chat-completion API types `finish_reason` as `str` and returns OpenAI-style values, as the '
    '`test_huggingface` cassettes record.'
)

MAPS: dict[str, Case | Exempt] = {
    'pydantic_ai.models.anthropic._response.FINISH_REASON_MAP': Case(
        source='anthropic.types.beta.BetaStopReason',
        sdk_values=lambda: literal_values(BetaStopReason),
        map_items=lambda: anthropic_map.items(),
        marks=requires(anthropic_available, 'anthropic'),
    ),
    'pydantic_ai.models.bedrock._FINISH_REASON_MAP': Case(
        source='mypy_boto3_bedrock_runtime.literals.StopReasonType',
        sdk_values=lambda: literal_values(StopReasonType),
        map_items=lambda: bedrock_map.items(),
        marks=requires(bedrock_available, 'boto3 or boto3-stubs'),
    ),
    'pydantic_ai.models.cohere._FINISH_REASON_MAP': Case(
        source='cohere.ChatFinishReason',
        sdk_values=lambda: literal_values(ChatFinishReason),
        map_items=lambda: cohere_map.items(),
        marks=requires(cohere_available, 'cohere'),
    ),
    'pydantic_ai.models.google._FINISH_REASON_MAP': Case(
        source='google.genai.types.FinishReason',
        sdk_values=lambda: (member.value for member in GoogleFinishReason),
        map_items=lambda: google_map.items(),
        unlisted={
            'MODEL_ARMOR': Exempt(
                'Vertex AI returns it when Model Armor blocks a prompt, as '
                '`test_google_model_armor_prompt_template_text_gets_blocked` records, but the SDK enum lacks it.'
            ),
        },
        marks=requires(google_available, 'google-genai'),
    ),
    'pydantic_ai.models.groq._FINISH_REASON_MAP': Case(
        source='groq.types.chat.chat_completion(_chunk).Choice.finish_reason',
        sdk_values=lambda: (
            literal_values(groq_completion.Choice.model_fields['finish_reason'].annotation)
            | literal_values(groq_chunk.Choice.model_fields['finish_reason'].annotation)
        ),
        map_items=lambda: groq_map.items(),
        marks=requires(groq_available, 'groq'),
    ),
    'pydantic_ai.models.huggingface._FINISH_REASON_MAP': Case(
        source='huggingface_hub.TextGenerationOutputFinishReason',
        sdk_values=lambda: literal_values(TextGenerationOutputFinishReason),
        map_items=lambda: huggingface_map.items(),
        unlisted={'stop': _OPENAI_STYLE, 'tool_calls': _OPENAI_STYLE},
        marks=requires(huggingface_available, 'huggingface_hub'),
    ),
    'pydantic_ai.models.mistral._FINISH_REASON_MAP': Case(
        source='mistralai.client.models.ChatCompletionChoiceFinishReason',
        sdk_values=lambda: literal_values(ChatCompletionChoiceFinishReason),
        map_items=lambda: mistral_map.items(),
        marks=requires(mistral_available, 'mistralai'),
    ),
    'pydantic_ai.models.openai._chat_completions.CHAT_FINISH_REASON_MAP': Case(
        source='openai.types.chat.chat_completion(_chunk).Choice.finish_reason',
        sdk_values=lambda: (
            literal_values(chat_completion.Choice.model_fields['finish_reason'].annotation)
            | literal_values(chat_completion_chunk.Choice.model_fields['finish_reason'].annotation)
        ),
        map_items=lambda: CHAT_FINISH_REASON_MAP.items(),
        marks=requires(openai_available, 'openai'),
    ),
    'pydantic_ai.models.openai._response_fields.RESPONSES_FINISH_REASON_MAP': Case(
        source='openai.types.responses IncompleteDetails.reason | ResponseStatus',
        sdk_values=lambda: (
            literal_values(IncompleteDetails.model_fields['reason'].annotation) | literal_values(ResponseStatus)
        ),
        map_items=lambda: RESPONSES_FINISH_REASON_MAP.items(),
        unmapped={
            'in_progress': _PENDING,
            'queued': _PENDING,
            'incomplete': Exempt(
                'The status alone says nothing about why the response stopped: the finish reason is read from '
                '`incomplete_details.reason` first.'
            ),
            'steered': Exempt(
                'Only follows a WebSocket `response.steer` event, which Pydantic AI never sends; the Responses '
                'WebSocket stream raises on the steer events.'
            ),
            'max_messages': Exempt(
                'The SDK declares it without documenting it (its docstring explains only `steered`), so the map '
                'leaves the finish reason unset.'
            ),
        },
        marks=requires(openai_available, 'openai'),
    ),
    'pydantic_ai.models.openrouter._CHAT_FINISH_REASON_MAP': Exempt(
        "Keyed by the `finish_reason` Literal Pydantic AI declares on its own OpenRouter choice models: there's no SDK "
        'type to check it against.'
    ),
    'pydantic_ai.models.xai._FINISH_REASON_PROTO_MAP': Case(
        source='xai_sdk.proto.sample_pb2.FinishReason',
        sdk_values=lambda: sample_pb2.FinishReason.keys(),
        map_items=lambda: ((sample_pb2.FinishReason.Name(value), target) for value, target in xai_map.items()),
        unmapped={
            'REASON_INVALID': Exempt(
                'The proto zero value, which an unfinished sample carries, so intermediate stream chunks map to `None`.'
            ),
        },
        marks=requires(xai_available, 'xai-sdk'),
    ),
    'pydantic_ai.models.zai._ZAI_FINISH_REASON_MAP': Exempt(
        "Keyed by `_ZaiFinishReason`, which Pydantic AI declares from the Z.AI docs: there's no SDK type to check it "
        'against.'
    ),
    'pydantic_ai.ui.vercel_ai._event_stream._FINISH_REASON_MAP': Case(
        source='pydantic_ai.messages.FinishReason',
        sdk_values=lambda: literal_values(FinishReason),
        map_items=lambda: vercel_ai_map.items(),
    ),
}
"""Each finish-reason map in `pydantic_ai`, by qualified name, with the case that checks it or why none can."""


def test_every_finish_reason_map_has_a_case():
    """A new module-level `*FINISH_REASON*MAP` fails until `MAPS` gives it a case or an exemption."""
    package = Path(pydantic_ai.__file__).parent
    found: set[str] = set()
    for path in package.rglob('*.py'):
        source = path.read_text(encoding='utf-8')
        if 'FINISH_REASON' not in source:
            continue
        module = '.'.join(('pydantic_ai', *path.relative_to(package).with_suffix('').parts))
        for node in ast.parse(source).body:
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, ast.AnnAssign)
                else []
            )
            found.update(
                f'{module}.{target.id}'
                for target in targets
                if isinstance(target, ast.Name) and 'FINISH_REASON' in target.id and target.id.endswith('MAP')
            )
    assert_classified(found, MAPS, complete_walk=True, table_name='MAPS')


@pytest.mark.parametrize(
    ('name', 'case'),
    [pytest.param(name, case, id=name, marks=case.marks) for name, case in MAPS.items() if isinstance(case, Case)],
)
def test_every_sdk_finish_reason_is_mapped_or_exempt(name: str, case: Case):
    """A value the SDK adds fails until the map translates it or the case exempts it; a stale entry fails too."""
    sdk_values = set(case.sdk_values())
    finish_map = dict(case.map_items())
    mapped = {value: Mapped(target) for value, target in finish_map.items() if value in sdk_values}
    remapped = mapped.keys() & case.unmapped.keys()
    assert not remapped, f'`{name}` now maps {sorted(remapped)}: remove their `unmapped` entries.'
    assert_classified(sdk_values, mapped | case.unmapped, complete_walk=True, table_name=name)
    assert_classified(
        finish_map.keys() - sdk_values,
        case.unlisted,
        complete_walk=True,
        table_name=f'{name} keys missing from {case.source}',
    )
