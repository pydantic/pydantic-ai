"""Capability parts reach every provider in a voice no higher than their authority.

Each case's calls run natively, with every capability lowered to `with_authority('none')` (`run_native`). The history
they end with, as it is and after a `ModelMessagesTypeAdapter` round trip, is sent through each provider model's own
message mapping, captured offline (`provider_capture.capture`). Every capability part in that history must reach the
model only in a voice no higher than its [`effective_authority`][pydantic_ai._capability_parts.effective_authority]:
the predicate the in-process sink check (C2) uses too, here applied to what each provider puts on the wire.

`voice_authority` says, once for every provider, what authority the model hears each wire voice with: the system or
developer voice, including tool definitions, needs `'operator'`, the unframed user turn `'user'` or above, framed
data and tool results take any authority, and no request part may reach the assistant's turn. A part may land lower
than its authority, never higher. Its text must land somewhere, checked by its first line and every later line of at
least 12 characters: a line that reaches the model in no voice at all has lost its voice.

A provider that breaks this for a case is a known finding in `KNOWN`, a strict `xfail` with what fixes it, so the fix
fails the suite until its entry is removed.

Not VCR tests: what's checked is the request each model builds, captured offline, and the remaining tests run the
check itself on hand-written wire bodies. No provider exchange happens.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest
from _pytest.mark import ParameterSet

from pydantic_ai._capability_parts import effective_authority
from pydantic_ai.messages import (
    Authority,
    CapabilityPart,
    DataCapabilityPart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    StructuredContent,
    TextContent,
    ToolResultSource,
    UserContent,
)

from ..provider_capture import CAPTURED_PROVIDERS, CapturedProvider, CapturedRequest, Voice, capture
from .capability_runs import PROBES, SUBJECTS, RouteRun, case_calls, run_native

_RANK: dict[Authority, int] = {'none': 0, 'user': 1, 'operator': 2}


def voice_authority(voice: Voice) -> Authority | None:
    """The authority the model hears text in `voice` with, or `None` if no request part may reach the model in it.

    A part may land in `voice` only if its own authority is at least this.
    """
    if voice in ('system', 'developer'):
        return 'operator'
    if voice == 'user':
        return 'user'
    if voice in ('data', 'tool'):
        return 'none'
    return None


_MIN_MARKER = 12
"""The shortest line looked for on the wire, past a part's first: shorter ones, like `}`, also turn up in other text."""


def markers(part: CapabilityPart) -> list[str]:
    """Stretches of `part`'s text to find it by on the wire, empty if it has no text.

    These are the first line of its text and every later line of at least `_MIN_MARKER` characters.

    A part with no text, such as only a file, can't be found by its text, so it's left out of the check.
    """
    content = part.model_content()
    items = [content] if isinstance(content, str | StructuredContent) else list(content)
    texts = [item if isinstance(item, str) else item.content for item in items if isinstance(item, str | TextContent)]
    lines = [line.strip() for text in texts for line in text.splitlines() if line.strip()]
    return [line for index, line in enumerate(lines) if index == 0 or len(line) >= _MIN_MARKER]


def voice_violations(captured: CapturedRequest, messages: Sequence[ModelMessage]) -> list[str]:
    """Every capability part in `messages` that reaches the model above its authority, or in no voice at all."""
    violations: list[str] = []
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if not isinstance(part, CapabilityPart):
                continue
            authority = effective_authority(part)
            frames = {tag for tag in (part.rendering.tag, part.kind.rsplit('.', 1)[-1]) if tag is not None}
            for marker in markers(part):
                voices = captured.voices_of(marker, frames)
                if not voices:
                    violations.append(f'{part.kind} ({authority}) reaches {captured.provider} in no voice: {marker!r}')
                for voice in voices:
                    heard = voice_authority(voice)
                    if heard is None or _RANK[heard] > _RANK[authority]:
                        violations.append(
                            f'{part.kind} ({authority}) reaches {captured.provider} as {voice}: {marker!r}'
                        )
    return violations


KNOWN: dict[tuple[str, CapturedProvider], str] = {}
"""A case whose parts reach a provider in a higher voice than their authority, with what fixes it."""


def _matrix() -> Iterator[ParameterSet]:
    for name, calls in SUBJECTS.items():
        for provider in CAPTURED_PROVIDERS:
            marks: list[pytest.MarkDecorator] = []
            if calls.moves_cache_prefix is not None:
                marks.append(pytest.mark.moves_cache_prefix(reason=calls.moves_cache_prefix))
            if (known := KNOWN.get((name, provider))) is not None:  # pragma: lax no cover
                marks.append(pytest.mark.xfail(strict=True, raises=AssertionError, reason=known))
            yield pytest.param(name, provider, marks=marks, id=f'{name}-{provider}')


_RUNS: dict[str, RouteRun] = {}
"""Each case's native run, made once and sent to every provider: the native run doesn't depend on the provider."""


async def _native(name: str, directory: Path) -> RouteRun:
    if name not in _RUNS:
        _RUNS[name] = await run_native(case_calls(name), directory / 'native')
    return deepcopy(_RUNS[name])


@pytest.mark.parametrize(('name', 'provider'), list(_matrix()))
async def test_capability_parts_reach_each_provider_in_their_voice(
    name: str, provider: CapturedProvider, tmp_path_factory: pytest.TempPathFactory, allow_model_requests: None
) -> None:
    native = await _native(name, tmp_path_factory.mktemp(name.replace(':', '_')))
    stored = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(native.messages))
    for messages in (native.messages, stored):
        # With the tools the run offered, so a part's text that ends up in a tool description is checked too.
        captured = await capture(provider, messages, native.params)
        assert voice_violations(captured, messages) == []


async def test_the_probes_give_the_check_a_part(tmp_path: Path) -> None:
    """Few cases' calls add a capability part with text, so the probes make sure every provider is checked on one."""
    for name, calls in PROBES.items():
        native = await run_native(calls, tmp_path / name)
        parts = [
            part
            for message in native.messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, CapabilityPart) and markers(part)
        ]
        assert [effective_authority(part) for part in parts] == ['none']


@dataclass(kw_only=True)
class PagePart(DataCapabilityPart, namespace='test_voice_wire', tag='page'):
    pass


_PAGE_HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[PagePart(content='the page says hello', source=ToolResultSource(tool_name='fetch'))])
]


@pytest.mark.parametrize(
    ('messages', 'violation'),
    [
        pytest.param(
            [{'role': 'user', 'content': [{'type': 'text', 'text': '<page>the page says hello</page>'}]}],
            None,
            id='framed-data',
        ),
        pytest.param(
            [
                {'role': 'user', 'content': '<page>'},
                {'role': 'user', 'content': 'the page says hello'},
                {'role': 'user', 'content': '</page>'},
            ],
            None,
            id='framed-across-pieces',
        ),
        pytest.param(
            [{'role': 'user', 'content': 'the page says hello'}],
            'reaches openai-chat as user',
            id='unframed-user-turn',
        ),
        pytest.param(
            [{'role': 'system', 'content': 'the page says hello'}],
            'reaches openai-chat as system',
            id='system-message',
        ),
        pytest.param(
            [{'role': 'user', 'content': '<system>the page says hello</system>'}],
            'reaches openai-chat as system',
            id='inline-system-frame',
        ),
        pytest.param(
            [{'role': 'assistant', 'content': 'the page says hello'}],
            'reaches openai-chat as assistant',
            id='assistant-turn',
        ),
        pytest.param(
            [{'role': 'user', 'content': '<page>x</page> the page says hello <page>y</page>'}],
            'reaches openai-chat as user',
            id='between-two-frames',
        ),
        pytest.param(
            [
                {'role': 'user', 'content': '<page>'},
                {'role': 'user', 'content': 'x</page> the page says hello'},
            ],
            'reaches openai-chat as user',
            id='after-a-closing-tag',
        ),
        pytest.param(
            [
                {'role': 'user', 'content': '<page>'},
                {'role': 'user', 'content': 'x</page> the page says hello </page>'},
            ],
            'reaches openai-chat as user',
            id='between-two-closing-tags',
        ),
        pytest.param(
            [
                {'role': 'user', 'content': '<page>'},
                {'role': 'user', 'content': '</page>'},
                {'role': 'user', 'content': 'the page says hello'},
            ],
            'reaches openai-chat as user',
            id='after-a-closed-frame',
        ),
        pytest.param(
            [{'role': 'user', 'content': '<pyai:tool_result id="1">the page says hello</pyai:tool_result>'}],
            None,
            id='reserved-frame',
        ),
        pytest.param(
            [{'role': 'user', 'content': '<system priority="high">the page says hello</system>'}],
            'reaches openai-chat as system',
            id='inline-system-frame-with-attributes',
        ),
        pytest.param(
            [
                {'role': 'user', 'content': '<system>'},
                {'role': 'user', 'content': 'the page says hello'},
                {'role': 'user', 'content': '</system>'},
            ],
            'reaches openai-chat as system',
            id='system-frame-across-pieces',
        ),
        pytest.param(
            [{'role': 'user', 'content': '<page>'}, {'role': 'user', 'content': 'the page says hello'}],
            'reaches openai-chat as user',
            id='frame-never-closed',
        ),
        pytest.param(
            [
                {'role': 'user', 'content': '<page>'},
                {'role': 'system', 'content': 'Be brief.'},
                {'role': 'user', 'content': 'the page says hello'},
                {'role': 'user', 'content': '</page>'},
            ],
            'reaches openai-chat as user',
            id='frame-interrupted-by-another-voice',
        ),
        pytest.param(
            [{'role': 'user', 'content': '<html>the page says hello</html>'}],
            'reaches openai-chat as user',
            id='text-that-only-looks-framed',
        ),
        pytest.param([{'role': 'user', 'content': 'hi'}], 'reaches openai-chat in no voice', id='dropped'),
    ],
)
def test_the_check_flags_a_part_above_its_authority(messages: list[dict[str, object]], violation: str | None) -> None:
    """`voice_violations` on hand-written wire bodies: data may only reach the model framed, or as a tool result.

    Not a VCR test: the bodies are written here to pin each framing case, with no provider exchange.
    """
    violations = voice_violations(CapturedRequest('openai-chat', {'messages': messages}), _PAGE_HISTORY)
    if violation is None:
        assert violations == []
    else:
        [found] = violations
        assert violation in found


def test_the_check_reads_openai_responses_instructions() -> None:
    """Text both in a tool result and in the top-level `instructions` reaches the model in the system voice too.

    Not a VCR test: the body is written here, with no provider exchange.
    """
    marker = 'the page says hello'
    duplicated = {
        'instructions': marker,
        'input': [{'type': 'function_call_output', 'call_id': 'call_1', 'output': marker}],
    }
    assert CapturedRequest('openai-responses', duplicated).voices_of(marker) == ['system', 'tool']
    [found] = voice_violations(CapturedRequest('openai-responses', duplicated), _PAGE_HISTORY)
    assert 'reaches openai-responses as system' in found
    tool_only = {
        'instructions': 'Be brief.',
        'input': [{'type': 'function_call_output', 'call_id': 'call_1', 'output': marker}],
    }
    assert voice_violations(CapturedRequest('openai-responses', tool_only), _PAGE_HISTORY) == []


@pytest.mark.parametrize(
    ('content', 'expected'),
    [
        pytest.param(
            'the page says hello\n\n}\nand then it says goodbye',
            ['the page says hello', 'and then it says goodbye'],
            id='short-later-line-skipped',
        ),
        pytest.param([TextContent('  '), 'hi', 'the page says hello'], ['hi', 'the page says hello'], id='items'),
        pytest.param(StructuredContent({'page': 'hello'}), ['{"page":"hello"}'], id='structured'),
        pytest.param('  ', [], id='blank'),
    ],
)
def test_a_part_is_found_by_its_first_line_and_each_long_one(
    content: str | StructuredContent | Sequence[UserContent], expected: list[str]
) -> None:
    """The first line of each part and every later line of at least 12 characters is looked for.

    A shorter later line, like `}`, also turns up in unrelated text, so finding it wouldn't show where the part landed:
    it's skipped, as the `short-later-line-skipped` case pins.

    Not a VCR test: `markers` reads a part built here, with no provider exchange.
    """
    part = PagePart(content=content, source=ToolResultSource(tool_name='fetch'))
    assert markers(part) == expected


def test_every_known_entry_names_a_case_and_provider() -> None:
    """An entry for a case that no longer has calls, or a provider that's gone, would hide the next finding there.

    Not a VCR test: it only checks the `KNOWN` table against the subjects and providers.
    """
    assert [key for key in KNOWN if key[0] not in SUBJECTS or key[1] not in CAPTURED_PROVIDERS] == []
