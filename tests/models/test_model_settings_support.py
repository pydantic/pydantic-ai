"""Pin each `ModelSettings` field's `Supported by:` list to what the models actually send.

The lists in `pydantic_ai.settings` are the only place a user can learn whether a general setting
reaches a given model, and an unsupported setting is silently dropped rather than rejected — so a
stale list is indistinguishable from a broken provider. This module derives the truth from the wire
and asserts the docstrings match it exactly, in both directions.

Each probe sends one request per field through an injected recorder that captures the outgoing
payload and then fails the call, and compares that payload against a baseline request carrying no
settings at all. A field counts as forwarded when it moves the payload. Diffing the payload rather
than hunting for the value survives wire renames (`stop_sequences` -> `stop`), unit conversions
(Mistral's `timeout` -> `timeout_ms`), enum remapping (`service_tier`), booleans that carry no
distinguishing value of their own (`parallel_tool_calls`), and the `extra_body` merges the
OpenAI-derived models perform.

This is not a VCR test, and can't be one: a cassette is a frozen recording, so it keeps matching after the
code stops sending a field — the very drift this file exists to catch. Nothing is recorded either, since
every probe aborts before a response. For the same reason it doesn't use the `request_capture` fixture:
that records only path, body and headers on a live transport, while `timeout` is observable only in
`Request.extensions` and the probe needs a `MockTransport` to fail the call without a network.

`tool_choice` and `thinking` are excluded and stay hand-maintained, for different reasons:

- `tool_choice` reaches every adapter through `resolve_tool_choice`, and the adapters that cannot
  express a named subset honor it by *filtering the tool list* instead (Cohere, Mistral). A payload
  diff therefore cannot tell "sent `tool_choice`" from "dropped tools", and no single probe value
  stands in for a field whose value space spans a scalar, a list of names, and `ToolOrOutput`.
- `thinking` is gated per model *name*, not per model class: `Model.prepare_request` only forwards it
  when the resolved profile sets `supports_thinking`, and a `thinking_always_enabled` model (Cohere,
  Mistral's magistral) supports it while sending nothing at all. "Payload unchanged" is therefore not
  evidence of non-support, which is exactly what this harness reads it as.

That per-model-name gating bounds this file generally: each class is probed at ONE representative
model, chosen to exercise the class's full capability, so an entry whose support varies by model
carries a parenthetical caveat in the docstring (`Bedrock (Anthropic and Amazon Nova models only)`).
`_parse_bullets` strips those caveats, which is what keeps the equality assertion meaningful.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import pkgutil
import re
import textwrap
import types
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, cast

import httpx
import httpx2
import pytest

from pydantic_ai import models
from pydantic_ai.direct import model_request
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from ..conftest import try_import

with try_import() as openai_available:
    from pydantic_ai.models.bedrock_mantle import BedrockMantleChatModel, BedrockMantleResponsesModel
    from pydantic_ai.models.cerebras import CerebrasModel
    from pydantic_ai.models.crusoe import CrusoeModel
    from pydantic_ai.models.ollama import OllamaModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.models.snowflake import SnowflakeModel
    from pydantic_ai.models.zai import ZaiModel
    from pydantic_ai.providers.bedrock_mantle import BedrockMantleProvider
    from pydantic_ai.providers.cerebras import CerebrasProvider
    from pydantic_ai.providers.crusoe import CrusoeProvider
    from pydantic_ai.providers.ollama import OllamaProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider
    from pydantic_ai.providers.snowflake import SnowflakeProvider
    from pydantic_ai.providers.zai import ZaiProvider

with try_import() as anthropic_available:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as google_available:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as groq_available:
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

with try_import() as mistral_available:
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider

with try_import() as cohere_available:
    from pydantic_ai.models.cohere import CohereModel
    from pydantic_ai.providers.cohere import CohereProvider

with try_import() as bedrock_available:
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

with try_import() as huggingface_available:
    from huggingface_hub import AsyncInferenceClient

    from pydantic_ai.models.huggingface import HuggingFaceModel
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

    from .test_huggingface import MockHuggingFace

with try_import() as xai_available:
    from xai_sdk import AsyncClient as AsyncXaiClient

    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.providers.xai import XaiProvider

    from .mock_xai import MockXai

with try_import() as mcp_available:
    from mcp import ServerSession

    from pydantic_ai.models.mcp_sampling import MCPSamplingModel

pytestmark = pytest.mark.anyio


HAND_MAINTAINED = frozenset({'tool_choice', 'thinking'})
"""Fields a payload diff cannot adjudicate; see the module docstring."""

PROBE_VALUES: dict[str, tuple[object, ...]] = {
    'max_tokens': (1234567,),
    'temperature': (0.123456,),
    'top_p': (0.234567,),
    'top_k': (4242,),
    'timeout': (987.654,),
    'parallel_tool_calls': (False, True),
    'seed': (424242,),
    'presence_penalty': (0.345678,),
    'frequency_penalty': (0.456789,),
    'logit_bias': ({'424243': 7},),
    'stop_sequences': (['__probe_stop__'],),
    'extra_headers': ({'x-probe-sentinel': 'probe'},),
    'service_tier': ('auto', 'default', 'flex', 'priority'),
    'extra_body': ({'probe_sentinel': 'probe'},),
}
"""Values to try per field; the field is forwarded when any one of them moves the payload.

Several values are needed where one alone is legitimately dropped: every `service_tier` value is
omitted by some model (Anthropic omits `'flex'` and `'priority'`, Bedrock and Google omit `'auto'`),
and `parallel_tool_calls` is a bare boolean whose two values are each a plausible default.
"""

PROBE_TOOL = ToolDefinition(
    name='probe_tool', description='Probe tool.', parameters_json_schema={'type': 'object', 'properties': {}}
)
"""OpenAI and Groq only send `parallel_tool_calls` when the request carries tools."""

PROBE_KEY = 'probe-key'


def _field_docstrings() -> dict[str, str]:
    """Each field's docstring, parsed once from the `ModelSettings` source.

    A `TypedDict` field's docstring is a bare string expression that Python discards at import
    time, leaving nothing to introspect at runtime, so the source is parsed to recover it.
    """
    class_def = ast.parse(textwrap.dedent(inspect.getsource(ModelSettings))).body[0]
    assert isinstance(class_def, ast.ClassDef)

    docstrings: dict[str, str] = {}
    field_name: str | None = None
    for node in class_def.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            field_name = node.target.id
        elif (
            field_name is not None
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            docstrings[field_name] = node.value.value
            field_name = None
    return docstrings


def parse_supported_by_lists() -> dict[str, list[str]]:
    """Read every `Supported by:` list out of the `ModelSettings` field docstrings."""
    return {field: _parse_bullets(doc) for field, doc in _field_docstrings().items()}


def parse_caveats() -> dict[str, list[str]]:
    """Every parenthetical caveat in a `Supported by:` list, keyed by the field it sits on."""
    caveats: dict[str, list[str]] = {}
    for field_name, block in _supported_by_blocks().items():
        caveats[field_name] = [
            stripped[2:].split(' (', 1)[1]
            for line in block.splitlines()
            if (stripped := line.strip()).startswith('* ') and ' (' in stripped
        ]
    return caveats


def _parse_bullets(docstring: str) -> list[str]:
    """Take the model names out of a `Supported by:` block, dropping any parenthetical caveat.

    Only the first paragraph after the heading is the list: a caveat long enough to wrap continues
    its bullet on the next line, and a blank line ends the list rather than a note that follows it.
    """
    block = _supported_by_block(docstring)
    return [
        stripped[2:].split(' (')[0].strip()
        for line in block.splitlines()
        if (stripped := line.strip()).startswith('* ')
    ]


def _supported_by_block(docstring: str) -> str:
    """The bullet paragraph after the heading — a blank line ends it, so a trailing note is excluded."""
    _, _, tail = docstring.partition('Supported by:')
    block, _, _ = tail.strip('\n').partition('\n\n')
    return block


def _supported_by_blocks() -> dict[str, str]:
    """Each field's bullet paragraph, parsed once from the `ModelSettings` source."""
    return {field: _supported_by_block(doc) for field, doc in _field_docstrings().items()}


class ProbeAborted(Exception):
    """Raised once the payload is recorded, to stop short of an actual request."""


def canonical(payload: dict[str, object]) -> str:
    """One order-independent string per request, so baseline and probe compare by value."""
    return json.dumps(payload, sort_keys=True, default=repr)


@dataclass
class Recorder:
    """Collects one canonical string per outgoing request, for baseline-vs-probe comparison."""

    payloads: list[str] = field(default_factory=list[str])

    def record(self, payload: dict[str, object]) -> None:
        self.payloads.append(canonical(payload))

    @property
    def first(self) -> str | None:
        return self.payloads[0] if self.payloads else None


async def run_probe_request(model: Model, settings: ModelSettings) -> None:
    """Make the one request the recorder is there to capture, swallowing its inevitable failure."""
    try:
        await model_request(
            model,
            [ModelRequest.user_text_prompt('probe')],
            model_settings=settings,
            model_request_parameters=ModelRequestParameters(function_tools=[PROBE_TOOL]),
        )
    except Exception:
        pass


HTTP_VOLATILE = frozenset(
    {
        'authorization',
        'x-api-key',
        'x-goog-api-key',
        'content-length',
        'user-agent',
        'x-stainless-retry-count',
        # W3C trace context carries a fresh trace and span id on every request, so leaving these in
        # makes each probe differ from the baseline and every setting look forwarded.
        'traceparent',
        'tracestate',
    }
)
BEDROCK_VOLATILE = HTTP_VOLATILE | {
    'x-amz-date',
    'x-amz-content-sha256',
    # botocore stamps a fresh UUID and an attempt counter on every request; left in, they make each
    # probe differ from the baseline and every field look forwarded.
    'amz-sdk-invocation-id',
    'amz-sdk-request',
}

Probe = Callable[[ModelSettings], Awaitable[str | None]]
"""Takes the settings to probe with, returns the canonical payload the model sent (or `None`)."""


def _request_payload(
    content: bytes, headers: Iterable[tuple[str, str]], timeout_extension: object
) -> dict[str, object]:
    """What one outgoing request contributes to the diff, read the same way for either client family.

    Body alone is not enough: `timeout` rides in `Request.extensions` and `extra_headers` in the
    headers, so neither would ever show up in a body-only diff.
    """
    # A `Timeout` rides in `extensions`, which is where `ModelSettings['timeout']` is observable at
    # all. Pull only the four numbers out: the mapping can also hold objects whose `repr` carries a
    # memory address, which would make every probe differ from the baseline and so make every field
    # look forwarded.
    timeout: str | None = (
        f'{sorted(timeout_extension.items())}'  # pyright: ignore[reportUnknownArgumentType]
        if isinstance(timeout_extension, dict)
        else None
    )
    return {
        # Parsed, so an SDK's key ordering can't read as a difference. Every model class probed
        # through this path posts JSON, so a decode error here is a real surprise.
        'body': json.loads(content),
        'headers': sorted(f'{k}:{v}' for k, v in headers if k not in HTTP_VOLATILE),
        'timeout': timeout,
    }


def http_probe(build: Callable[[httpx2.AsyncClient], Model]) -> Probe:
    """Probe any model whose provider takes the preferred HTTPX2 `http_client`, recording the whole request."""

    async def probe(settings: ModelSettings) -> str | None:
        recorder = Recorder()

        def handle(request: httpx2.Request) -> httpx2.Response:
            request.read()
            recorder.record(
                _request_payload(request.content, request.headers.items(), request.extensions.get('timeout'))
            )
            return httpx2.Response(400, json={'error': {'message': 'probe', 'type': 'probe'}})

        client = httpx2.AsyncClient(transport=httpx2.MockTransport(handle))
        try:
            await run_probe_request(build(client), settings)
        finally:
            await client.aclose()
        return recorder.first

    return probe


def legacy_http_probe(build: Callable[[httpx.AsyncClient], Model]) -> Probe:
    """Probe a model whose SDK still rejects HTTPX2, so its provider takes a legacy `httpx.AsyncClient`.

    For these providers a legacy client is the supported input rather than a deprecated one, so none of
    the migration warnings the migrated providers raise applies here.
    """

    async def probe(settings: ModelSettings) -> str | None:
        recorder = Recorder()

        def handle(request: httpx.Request) -> httpx.Response:
            request.read()
            recorder.record(
                _request_payload(request.content, request.headers.items(), request.extensions.get('timeout'))
            )
            return httpx.Response(400, json={'error': {'message': 'probe', 'type': 'probe'}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        try:
            await run_probe_request(build(client), settings)
        finally:
            await client.aclose()
        return recorder.first

    return probe


async def bedrock_probe(settings: ModelSettings) -> str | None:
    """Probe Bedrock through botocore's `before-send` event — there is no httpx client to inject."""
    recorder = Recorder()
    model = BedrockConverseModel(
        'anthropic.claude-sonnet-4-5-20250929-v1:0',
        provider=BedrockProvider(aws_access_key_id=PROBE_KEY, aws_secret_access_key=PROBE_KEY, region_name='us-east-1'),
    )

    def handle(request: Any, **_: Any) -> None:
        body = request.body
        recorder.record(
            {
                'body': body.decode('utf8', 'replace') if isinstance(body, bytes) else str(body),
                'headers': sorted(f'{k}:{v}' for k, v in request.headers.items() if k.lower() not in BEDROCK_VOLATILE),
            }
        )
        raise ProbeAborted

    for operation in ('Converse', 'ConverseStream'):
        model.client.meta.events.register_last(f'before-send.bedrock-runtime.{operation}', handle)
    await run_probe_request(model, settings)
    return recorder.first


async def huggingface_probe(settings: ModelSettings) -> str | None:
    """Probe HuggingFace at the SDK call — its provider rejects `http_client` outright."""
    recorder = MockHuggingFace(completions=[ProbeAborted()])
    model = HuggingFaceModel(
        'probe-model',
        provider=HuggingFaceProvider(api_key=PROBE_KEY, hf_client=cast(AsyncInferenceClient, recorder)),
    )
    await run_probe_request(model, settings)
    return canonical(recorder.chat_completion_kwargs[0]) if recorder.chat_completion_kwargs else None


async def xai_probe(settings: ModelSettings) -> str | None:
    """Probe xAI at the SDK call — its transport is gRPC, so there is no HTTP request to read."""
    recorder = MockXai(responses=[ProbeAborted()])
    model = XaiModel('grok-4', provider=XaiProvider(xai_client=cast(AsyncXaiClient, recorder)))
    await run_probe_request(model, settings)
    return canonical(recorder.chat_create_kwargs[0]) if recorder.chat_create_kwargs else None


async def mcp_sampling_probe(settings: ModelSettings) -> str | None:
    """Probe MCP sampling at the session call — the payload is an MCP message, not an HTTP request."""
    recorder = Recorder()

    class _Session:
        async def create_message(self, messages: Any, **kwargs: Any) -> Any:
            recorder.record(kwargs)
            raise ProbeAborted

    model = MCPSamplingModel(session=cast(ServerSession, _Session()))
    await run_probe_request(model, settings)
    return recorder.first


def _needs(available: Callable[[], bool], package: str) -> tuple[pytest.MarkDecorator, ...]:
    """Skip a case when its optional SDK isn't installed.

    `try_import` yields a callable, so the flag must be *called*: `not available` would be
    `not <function>`, i.e. always `False`, leaving the case to fail with a `NameError` on the CI
    matrix jobs that don't install the SDK instead of skipping.
    """
    return (pytest.mark.skipif(not available(), reason=f'{package} not installed'),)


@dataclass(frozen=True)
class Case:
    """One Model class, the `Supported by:` names that cover it, and how to probe it."""

    id: str
    names: tuple[str, ...]
    probe: Probe
    marks: tuple[pytest.MarkDecorator, ...] = ()


def _openai_chat(client: httpx2.AsyncClient) -> Model:
    return OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=PROBE_KEY, http_client=client))


def _openai_responses(client: httpx2.AsyncClient) -> Model:
    return OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=PROBE_KEY, http_client=client))


def _cerebras(client: httpx2.AsyncClient) -> Model:
    return CerebrasModel('llama3.1-8b', provider=CerebrasProvider(api_key=PROBE_KEY, http_client=client))


def _crusoe(client: httpx2.AsyncClient) -> Model:
    return CrusoeModel('openai/gpt-oss-120b', provider=CrusoeProvider(api_key=PROBE_KEY, http_client=client))


def _ollama(client: httpx2.AsyncClient) -> Model:
    return OllamaModel(
        'llama3.2', provider=OllamaProvider(base_url='http://probe/v1', api_key=PROBE_KEY, http_client=client)
    )


def _openrouter(client: httpx2.AsyncClient) -> Model:
    return OpenRouterModel('openai/gpt-4o', provider=OpenRouterProvider(api_key=PROBE_KEY, http_client=client))


def _snowflake(client: httpx2.AsyncClient) -> Model:
    return SnowflakeModel(
        'llama3.1-70b', provider=SnowflakeProvider(account='probe', token=PROBE_KEY, http_client=client)
    )


def _zai(client: httpx2.AsyncClient) -> Model:
    return ZaiModel('glm-4.6', provider=ZaiProvider(api_key=PROBE_KEY, http_client=client))


def _bedrock_mantle_chat(client: httpx2.AsyncClient) -> Model:
    # `gpt-oss-safeguard-*` are the only Mantle models served on the Chat Completions interface.
    return BedrockMantleChatModel(
        'openai.gpt-oss-safeguard-20b',
        provider=BedrockMantleProvider(api_key=PROBE_KEY, region_name='us-east-1', http_client=client),
    )


def _bedrock_mantle_responses(client: httpx2.AsyncClient) -> Model:
    # GPT-5.4 keeps reasoning off by default, so sampling params are not dropped before the wire.
    return BedrockMantleResponsesModel(
        'openai.gpt-5.4',
        provider=BedrockMantleProvider(api_key=PROBE_KEY, region_name='us-east-1', http_client=client),
    )


def _anthropic(client: httpx2.AsyncClient) -> Model:
    return AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=PROBE_KEY, http_client=client))


def _groq(client: httpx.AsyncClient) -> Model:
    return GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(api_key=PROBE_KEY, http_client=client))


def _mistral(client: httpx2.AsyncClient) -> Model:
    return MistralModel('mistral-large-latest', provider=MistralProvider(api_key=PROBE_KEY, http_client=client))


def _cohere(client: httpx.AsyncClient) -> Model:
    return CohereModel('command-r-plus', provider=CohereProvider(api_key=PROBE_KEY, http_client=client))


def _google(client: httpx2.AsyncClient) -> Model:
    return GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key=PROBE_KEY, http_client=client))


CASES = [
    Case(
        'OpenAIChatModel',
        ('OpenAI', 'OpenAI Chat Completions'),
        http_probe(_openai_chat),
        _needs(openai_available, 'openai'),
    ),
    Case('OpenAIResponsesModel', ('OpenAI',), http_probe(_openai_responses), _needs(openai_available, 'openai')),
    Case('CerebrasModel', ('Cerebras',), http_probe(_cerebras), _needs(openai_available, 'openai')),
    Case('CrusoeModel', ('Crusoe',), http_probe(_crusoe), _needs(openai_available, 'openai')),
    Case('OllamaModel', ('Ollama',), http_probe(_ollama), _needs(openai_available, 'openai')),
    Case('OpenRouterModel', ('OpenRouter',), http_probe(_openrouter), _needs(openai_available, 'openai')),
    Case('SnowflakeModel', ('Snowflake',), http_probe(_snowflake), _needs(openai_available, 'openai')),
    Case('ZaiModel', ('Z.AI',), http_probe(_zai), _needs(openai_available, 'openai')),
    Case(
        'BedrockMantleChatModel',
        ('Bedrock Mantle', 'Bedrock Mantle Chat Completions'),
        http_probe(_bedrock_mantle_chat),
        _needs(openai_available, 'openai'),
    ),
    Case(
        'BedrockMantleResponsesModel',
        ('Bedrock Mantle',),
        http_probe(_bedrock_mantle_responses),
        _needs(openai_available, 'openai'),
    ),
    Case('AnthropicModel', ('Anthropic',), http_probe(_anthropic), _needs(anthropic_available, 'anthropic')),
    Case('GroqModel', ('Groq',), legacy_http_probe(_groq), _needs(groq_available, 'groq')),
    Case('MistralModel', ('Mistral',), http_probe(_mistral), _needs(mistral_available, 'mistral')),
    Case('CohereModel', ('Cohere',), legacy_http_probe(_cohere), _needs(cohere_available, 'cohere')),
    Case('GoogleModel', ('Google',), http_probe(_google), _needs(google_available, 'google')),
    Case('BedrockConverseModel', ('Bedrock',), bedrock_probe, _needs(bedrock_available, 'bedrock')),
    Case('HuggingFaceModel', ('HuggingFace',), huggingface_probe, _needs(huggingface_available, 'huggingface')),
    Case('XaiModel', ('xAI',), xai_probe, _needs(xai_available, 'xai')),
    Case('MCPSamplingModel', ('MCP Sampling',), mcp_sampling_probe, _needs(mcp_available, 'mcp')),
]


SUPPORTED_BY_LISTS = parse_supported_by_lists()
"""Every field's documented list, parsed once from `settings.py`."""

DOCUMENTED_NAMES = {name for names in SUPPORTED_BY_LISTS.values() for name in names}
PROBED_NAMES = {name for case in CASES for name in case.names}


async def _forwarded_fields(case: Case, baseline: str) -> set[str]:
    """The fields whose presence changes what `case` puts on the wire."""
    forwarded: set[str] = set()
    for field_name, values in PROBE_VALUES.items():
        for value in values:
            settings: ModelSettings = {field_name: value}  # pyright: ignore[reportAssignmentType]
            payload = await case.probe(settings)
            # A probe that recorded nothing means the request died before the recorder ran; scoring
            # that as "not forwarded" would let a harness failure quietly agree with a stale list.
            assert payload is not None, f'{case.id} sent no request while probing {field_name}'
            if payload != baseline:
                forwarded.add(field_name)
                break
    return forwarded


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id, marks=case.marks) for case in CASES])
async def test_supported_by_lists_match_the_wire(case: Case, allow_model_requests: None):
    """Every `Supported by:` list names exactly the models that send the field."""
    baseline = await case.probe({})
    assert baseline is not None, f'{case.id} sent nothing for the probe to record'
    # Without this, a request that varies between identical calls makes every field differ from the
    # baseline, and the whole matrix reads as "forwarded" instead of failing.
    assert await case.probe({}) == baseline, (
        f'{case.id} builds a different request for identical settings, so the probe cannot separate a '
        f'forwarded field from noise'
    )

    forwarded = await _forwarded_fields(case, baseline)
    documented = {name for name in PROBE_VALUES if set(SUPPORTED_BY_LISTS[name]) & set(case.names)}

    assert forwarded == documented, (
        f'{case.id} disagrees with its `Supported by:` lists in `pydantic_ai/settings.py`.\n'
        f'  sends, but {case.names} is missing from: {sorted(forwarded - documented)}\n'
        f'  does not send, but {case.names} is listed on: {sorted(documented - forwarded)}'
    )


def test_hand_maintained_fields_are_the_only_unprobed_ones():
    """Every field is either probed above or explicitly hand-maintained — none silently unchecked."""
    assert set(SUPPORTED_BY_LISTS) == set(PROBE_VALUES) | HAND_MAINTAINED


def test_every_documented_name_is_probed():
    """No list may name a model the probe does not cover, so a rename cannot go unnoticed."""
    assert DOCUMENTED_NAMES <= PROBED_NAMES, (
        f'named in `Supported by:` lists but never probed: {sorted(DOCUMENTED_NAMES - PROBED_NAMES)}'
    )


_NOT_API_BACKED = frozenset(
    {'WrapperModel', 'InstrumentedModel', 'ConcurrencyLimitedModel', 'FallbackModel', 'FunctionModel', 'TestModel'}
)
"""Model classes that wrap or stand in for another model, so they have no wire of their own to probe."""


def test_every_api_backed_model_class_is_probed():
    """A new `Model` class in a `pydantic_ai.models` module has to join `CASES`.

    The walk is not recursive, matching the package's flat layout; a model added inside a subpackage would
    escape it.

    Derived from the package rather than a hardcoded list because the hardcoded list is the failure
    this file exists to prevent: `CrusoeModel`, `SnowflakeModel` and the Bedrock Mantle models drifted
    out of the `Supported by:` lists exactly by being added and never enumerated anywhere.
    """
    discovered: set[str] = set()
    for module_info in pkgutil.iter_modules(models.__path__, f'{models.__name__}.'):
        try:
            module = importlib.import_module(module_info.name)
        except ImportError:  # pragma: lax no cover
            continue  # an optional provider SDK isn't installed; its classes can't be probed either
        for name, obj in vars(module).items():
            # A subscripted generic (`dict[str, Any]`) passes `isinstance(_, type)` on Python 3.10 and
            # then makes `issubclass` raise, so both it and anything merely imported into the module
            # are screened out before the base-class question is asked.
            if not isinstance(obj, type) or isinstance(obj, types.GenericAlias):
                continue
            if obj.__module__ == module_info.name and issubclass(obj, Model):
                discovered.add(name)

    unprobed = discovered - _NOT_API_BACKED - {case.id for case in CASES}
    assert not unprobed, f'model classes with no probe case, so no `Supported by:` list covers them: {sorted(unprobed)}'


def test_no_caveat_describes_a_different_setting():
    """A caveat naming a `ModelSettings` field must name the field it sits on.

    `_parse_bullets` drops everything after the first `(` so the equality assertion stays about model
    names, which leaves caveat text unchecked — and a bulk edit did once copy `tool_choice`'s caveats
    onto `top_p`, where they read as confident nonsense. This closes that gap.
    """
    fields = set(SUPPORTED_BY_LISTS)
    misplaced = {
        f'{field_name}: {caveat}'
        for field_name, caveats in parse_caveats().items()
        for caveat in caveats
        for referenced in re.findall(r'`([a-z_]+)`', caveat)
        if referenced in fields and referenced != field_name
    }
    assert not misplaced, f'caveats describing another setting: {sorted(misplaced)}'
