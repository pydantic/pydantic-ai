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

`tool_choice` and `thinking` are excluded and stay hand-maintained: `Model.prepare_request` moves
both onto `ModelRequestParameters`, from where they reach every adapter whether it honors them or
not, so a payload diff cannot tell support from indifference.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

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

with try_import() as xai_available:
    from xai_sdk import AsyncClient as AsyncXaiClient

    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.providers.xai import XaiProvider

with try_import() as mcp_available:
    from mcp import ServerSession

    from pydantic_ai.models.mcp_sampling import MCPSamplingModel

pytestmark = pytest.mark.anyio


HAND_MAINTAINED = frozenset({'tool_choice', 'thinking'})
"""Fields a payload diff cannot adjudicate; see the module docstring."""

PROBE_VALUES: dict[str, tuple[Any, ...]] = {
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


def parse_supported_by_lists() -> dict[str, list[str]]:
    """Read every `Supported by:` list out of the `ModelSettings` field docstrings.

    Parses the source because a `TypedDict` field's docstring is a bare string expression that Python
    discards at import time, leaving nothing to introspect at runtime.
    """
    module_path = Path(__file__).parents[2] / 'pydantic_ai_slim' / 'pydantic_ai' / 'settings.py'
    class_def = next(
        node
        for node in ast.parse(module_path.read_text(encoding='utf-8')).body
        if isinstance(node, ast.ClassDef) and node.name == 'ModelSettings'
    )

    lists: dict[str, list[str]] = {}
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
            lists[field_name] = _parse_bullets(node.value.value)
            field_name = None
    return lists


def _parse_bullets(docstring: str) -> list[str]:
    """Take the model names out of a `Supported by:` block, dropping any parenthetical caveat.

    Only the first paragraph after the heading is the list: a caveat long enough to wrap continues
    its bullet on the next line, and a blank line ends the list rather than a note that follows it.
    """
    _, _, tail = docstring.partition('Supported by:')
    block, _, _ = tail.strip('\n').partition('\n\n')
    return [
        stripped[2:].split(' (')[0].strip()
        for line in block.splitlines()
        if (stripped := line.strip()).startswith('* ')
    ]


class ProbeAborted(Exception):
    """Raised once the payload is recorded, to stop short of an actual request."""


@dataclass
class Recorder:
    """Collects one canonical string per outgoing request, for baseline-vs-probe comparison."""

    payloads: list[str] = field(default_factory=list[str])

    def record(self, payload: dict[str, Any]) -> None:
        self.payloads.append(json.dumps(payload, sort_keys=True, default=repr))

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
    {'authorization', 'x-api-key', 'x-goog-api-key', 'content-length', 'user-agent', 'x-stainless-retry-count'}
)
BEDROCK_VOLATILE = HTTP_VOLATILE | {
    'x-amz-date',
    'x-amz-content-sha256',
    # botocore stamps a fresh UUID and an attempt counter on every request; left in, they make each
    # probe differ from the baseline and every field look forwarded.
    'amz-sdk-invocation-id',
    'amz-sdk-request',
}

Probe = Callable[[ModelSettings], Any]
"""Takes the settings to probe with, returns an awaitable of the canonical payload (or `None`)."""


def http_probe(build: Callable[[httpx.AsyncClient], Model]) -> Probe:
    """Probe any model whose provider accepts an `http_client`, recording the whole request.

    Body alone is not enough: `timeout` rides in `httpx.Request.extensions` and `extra_headers` in
    the headers, so neither would ever show up in a body-only diff.
    """

    async def probe(settings: ModelSettings) -> str | None:
        recorder = Recorder()

        def handle(request: httpx.Request) -> httpx.Response:
            request.read()
            recorder.record(
                {
                    'body': request.content.decode('utf8', 'replace'),
                    'headers': sorted(f'{k}:{v}' for k, v in request.headers.items() if k not in HTTP_VOLATILE),
                    'timeout': request.extensions.get('timeout'),
                }
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
    recorder = Recorder()

    async def create(**kwargs: Any) -> Any:
        recorder.record(kwargs)
        raise ProbeAborted

    class _Client:
        model = 'probe'
        chat = type('Chat', (), {'completions': type('Completions', (), {'create': staticmethod(create)})})

    model = HuggingFaceModel(
        'probe-model',
        provider=HuggingFaceProvider(api_key=PROBE_KEY, hf_client=cast(AsyncInferenceClient, _Client())),
    )
    await run_probe_request(model, settings)
    return recorder.first


async def xai_probe(settings: ModelSettings) -> str | None:
    """Probe xAI at the SDK call — its transport is gRPC, so there is no HTTP request to read."""
    recorder = Recorder()

    def create(**kwargs: Any) -> Any:
        recorder.record(kwargs)
        raise ProbeAborted

    class _Client:
        chat = type('Chat', (), {'create': staticmethod(create)})

    model = XaiModel('grok-4', provider=XaiProvider(xai_client=cast(AsyncXaiClient, _Client())))
    await run_probe_request(model, settings)
    return recorder.first


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


@dataclass(frozen=True)
class Case:
    """One Model class, the `Supported by:` names that cover it, and how to probe it."""

    id: str
    names: tuple[str, ...]
    probe: Probe
    marks: tuple[pytest.MarkDecorator, ...] = ()


def _openai_chat(client: httpx.AsyncClient) -> Model:
    return OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=PROBE_KEY, http_client=client))


def _openai_responses(client: httpx.AsyncClient) -> Model:
    return OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=PROBE_KEY, http_client=client))


def _cerebras(client: httpx.AsyncClient) -> Model:
    return CerebrasModel('llama3.1-8b', provider=CerebrasProvider(api_key=PROBE_KEY, http_client=client))


def _crusoe(client: httpx.AsyncClient) -> Model:
    return CrusoeModel('openai/gpt-oss-120b', provider=CrusoeProvider(api_key=PROBE_KEY, http_client=client))


def _ollama(client: httpx.AsyncClient) -> Model:
    return OllamaModel(
        'llama3.2', provider=OllamaProvider(base_url='http://probe/v1', api_key=PROBE_KEY, http_client=client)
    )


def _openrouter(client: httpx.AsyncClient) -> Model:
    return OpenRouterModel('openai/gpt-4o', provider=OpenRouterProvider(api_key=PROBE_KEY, http_client=client))


def _snowflake(client: httpx.AsyncClient) -> Model:
    return SnowflakeModel(
        'llama3.1-70b', provider=SnowflakeProvider(account='probe', token=PROBE_KEY, http_client=client)
    )


def _zai(client: httpx.AsyncClient) -> Model:
    return ZaiModel('glm-4.6', provider=ZaiProvider(api_key=PROBE_KEY, http_client=client))


def _bedrock_mantle_chat(client: httpx.AsyncClient) -> Model:
    # `gpt-oss-safeguard-*` are the only Mantle models served on the Chat Completions interface.
    return BedrockMantleChatModel(
        'openai.gpt-oss-safeguard-20b',
        provider=BedrockMantleProvider(api_key=PROBE_KEY, region_name='us-east-1', http_client=client),
    )


def _bedrock_mantle_responses(client: httpx.AsyncClient) -> Model:
    # GPT-5.4 keeps reasoning off by default, so sampling params are not dropped before the wire.
    return BedrockMantleResponsesModel(
        'openai.gpt-5.4',
        provider=BedrockMantleProvider(api_key=PROBE_KEY, region_name='us-east-1', http_client=client),
    )


def _anthropic(client: httpx.AsyncClient) -> Model:
    return AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=PROBE_KEY, http_client=client))


def _groq(client: httpx.AsyncClient) -> Model:
    return GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(api_key=PROBE_KEY, http_client=client))


def _mistral(client: httpx.AsyncClient) -> Model:
    return MistralModel('mistral-large-latest', provider=MistralProvider(api_key=PROBE_KEY, http_client=client))


def _cohere(client: httpx.AsyncClient) -> Model:
    return CohereModel('command-r-plus', provider=CohereProvider(api_key=PROBE_KEY, http_client=client))


def _google(client: httpx.AsyncClient) -> Model:
    return GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key=PROBE_KEY, http_client=client))


CASES = [
    Case('OpenAIChatModel', ('OpenAI', 'OpenAI Chat Completions'), http_probe(_openai_chat)),
    Case('OpenAIResponsesModel', ('OpenAI',), http_probe(_openai_responses)),
    Case('CerebrasModel', ('Cerebras',), http_probe(_cerebras)),
    Case('CrusoeModel', ('Crusoe',), http_probe(_crusoe)),
    Case('OllamaModel', ('Ollama',), http_probe(_ollama)),
    Case('OpenRouterModel', ('OpenRouter',), http_probe(_openrouter)),
    Case('SnowflakeModel', ('Snowflake',), http_probe(_snowflake)),
    Case('ZaiModel', ('Z.AI',), http_probe(_zai)),
    Case(
        'BedrockMantleChatModel',
        ('Bedrock Mantle', 'Bedrock Mantle Chat Completions'),
        http_probe(_bedrock_mantle_chat),
    ),
    Case('BedrockMantleResponsesModel', ('Bedrock Mantle',), http_probe(_bedrock_mantle_responses)),
    Case(
        'AnthropicModel',
        ('Anthropic',),
        http_probe(_anthropic),
        (pytest.mark.skipif(not anthropic_available, reason='anthropic not installed'),),
    ),
    Case(
        'GroqModel',
        ('Groq',),
        http_probe(_groq),
        (pytest.mark.skipif(not groq_available, reason='groq not installed'),),
    ),
    Case(
        'MistralModel',
        ('Mistral',),
        http_probe(_mistral),
        (pytest.mark.skipif(not mistral_available, reason='mistral not installed'),),
    ),
    Case(
        'CohereModel',
        ('Cohere',),
        http_probe(_cohere),
        (pytest.mark.skipif(not cohere_available, reason='cohere not installed'),),
    ),
    Case(
        'GoogleModel',
        ('Google',),
        http_probe(_google),
        (pytest.mark.skipif(not google_available, reason='google not installed'),),
    ),
    Case(
        'BedrockConverseModel',
        ('Bedrock',),
        bedrock_probe,
        (pytest.mark.skipif(not bedrock_available, reason='bedrock not installed'),),
    ),
    Case(
        'HuggingFaceModel',
        ('HuggingFace',),
        huggingface_probe,
        (pytest.mark.skipif(not huggingface_available, reason='huggingface not installed'),),
    ),
    Case('XaiModel', ('xAI',), xai_probe, (pytest.mark.skipif(not xai_available, reason='xai not installed'),)),
    Case(
        'MCPSamplingModel',
        ('MCP Sampling',),
        mcp_sampling_probe,
        (pytest.mark.skipif(not mcp_available, reason='mcp not installed'),),
    ),
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
            if payload is not None and payload != baseline:
                forwarded.add(field_name)
                break
    return forwarded


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id, marks=case.marks) for case in CASES])
async def test_supported_by_lists_match_the_wire(case: Case, allow_model_requests: None):
    """Every `Supported by:` list names exactly the models that send the field."""
    baseline = await case.probe({})
    assert baseline is not None, f'{case.id} sent nothing for the probe to record'

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
