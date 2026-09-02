"""Tests for Google Gemini's `VALIDATED` function-calling mode (the `strict` tool flag).

On supported models (Gemini 2.5+), `VALIDATED` is the default — it enforces the declared schema with no
schema rewrites, so it's a safe silent improvement — and a caller opts a tool out with `strict=False`.

Test organization:
1. Mode resolution (unit, against an offline `genai.Client`)
2. `strict` resolution via `GoogleJsonSchemaTransformer`
3. End-to-end wire contract (live recording)
"""

from __future__ import annotations as _annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import httpx2
import pytest
from pydantic import AnyUrl, BaseModel, ConfigDict, Field

from pydantic_ai import Agent
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition, ToolKind

from ..._inline_snapshot import snapshot
from ...conftest import try_import

with try_import() as imports_successful:
    from google.genai import Client

    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

if TYPE_CHECKING:
    GoogleModelFactory = Callable[..., GoogleModel]

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


def make_tool(name: str, *, strict: bool | None = None, kind: ToolKind = 'function') -> ToolDefinition:
    return ToolDefinition(
        name=name,
        parameters_json_schema={'type': 'object', 'properties': {}},
        strict=strict,
        kind=kind,
    )


# =============================================================================
# Mode resolution
# =============================================================================


STRICT_TOOL_CONFIG_CASES = [
    dict(
        # A supported model defaults to `VALIDATED` even with no `strict` flag set: the silent improvement.
        id='default-supported-model-uses-validated',
        model='gemini-2.5-flash',
        function_tools=[make_tool('a'), make_tool('b')],
        settings={},
        expected_mode='VALIDATED',
    ),
    dict(
        id='unsupported-model-stays-auto',
        model='gemini-2.0-flash',
        function_tools=[make_tool('a')],
        settings={},
        expected_mode='AUTO',
    ),
    dict(
        id='explicit-strict-uses-validated',
        model='gemini-2.5-flash',
        function_tools=[make_tool('a', strict=True), make_tool('b', strict=True)],
        settings={},
        expected_mode='VALIDATED',
    ),
    dict(
        # A single tool opting out with `strict=False` drops the whole request back to `AUTO`.
        id='opt-out-tool-stays-auto',
        model='gemini-2.5-flash',
        function_tools=[make_tool('a'), make_tool('b', strict=False)],
        settings={},
        expected_mode='AUTO',
    ),
    dict(
        id='required-tool-choice-stays-any',
        model='gemini-2.5-flash',
        function_tools=[make_tool('a')],
        settings={'tool_choice': 'required'},
        expected_mode='ANY',
    ),
    dict(
        id='none-tool-choice-stays-none',
        model='gemini-2.5-flash',
        function_tools=[make_tool('a')],
        settings={'tool_choice': 'none'},
        expected_mode='NONE',
    ),
    dict(
        # `tool_defs` spans function *and* output tools; a default output tool doesn't block `VALIDATED`, so a
        # plain `output_type` still gets the benefit (no need to set `strict=True` on every tool).
        id='default-output-tool-uses-validated',
        model='gemini-2.5-flash',
        function_tools=[make_tool('a')],
        output_tools=[make_tool('final_result', kind='output')],
        settings={},
        expected_mode='VALIDATED',
    ),
    dict(
        # An output tool opting out with `strict=False` drops the request to `AUTO`, same as a function tool.
        id='opt-out-output-tool-stays-auto',
        model='gemini-2.5-flash',
        function_tools=[make_tool('a')],
        output_tools=[make_tool('final_result', strict=False, kind='output')],
        settings={},
        expected_mode='AUTO',
    ),
]


@pytest.mark.parametrize('case', STRICT_TOOL_CONFIG_CASES, ids=lambda c: c['id'])
def test_google_strict_tools_upgrade_auto_to_validated(case: dict[str, Any]):
    """On a supported model, `AUTO` is upgraded to Gemini's `VALIDATED` mode unless a tool (function *or*
    output) opts out with `strict=False`; `required`/`none` tool choices are never upgraded.

    Asserted on the request shape directly rather than via VCR: a cassette replay can't catch the mode we send,
    since it replays a recorded response without re-validating the request against the API.
    """
    m = GoogleModel(case['model'], provider=GoogleProvider(client=Client(vertexai=False, api_key='mock-api-key')))
    params = ModelRequestParameters(
        function_tools=case['function_tools'],
        output_tools=case.get('output_tools', []),
        allow_text_output=True,
    )

    _, tool_config, _ = m._get_tool_config(params, case['settings'])  # pyright: ignore[reportPrivateUsage]

    assert tool_config is not None
    assert tool_config['function_calling_config']['mode'].name == case['expected_mode']  # pyright: ignore[reportTypedDictNotRequiredAccess,reportOptionalMemberAccess,reportOptionalSubscript,reportUnknownMemberType]


# =============================================================================
# `strict` resolution via `GoogleJsonSchemaTransformer`
# =============================================================================


def test_google_strict_resolution_via_transformer():
    """`GoogleJsonSchemaTransformer` treats every schema as `VALIDATED`-compatible (the mode needs no schema
    rewrites): `strict=None` resolves to `True` (VALIDATED-eligible), and an explicit `strict=False` is
    preserved as the per-tool opt-out."""
    m = GoogleModel('gemini-2.5-flash', provider=GoogleProvider(client=Client(vertexai=False, api_key='mock-api-key')))

    # `strict` left as `None` resolves to `True`: default-on, VALIDATED-eligible.
    params = m.customize_request_parameters(
        ModelRequestParameters(function_tools=[make_tool('a')], allow_text_output=True)
    )
    assert params.function_tools[0].strict is True

    # An explicit `strict=False` is preserved so the caller can opt the tool out of `VALIDATED`.
    params = m.customize_request_parameters(
        ModelRequestParameters(function_tools=[make_tool('a', strict=False)], allow_text_output=True)
    )
    assert params.function_tools[0].strict is False


# =============================================================================
# End-to-end wire contract
# =============================================================================


@pytest.mark.vcr
async def test_google_default_tools_use_validated_mode(
    allow_model_requests: None,
    google_model: GoogleModelFactory,
):
    """On a supported model, function tools default to `VALIDATED` mode with no `strict` flag set, and Gemini
    accepts that enum end-to-end.

    The mode-resolution cases above gate what `_get_tool_config` returns; the httpx event hook here gates
    what actually leaves the client, so drift anywhere between the two fails the assertion instead of
    hiding behind the recording.
    """
    sent_bodies: list[dict[str, Any]] = []

    async def capture_request(request: httpx2.Request) -> None:
        sent_bodies.append(json.loads(request.read()))

    http_client = httpx2.AsyncClient(event_hooks={'request': [capture_request]})
    agent = Agent(google_model('gemini-2.5-flash', http_client=http_client))

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'The weather in {city} is sunny and 24C.'

    @agent.tool_plain
    def get_time(city: str) -> str:
        return f'The time in {city} is 3pm.'

    result = await agent.run('What is the weather and the time in Paris? Use the tools.')
    assert result.output == snapshot('The weather in Paris is sunny and 24C. The time in Paris is 3pm.')

    assert sent_bodies[0]['toolConfig']['functionCallingConfig']['mode'] == 'VALIDATED'
    assert len(sent_bodies[0]['tools'][0]['functionDeclarations']) == 2


class Address(BaseModel):
    street: str
    unit: str | None = None


class HostileToStrict(BaseModel):
    """A schema carrying the shapes OpenAI/Anthropic strict mode reject or lossily rewrite.

    `minLength`/`maxLength`, a lookaround `pattern`, a free-form dict (`additionalProperties`), a
    `set` (`uniqueItems`), numeric bounds, a `tuple` (`prefixItems`), optional fields (absent from
    `required`), and a nested object with its own optional field. `GoogleJsonSchemaTransformer`
    keeps all of these, so they reach Gemini unchanged and exercise what `VALIDATED` tolerates.
    """

    # `python-re` so the lookaround `password` pattern below is definable — Pydantic's default Rust
    # engine rejects lookaround before the schema could ever reach Gemini.
    model_config = ConfigDict(regex_engine='python-re')

    name: str = Field(min_length=1, max_length=50)
    homepage: AnyUrl
    password: str = Field(pattern=r'(?=.*[0-9]).+')
    metadata: dict[str, str]
    tags: set[str]
    score: float = Field(ge=0, le=1)
    retries: int = 3
    nickname: str | None = None
    coordinate: tuple[float, float]
    address: Address


@pytest.mark.vcr
async def test_google_validated_accepts_strict_incompatible_schema(
    allow_model_requests: None,
    google_model: GoogleModelFactory,
):
    """Gemini `VALIDATED` accepts a schema that OpenAI/Anthropic strict mode would reject or rewrite.

    This is the safety proof behind defaulting supported models to `VALIDATED`: `HostileToStrict`
    carries every reject-trigger our OpenAI/Anthropic transformers flag. Gemini accepts it end-to-end
    under `VALIDATED` and returns a schema-adherent call — the `register` tool only runs if the args
    passed Pydantic validation — so the default doesn't break complex schemas.
    """
    sent_bodies: list[dict[str, Any]] = []

    async def capture_request(request: httpx2.Request) -> None:
        sent_bodies.append(json.loads(request.read()))

    http_client = httpx2.AsyncClient(event_hooks={'request': [capture_request]})
    agent = Agent(google_model('gemini-2.5-flash', http_client=http_client))

    @agent.tool_plain
    def register(profile: HostileToStrict) -> str:
        return f'Registered {profile.name} with {len(profile.tags)} tags.'

    result = await agent.run(
        'Register a user with name John Doe, homepage https://example.com, password Secret1, '
        'metadata city=NYC, tags premium and user, score 0.9, coordinate 1.0 and 2.0, and '
        'address 123 Main St. Use the register tool.'
    )
    assert result.output == snapshot('User John Doe registered successfully with 2 tags.')

    # Read off the hook, not the cassette: if the code stopped sending `VALIDATED` this fails, where an
    # assertion on the recorded body would keep passing against frozen YAML.
    assert sent_bodies[0]['toolConfig']['functionCallingConfig']['mode'] == 'VALIDATED'


class TreeNode(BaseModel):
    """A self-referencing node, so Pydantic emits a `$ref` cycle back into `$defs`."""

    label: str
    children: list[TreeNode] = []


class DeepD(BaseModel):
    value: str


class DeepC(BaseModel):
    d: DeepD


class DeepB(BaseModel):
    c: DeepC


class DeepA(BaseModel):
    b: DeepB


class RecursiveAndDeep(BaseModel):
    """The two schema shapes Google names as risky, neither of which `HostileToStrict` covers.

    Google's JSON schema reference demonstrates recursive `$ref` and warns that very large or deeply
    nested schemas may be rejected, so these are where `VALIDATED` could plausibly accept less than
    `AUTO` does.

    See <https://ai.google.dev/gemini-api/docs/structured-output#json-schema-support>.
    """

    tree: TreeNode
    deep: DeepA


@pytest.mark.parametrize(
    'strict,expected_mode,expected_output',
    [
        pytest.param(
            None,
            'VALIDATED',
            snapshot(
                'I have recorded a tree with root "root" and two children "a" and "b", and a deep value of "hello".'
            ),
            id='validated',
        ),
        pytest.param(
            False,
            'AUTO',
            snapshot('I have recorded a tree with root "root" and children "a" and "b", and a deep value of "hello".'),
            id='auto',
        ),
    ],
)
@pytest.mark.vcr
async def test_google_validated_accepts_what_auto_accepts(
    allow_model_requests: None,
    google_model: GoogleModelFactory,
    strict: bool | None,
    expected_mode: str,
    expected_output: str,
):
    """`VALIDATED` accepts the same schema `AUTO` does — the claim defaulting to `VALIDATED` rests on.

    Making `VALIDATED` the default is only backward compatible if it never narrows the schema surface
    the API accepts. Both cases declare an identical `RecursiveAndDeep` tool and differ only in mode,
    so a schema Gemini accepted under `AUTO` but rejected under `VALIDATED` would have failed while
    recording the `validated` case. Both cassettes exist, which is the evidence.

    The `strict=False` case doubles as the end-to-end proof of the documented opt-out: one non-strict
    tool puts the whole request back on `AUTO`, because Gemini's mode is request-wide.
    """
    sent_bodies: list[dict[str, Any]] = []

    async def capture_request(request: httpx2.Request) -> None:
        sent_bodies.append(json.loads(request.read()))

    http_client = httpx2.AsyncClient(event_hooks={'request': [capture_request]})
    agent = Agent(google_model('gemini-2.5-flash', http_client=http_client))

    @agent.tool_plain(strict=strict)
    def record(payload: RecursiveAndDeep) -> str:
        return f'Recorded tree "{payload.tree.label}" with {len(payload.tree.children)} children.'

    result = await agent.run(
        'Record a tree whose root is labelled "root" with two children labelled "a" and "b", '
        'and whose deep value is "hello". Use the record tool.'
    )
    assert result.output == expected_output

    assert sent_bodies[0]['toolConfig']['functionCallingConfig']['mode'] == expected_mode
    # Pin both shapes as actually reaching the wire — if the transformer started inlining or
    # flattening them, the mode assertion above would still pass and the test would prove nothing.
    schema = sent_bodies[0]['tools'][0]['functionDeclarations'][0]['parameters_json_schema']
    assert schema['$defs']['TreeNode']['properties']['children']['items'] == {'$ref': '#/$defs/TreeNode'}
    assert schema['$defs']['DeepA']['properties']['b'] == {'$ref': '#/$defs/DeepB'}
