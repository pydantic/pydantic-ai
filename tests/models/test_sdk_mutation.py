"""Every adapter returns a `ModelResponse` or raises `UnexpectedModelBehavior` for recorded responses mutated along its
provider SDK's response types.

`tests/models/sdk_mutation.py` samples the responses recorded at the endpoint each adapter reads, and mutates them as
the SDK's types admit: a null in an optional field, an unknown value of a `Literal`, an unknown member of a
discriminated union, a list of parts where a string usually is. `request` is served each mutated complete response,
and `request_stream` each mutated stream, over the transport `tests/model_adapter_cases.py` builds the adapter on.
Only a recording the adapter reads to a `ModelResponse` as recorded is mutated, so each failure is the mutation's. A
recorded response answers one request: a mutation that makes the adapter send another, to continue or retry, fails.

Not VCR tests: no provider sends these shapes on demand.
"""

from __future__ import annotations

import importlib
import itertools
import traceback
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import httpx2
import pytest
from pydantic import BaseModel

from pydantic_ai.direct import model_request, model_request_stream
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from ..conftest import try_import
from ..manifests import Exempt, assert_classified
from ..model_adapter_cases import CASES, MESSAGES, ModelAdapterCase, sdk_cells, shipped_model_classes
from ..model_adapter_transports import XaiTransport, instant_sdk_retries
from .sdk_mutation import (
    KNOWN,
    OPEN_FIELDS,
    SPEC_SHAPES,
    STREAM_SITE,
    Mutation,
    Recording,
    distinct_mutations,
    sample_recordings,
)

ENTRY_POINTS = ('request', 'request_stream')


@dataclass(frozen=True)
class MutationDriver:
    """Where an adapter's responses are recorded, and the SDK types it reads them as."""

    uri: str
    """A pattern that the URI of a request the adapter sends matches."""
    response_type: str
    """The SDK type of a complete response, as `<module>:<name>`."""
    event_type: str
    """The SDK type of each event of a stream, as `<module>:<name>`."""
    settings: ModelSettings | None = None
    """What keeps `request` from streaming under the hood, so it reads a complete response."""

    @property
    def reason(self) -> str:
        return f'mutates the responses recorded at `{self.uri}`'


_CHAT_SUBCLASS = Exempt(
    'subclasses `OpenAIChatModel`, whose driver mutates the Chat Completions types they share; the responses recorded '
    'from its own provider, and any response types of its own, are not mutated'
)
_RESPONSES_SUBCLASS = Exempt(
    'subclasses `OpenAIResponsesModel`, whose driver mutates the Responses types they share; the responses recorded '
    'from its own provider are not mutated'
)
_DECISION = Exempt('answers a decision request, outside the chat SDKs whose response types this check walks')

DRIVERS: dict[str, MutationDriver | Exempt] = {
    'OpenAIChatModel': MutationDriver(
        r'//api\.openai\.com/v1/chat/completions(?![\w/])',
        'openai.types.chat:ChatCompletion',
        'openai.types.chat:ChatCompletionChunk',
    ),
    'OpenAIResponsesModel': MutationDriver(
        r'//api\.openai\.com/v1/responses(?![\w/])',
        'openai.types.responses:Response',
        'openai.types.responses:ResponseStreamEvent',
    ),
    'AnthropicModel': MutationDriver(
        r'//api\.anthropic\.com/v1/messages(?![\w/])',
        'anthropic.types.beta:BetaMessage',
        'anthropic.types.beta:BetaRawMessageStreamEvent',
        # The default `max_tokens` is over what the SDK sends without streaming.
        settings={'max_tokens': 1024},
    ),
    'GoogleModel': MutationDriver(
        r'//generativelanguage\.googleapis\.com/v1beta/models/[^/:]+:(stream)?[gG]enerateContent',
        'google.genai.types:GenerateContentResponse',
        'google.genai.types:GenerateContentResponse',
    ),
    'MistralModel': MutationDriver(
        r'//api\.mistral\.ai/v1/chat/completions(?![\w/])',
        'mistralai.client.models:ChatCompletionResponse',
        'mistralai.client.models:CompletionChunk',
    ),
    'GroqModel': MutationDriver(
        r'//api\.groq\.com/openai/v1/chat/completions(?![\w/])',
        'groq.types.chat:ChatCompletion',
        'groq.types.chat:ChatCompletionChunk',
    ),
    'CohereModel': MutationDriver(
        r'//api\.cohere\.com/v2/chat(?![\w/])', 'cohere.v2.types:V2ChatResponse', 'cohere.v2.types:V2ChatStreamResponse'
    ),
    'CerebrasModel': _CHAT_SUBCLASS,
    'CrusoeModel': _CHAT_SUBCLASS,
    'GitHubCopilotModel': _CHAT_SUBCLASS,
    'OllamaModel': _CHAT_SUBCLASS,
    'OpenRouterModel': _CHAT_SUBCLASS,
    'SnowflakeModel': _CHAT_SUBCLASS,
    'ZaiModel': _CHAT_SUBCLASS,
    'BedrockMantleChatModel': _CHAT_SUBCLASS,
    'BedrockMantleResponsesModel': _RESPONSES_SUBCLASS,
    'OpenAICodexModel': _RESPONSES_SUBCLASS,
    'OpenAIDecisionsModel': _DECISION,
    'GoogleDecisionsModel': _DECISION,
    'TypeSafeModel': _DECISION,
    'SystemOneModel': _DECISION,
    'BedrockConverseModel': Exempt(
        'botocore parses a response from its service model into plain dicts: the `TypedDict`s it is typed with are '
        'stubs, not pydantic models'
    ),
    'HuggingFaceModel': Exempt('`huggingface_hub` types its responses as dataclasses, not pydantic models'),
    'XaiModel': Exempt('`xai_sdk` responses are protobuf messages, not pydantic models'),
}
"""How each model class whose `request` or `request_stream` calls its SDK gets its responses mutated, by class name."""


def test_every_chat_adapter_has_a_driver():
    _, skipped = shipped_model_classes()
    adapters = {cell.model_class.__name__ for cell in sdk_cells() if cell.entry_point in ENTRY_POINTS}
    assert_classified(adapters, DRIVERS, complete_walk=not skipped, table_name='DRIVERS')


def _resolve(path: str) -> object:
    module, _, name = path.partition(':')
    return getattr(importlib.import_module(module), name)


def test_tables_name_sdk_fields():
    """Each site in `SPEC_SHAPES`, `OPEN_FIELDS` and `KNOWN` names a field of its SDK, so a renamed one can't stay."""
    fields = {*SPEC_SHAPES, *OPEN_FIELDS, *(site for *_, site in KNOWN)}
    # An event inserted into a stream is at no field.
    for site in sorted(site for site in fields if not site.startswith(f'{STREAM_SITE}[')):
        module, model, field = site.rsplit('.', 2)
        with try_import() as installed:
            sdk_type = _resolve(f'{module}:{model}')
        if installed():
            assert isinstance(sdk_type, type) and issubclass(sdk_type, BaseModel), site
            assert field in sdk_type.model_fields, site


def _driven() -> list[tuple[str, MutationDriver, ModelAdapterCase]]:
    classes, _ = shipped_model_classes()
    return [
        (name, driver, case)
        for name in sorted(classes)
        if isinstance(driver := DRIVERS.get(name), MutationDriver)
        and isinstance(case := CASES.get(name), ModelAdapterCase)
    ]


@pytest.fixture(autouse=True)
def _instant_sdk_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    # An SDK retries the request `_OneAnswer` fails; without this it would back off on the real clock first.
    instant_sdk_retries(monkeypatch)


class _SentAgain(Exception):
    """The adapter sent another request, like a continuation or a retry, which one recorded response can't answer."""


@dataclass
class _OneAnswer:
    """Answers the first request with the recording, with `mutations` applied, and fails the rest."""

    recording: Recording
    mutations: Sequence[Mutation]
    requests: int = 0

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests += 1
        if self.requests > 1:
            raise httpx2.ConnectError('a recorded response answers one request', request=request)
        return self.recording.response(self.mutations)


async def _outcome(
    case: ModelAdapterCase,
    driver: MutationDriver,
    entry_point: str,
    recording: Recording,
    mutations: Sequence[Mutation] = (),
) -> ModelResponse | Exception:
    """What the adapter makes of the recording, with `mutations` applied."""
    transport = case.transport
    assert not isinstance(transport, XaiTransport)
    answer = _OneAnswer(recording, mutations)
    outcome: ModelResponse | Exception
    async with transport.model(answer.handler) as model:
        assert isinstance(model, Model)
        # A shape the SDK doesn't know may warn, and the suite raises warnings: what the adapter returns is the check.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                if entry_point == 'request':
                    outcome = await model_request(
                        model, MESSAGES, model_settings=driver.settings, model_request_parameters=case.parameters
                    )
                else:
                    async with model_request_stream(
                        model, MESSAGES, model_request_parameters=case.parameters
                    ) as stream:
                        async for _ in stream:
                            pass
                    outcome = stream.get()
            except Exception as error:
                # Any exception is an outcome to judge, not a failure of the harness.
                outcome = error
    return _SentAgain(f'sent {answer.requests} requests') if answer.requests > 1 else outcome


@dataclass
class _Sampled:
    """The sampled recordings, and each adapter's distinct mutations of them, found once per entry point."""

    recordings: dict[tuple[str, bool], list[Recording]]
    found: dict[tuple[str, str], list[tuple[Recording, tuple[Mutation, ...]]]] = field(
        default_factory=dict[tuple[str, str], list[tuple[Recording, tuple[Mutation, ...]]]]
    )

    async def mutations(
        self, name: str, driver: MutationDriver, case: ModelAdapterCase, entry_point: str
    ) -> list[tuple[Recording, tuple[Mutation, ...]]]:
        """The distinct mutations of the recordings the adapter reads to a `ModelResponse` as recorded."""
        if (key := (name, entry_point)) not in self.found:
            read = [
                recording
                for recording in self.recordings.get((driver.uri, entry_point == 'request_stream'), [])
                if isinstance(await _outcome(case, driver, entry_point, recording), ModelResponse)
            ]
            # Guards against a cassette format change leaving nothing to mutate.
            assert read, f'no recording at `{driver.uri}` reads to a response from `{entry_point}`'
            self.found[key] = distinct_mutations(read, _resolve(driver.response_type), _resolve(driver.event_type))
        return self.found[key]


@pytest.fixture(scope='module')
def sampled() -> _Sampled:
    return _Sampled(sample_recordings(driver.uri for _, driver, _ in _driven()))


def _problem(
    key: tuple[str, str], recording: Recording, places: Sequence[Mutation], outcome: ModelResponse | Exception
) -> str | None:
    """What's wrong with the outcome of the mutation at `places` for the adapter and entry point `key` names, if any."""
    mutation = places[0]
    if (known := KNOWN.get((*key, mutation.kind, mutation.site))) is not None:
        expected: tuple[type[ModelResponse | Exception], ...] = (known.raises,)
    elif mutation.kind == 'unknown-value' and mutation.site in OPEN_FIELDS:
        expected = (ModelResponse,)
    else:
        expected = (ModelResponse, UnexpectedModelBehavior)
    if isinstance(outcome, expected):
        return None
    return _described(recording, places, expected, outcome)  # pragma: no cover


def _described(  # pragma: no cover
    recording: Recording,
    places: Sequence[Mutation],
    expected: tuple[type[ModelResponse | Exception], ...],
    outcome: ModelResponse | Exception,
) -> str:
    mutation = places[0]
    where = f'{mutation.kind} at {mutation.site}, {recording.name} {list(mutation.path)} ({len(places)} places)'
    if isinstance(outcome, ModelResponse):
        got = 'a `ModelResponse`'
    else:
        frames = traceback.extract_tb(outcome.__traceback__)[-2:]
        got = f'`{type(outcome).__name__}: {outcome}` from ' + ' <- '.join(
            f'{frame.filename.rpartition("/")[2]}:{frame.name}' for frame in reversed(frames)
        )
    return f'{where}: expected {" or ".join(f"`{cls.__name__}`" for cls in expected)}, got {got}'


_STRIDE = 4
"""Splits each adapter's mutations across tests, so that no single test carries all of them against the budget."""


# One worker runs every test, so the cassettes are sampled once.
@pytest.mark.xdist_group(name='sdk_mutation')
@pytest.mark.parametrize('offset', range(_STRIDE))
@pytest.mark.parametrize('entry_point', ENTRY_POINTS)
@pytest.mark.parametrize(('name', 'driver', 'case'), [pytest.param(*driven, id=driven[0]) for driven in _driven()])
async def test_mutated_response_is_read_or_rejected(
    allow_model_requests: None,
    sampled: _Sampled,
    name: str,
    driver: MutationDriver,
    case: ModelAdapterCase,
    entry_point: str,
    offset: int,
):
    problems: list[str] = []
    for recording, places in itertools.islice(
        await sampled.mutations(name, driver, case, entry_point), offset, None, _STRIDE
    ):
        outcome = await _outcome(case, driver, entry_point, recording, places)
        if (problem := _problem((name, entry_point), recording, places, outcome)) is not None:
            problems.append(problem)  # pragma: no cover
    assert not problems, '\n'.join(
        [
            *problems,
            'Fix the adapter; if it crashes this way on `main` and an issue tracks the fix, add a `KNOWN` entry in '
            '`tests/models/sdk_mutation.py`',
        ]
    )


@pytest.mark.xdist_group(name='sdk_mutation')
async def test_known_crashes_name_generated_mutations(allow_model_requests: None, sampled: _Sampled):
    """Each `KNOWN` entry names a mutation the sampled recordings produce, so a stale one can't stay."""
    generated: set[tuple[str, str, str, str]] = set()
    for name, driver, case in _driven():
        for entry_point in ENTRY_POINTS:
            if any(key[:2] == (name, entry_point) for key in KNOWN):
                generated |= {
                    (name, entry_point, places[0].kind, places[0].site)
                    for _, places in await sampled.mutations(name, driver, case, entry_point)
                }
    driven = {name for name, _, _ in _driven()}
    stale = {key for key in KNOWN if key[0] in driven} - generated
    assert not stale, f'`KNOWN` entries {stale} name no mutation the recordings produce: remove them'
