"""The request mapper of each model class, as the history properties call it, checked against the adapter registry.

`MAPPER_CASES` has an entry for every model class that `tests/model_adapter_cases.py::sdk_cells` lists with a `request`
entry point: a `MapperCase` that calls the class's own message mapper and projects the body for the checkers in
`tests/history_invariants.py`, or `Exempt(reason)` for a class whose mapping another case already exercises or whose
body is not a conversation. `test_every_requesting_model_class_has_a_mapper_case` in `tests/test_history_properties.py`
fails until a new model class gets one.

A case calls the model's private mapper instead of recording requests: the body properties are about the body itself,
over many histories per run, and a cassette holds one recorded body per test and isn't always matched on it. Each
history goes through what an agent run does before mapping, in its order (`prepare`).

This module imports no provider SDK at module level outside `try_import`, so it loads on every install lane.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from functools import cache

import pydantic_core
import pytest
from pydantic import JsonValue

from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    repair_messages,
)
from pydantic_ai.models import Model, ModelRequestParameters

from .conftest import try_import
from .history_invariants import (
    WireBlock,
    WireMessage,
    anthropic_wire,
    bedrock_wire,
    chat_completions_wire,
    google_wire,
    openai_responses_wire,
    xai_wire,
)
from .history_strategies import ServerRefApi
from .manifests import Exempt
from .model_adapter_cases import DECISION_PARAMETERS

with try_import() as anthropic_imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as bedrock_imports_successful:
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

with try_import() as cohere_imports_successful:
    from pydantic_ai.models.cohere import CohereModel
    from pydantic_ai.providers.cohere import CohereProvider

with try_import() as google_imports_successful:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as groq_imports_successful:
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

with try_import() as huggingface_imports_successful:
    from pydantic_ai.models.huggingface import HuggingFaceModel
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

with try_import() as mistral_imports_successful:
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider

with try_import() as openai_imports_successful:
    from pydantic_ai.models.openai import (
        OpenAIChatModel,
        OpenAIChatModelSettings,
        OpenAIResponsesModel,
        OpenAIResponsesModelSettings,
    )
    from pydantic_ai.models.openai_decisions import (
        OpenAIDecisionsModel,
        _Conversation,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.models.snowflake import SnowflakeModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider
    from pydantic_ai.providers.snowflake import SnowflakeProvider

with try_import() as xai_imports_successful:
    from google.protobuf.json_format import MessageToDict

    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.providers.xai import XaiProvider


@dataclass(frozen=True)
class Body:
    """A mapped request body, normalized to JSON: the system prompt, where the API takes it apart, and the messages."""

    system: JsonValue
    messages: JsonValue
    projected: list[ModelMessage] = field(default_factory=list[ModelMessage], compare=False)
    """What `Model.prepare_messages` made of the history, before the run's second repair."""


@dataclass(frozen=True)
class Prepared:
    """A history as an agent run hands it to the model's mapper, and the steps on the way."""

    parameters: ModelRequestParameters
    """The parameters the model's `request` maps the messages with."""
    projected: list[ModelMessage]
    """What `Model.prepare_messages` returned, before the run repairs it again."""
    messages: list[ModelMessage]
    """What the mapper is handed."""


def prepare(
    model: Model, history: list[ModelMessage], run_parameters: ModelRequestParameters | None = None
) -> Prepared:
    """What an agent run does to a history before the model maps it, in its order: the run repairs the history,
    prepares it with the run's parameters, and repairs it again, then the model's `request` prepares the
    parameters it maps the messages with."""
    run_parameters = run_parameters or ModelRequestParameters()
    projected = model.prepare_messages(repair_messages(history), run_parameters)
    _, parameters = model.prepare_request(None, run_parameters)
    return Prepared(parameters, projected, repair_messages(projected))


def _body(system: object, messages: object, prepared: Prepared) -> Body:
    return Body(
        pydantic_core.to_jsonable_python(system, bytes_mode='base64'),
        pydantic_core.to_jsonable_python(messages, bytes_mode='base64'),
        prepared.projected,
    )


@cache
def _openai_chat_model() -> OpenAIChatModel:
    return OpenAIChatModel('gpt-5.5', provider=OpenAIProvider(api_key='test-key'))


async def _openai_chat_body(history: list[ModelMessage]) -> Body:
    model = _openai_chat_model()
    prepared = prepare(model, history)
    parameters, messages = prepared.parameters, prepared.messages
    mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, model_request_parameters=parameters, model_settings=OpenAIChatModelSettings()
    )
    return _body(None, mapped, prepared)


@cache
def _openrouter_model() -> OpenRouterModel:
    return OpenRouterModel('openai/gpt-5.5', provider=OpenRouterProvider(api_key='test-key'))


async def _openrouter_body(history: list[ModelMessage]) -> Body:
    model = _openrouter_model()
    prepared = prepare(model, history)
    parameters, messages = prepared.parameters, prepared.messages
    return _body(None, await model._map_messages(messages, parameters), prepared)  # pyright: ignore[reportPrivateUsage]


@cache
def _snowflake_model() -> SnowflakeModel:
    return SnowflakeModel('claude-sonnet-4-5', provider=SnowflakeProvider(account='test', token='test-key'))


async def _snowflake_body(history: list[ModelMessage]) -> Body:
    model = _snowflake_model()
    prepared = prepare(model, history)
    parameters, messages = prepared.parameters, prepared.messages
    return _body(None, await model._map_messages(messages, parameters), prepared)  # pyright: ignore[reportPrivateUsage]


@cache
def _openai_responses_model() -> OpenAIResponsesModel:
    return OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(api_key='test-key'))


async def _openai_responses_body(history: list[ModelMessage]) -> Body:
    model = _openai_responses_model()
    prepared = prepare(model, history)
    parameters, messages = prepared.parameters, prepared.messages
    instructions, items = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, model_settings=OpenAIResponsesModelSettings(), model_request_parameters=parameters
    )
    return _body(instructions if isinstance(instructions, str) else None, items, prepared)


@cache
def _openai_decisions_model() -> OpenAIDecisionsModel:
    return OpenAIDecisionsModel('gpt-6-luna', provider=OpenAIProvider(api_key='test-key'))


async def _openai_decisions_body(history: list[ModelMessage]) -> Body:
    """The conversation the Decisions API judges, as the input items `decide` sends.

    A lone text prompt goes as plain text, which the projection reads as a user message.
    """
    model = _openai_decisions_model()
    prepared = prepare(model, history, DECISION_PARAMETERS)
    messages = prepared.messages
    # The images `request.images` holds, which the conversation's image placeholders count through.
    _, image_inputs = model._map_decision_input(messages, turn=False)  # pyright: ignore[reportPrivateUsage]
    images = [image for image in image_inputs if isinstance(image, BinaryContent)]
    items = _Conversation(images, model.profile).items(messages)
    return _body(None, [{'role': 'user', 'content': items}] if isinstance(items, str) else items, prepared)


@cache
def _anthropic_model() -> AnthropicModel:
    # A model with the native mid-conversation `system` role, which the rejections in `tests/test_history_properties.py`
    # were probed against on `claude-sonnet-5-5` before it moved to the sealed `<system>` rendering.
    return AnthropicModel('claude-opus-5-5', provider=AnthropicProvider(api_key='test-key'))


async def _anthropic_body(history: list[ModelMessage]) -> Body:
    model = _anthropic_model()
    prepared = prepare(model, history)
    parameters, messages = prepared.parameters, prepared.messages
    system, mapped = await model._map_message(  # pyright: ignore[reportPrivateUsage]
        messages, model_request_parameters=parameters, model_settings=AnthropicModelSettings()
    )
    return _body(system, mapped, prepared)


@cache
def _google_model() -> GoogleModel:
    return GoogleModel('gemini-3.5-flash', provider=GoogleProvider(api_key='test-key'))


async def _google_body(history: list[ModelMessage]) -> Body:
    model = _google_model()
    prepared = prepare(model, history)
    parameters, messages = prepared.parameters, prepared.messages
    system, contents = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, model_request_parameters=parameters
    )
    return _body(system, contents, prepared)


@cache
def _bedrock_model() -> BedrockConverseModel:
    return BedrockConverseModel(
        'us.anthropic.claude-sonnet-5', provider=BedrockProvider(api_key='test-key', region_name='us-east-1')
    )


async def _bedrock_body(history: list[ModelMessage]) -> Body:
    model = _bedrock_model()
    prepared = prepare(model, history)
    parameters, messages = prepared.parameters, prepared.messages
    system, mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, model_request_parameters=parameters, model_settings=None
    )
    return _body(system, mapped, prepared)


@cache
def _groq_model() -> GroqModel:
    return GroqModel('openai/gpt-oss-120b', provider=GroqProvider(api_key='test-key'))


async def _groq_body(history: list[ModelMessage]) -> Body:
    model = _groq_model()
    prepared = prepare(model, history)
    parameters, messages = prepared.parameters, prepared.messages
    return _body(None, await model._map_messages(messages, model_request_parameters=parameters), prepared)  # pyright: ignore[reportPrivateUsage]


@cache
def _mistral_model() -> MistralModel:
    return MistralModel('mistral-large-latest', provider=MistralProvider(api_key='test-key'))


async def _mistral_body(history: list[ModelMessage]) -> Body:
    model = _mistral_model()
    prepared = prepare(model, history)
    parameters, messages = prepared.parameters, prepared.messages
    return _body(None, await model._map_messages(messages, model_request_parameters=parameters), prepared)  # pyright: ignore[reportPrivateUsage]


@cache
def _cohere_model() -> CohereModel:
    return CohereModel('command-r7b-12-2024', provider=CohereProvider(api_key='test-key'))


async def _cohere_body(history: list[ModelMessage]) -> Body:
    model = _cohere_model()
    prepared = prepare(model, history)
    parameters, messages = prepared.parameters, prepared.messages
    return _body(None, model._map_messages(messages, model_request_parameters=parameters), prepared)  # pyright: ignore[reportPrivateUsage]


@cache
def _huggingface_model() -> HuggingFaceModel:
    return HuggingFaceModel('Qwen/Qwen3-235B-A22B', provider=HuggingFaceProvider(api_key='test-key'))


async def _huggingface_body(history: list[ModelMessage]) -> Body:
    model = _huggingface_model()
    prepared = prepare(model, history)
    parameters, messages = prepared.parameters, prepared.messages
    return _body(None, await model._map_messages(messages, model_request_parameters=parameters), prepared)  # pyright: ignore[reportPrivateUsage]


@cache
def _xai_model() -> XaiModel:
    return XaiModel('grok-4.3', provider=XaiProvider(api_key='test-key'))


async def _xai_body(history: list[ModelMessage]) -> Body:
    """The xAI chat messages, as the JSON form of the protobuf messages the SDK sends over gRPC."""
    model = _xai_model()
    prepared = prepare(model, history)
    parameters, messages = prepared.parameters, prepared.messages
    mapped = await model._map_messages(messages, model_request_parameters=parameters)  # pyright: ignore[reportPrivateUsage]
    return _body(None, [MessageToDict(message) for message in mapped], prepared)


@dataclass(frozen=True)
class MapperCase:
    id: str
    body: Callable[[list[ModelMessage]], Coroutine[object, object, Body]]
    model: Callable[[], Model]
    """The model `body` maps with."""
    wire: Callable[[JsonValue], list[WireMessage]]
    baseline: list[WireMessage]
    """What `wire` reads off the body `BASELINE` maps to: if a provider SDK renames a field, the checkers stop
    seeing calls or text, and this tells."""
    results_follow_their_call: bool = False
    """Whether a tool result must answer a call of the nearest earlier message that is not made of results: Anthropic
    and Bedrock document it; OpenAI Chat (gpt-5.5) rejected a `tool` message after an assistant turn without
    `tool_calls`, probed 2026-10-23 (UTC)."""
    rejects_empty_text: bool = False
    """Whether the API rejects a text block with no text, which the adapter's own handling of empty text shows."""
    takes_system_alone: bool = False
    """Whether the API takes a system prompt with no message as a request: OpenAI Responses (gpt-5.5) accepted
    instructions with an empty `input`, probed 2026-10-23 (UTC)."""
    takes_empty_user_turns: bool = False
    """Whether the API takes a user turn with no content, which the adapter sends for `UserPromptPart([])`: OpenAI
    Chat and OpenAI Responses (gpt-5.5) and the Hugging Face router (Qwen/Qwen3-235B-A22B through
    `provider_name='auto'`, with a tool declared) accepted one, probed 2026-10-23 (UTC)."""
    takes_empty_assistant_turns: bool = False
    """Whether the API takes an assistant turn with no content, which the adapter sends for a response with no
    parts: Groq (openai/gpt-oss-120b) and the Hugging Face router (Qwen/Qwen3-235B-A22B through `provider_name='auto'`,
    with a tool declared to clear #11177) accepted one, probed 2026-10-23 (UTC)."""
    refusal: str | None = None
    """The start of the `UserError` the model raises, by design, for a history it can't send: such a history has no
    body to check."""
    api: ServerRefApi | None = None
    """The API whose server references the body may carry, back to the endpoint that issued them. A reference another
    API issued has no meaning here."""
    marks: tuple[pytest.MarkDecorator, ...] = ()

    @property
    def reason(self) -> str:
        return f'mapped by the `{self.id}` case'

    @property
    def rejects_unpaired_tool_calls(self) -> bool:
        """Whether the model's profile says its API rejects an unpaired tool call, so `prepare_messages` pairs what it
        translates from another provider's native tool search."""
        return self.model().profile.get('rejects_unpaired_tool_calls', False)


def _needs(imports_successful: Callable[[], bool], package: str) -> tuple[pytest.MarkDecorator, ...]:
    return (pytest.mark.skipif(not imports_successful(), reason=f'{package} not installed'),)


NO_TEXT_TO_JUDGE = 'A request without user text is not supported by this model'
"""How a decision model refuses a history with nothing to judge, by design."""


BASELINE: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart('Look it up')]),
    ModelResponse(parts=[TextPart('Looking'), ToolCallPart('lookup', {'q': 'a'}, 'call_1')]),
    ModelRequest(parts=[ToolReturnPart('lookup', 'a', 'call_1')]),
    ModelResponse(parts=[TextPart('Found it')]),
]


def _baseline_turns(*, assistant: str, results: str, call_id: str = 'call_1') -> list[WireMessage]:
    """`BASELINE` as a body of role-tagged messages projects to, with the roles and call id the provider uses."""
    return [
        WireMessage('user', [WireBlock('text', 'Look it up')]),
        WireMessage(assistant, [WireBlock('text', 'Looking'), WireBlock('call', call_id)]),
        WireMessage(results, [WireBlock('result', call_id)]),
        WireMessage(assistant, [WireBlock('text', 'Found it')]),
    ]


_RESPONSES_BASELINE = [
    WireMessage('user', [WireBlock('text', 'Look it up')]),
    WireMessage('assistant', [WireBlock('text', 'Looking')]),
    WireMessage('function_call', [WireBlock('call', 'call_1')]),
    WireMessage('function_call_output', [WireBlock('result', 'call_1')]),
    WireMessage('assistant', [WireBlock('text', 'Found it')]),
]

_OPENAI = _needs(openai_imports_successful, 'openai')
_SAME_CHAT_MAPPER = (
    'subclasses `OpenAIChatModel` and maps messages through its mapper unchanged, which the `openai-chat` case runs; '
    'the request property still sends its histories through this class'
)
_SAME_RESPONSES_MAPPER = (
    'subclasses `OpenAIResponsesModel` and maps messages through its mapper unchanged, which the `openai-responses` '
    'case runs; the request property still sends its histories through this class'
)
_JUDGMENT_BODY = (
    'a decision model whose body asks questions about the conversation as one JSON document, not a conversation the '
    'body checkers can read; the request property still sends its histories through this class'
)

MAPPER_CASES: dict[str, MapperCase | Exempt] = {
    'OpenAIChatModel': MapperCase(
        'openai-chat',
        _openai_chat_body,
        _openai_chat_model,
        chat_completions_wire,
        _baseline_turns(assistant='assistant', results='tool'),
        results_follow_their_call=True,
        takes_empty_user_turns=True,
        api='openai-chat',
        marks=_OPENAI,
    ),
    'OpenRouterModel': MapperCase(
        'openrouter',
        _openrouter_body,
        _openrouter_model,
        chat_completions_wire,
        _baseline_turns(assistant='assistant', results='tool'),
        results_follow_their_call=True,
        takes_empty_user_turns=True,
        api='openai-chat',
        marks=_OPENAI,
    ),
    'SnowflakeModel': MapperCase(
        'snowflake',
        _snowflake_body,
        _snowflake_model,
        chat_completions_wire,
        _baseline_turns(assistant='assistant', results='tool'),
        results_follow_their_call=True,
        takes_empty_user_turns=True,
        api='openai-chat',
        marks=_OPENAI,
    ),
    'CerebrasModel': Exempt(_SAME_CHAT_MAPPER),
    'CrusoeModel': Exempt(_SAME_CHAT_MAPPER),
    'GitHubCopilotModel': Exempt(_SAME_CHAT_MAPPER),
    'OllamaModel': Exempt(_SAME_CHAT_MAPPER),
    'ZaiModel': Exempt(_SAME_CHAT_MAPPER),
    'BedrockMantleChatModel': Exempt(_SAME_CHAT_MAPPER),
    'OpenAIResponsesModel': MapperCase(
        'openai-responses',
        _openai_responses_body,
        _openai_responses_model,
        openai_responses_wire,
        _RESPONSES_BASELINE,
        takes_system_alone=True,
        takes_empty_user_turns=True,
        api='openai-responses',
        marks=_OPENAI,
    ),
    'BedrockMantleResponsesModel': Exempt(_SAME_RESPONSES_MAPPER),
    'OpenAICodexModel': Exempt(_SAME_RESPONSES_MAPPER),
    'OpenAIDecisionsModel': MapperCase(
        'openai-decisions',
        _openai_decisions_body,
        _openai_decisions_model,
        openai_responses_wire,
        _RESPONSES_BASELINE,
        # The Responses endpoint `OpenAIResponsesModel` sends to.
        takes_system_alone=True,
        takes_empty_user_turns=True,
        # It judges the latest user text, or the history before it, and refuses a request with neither.
        refusal=NO_TEXT_TO_JUDGE,
        api='openai-responses',
        marks=_OPENAI,
    ),
    'AnthropicModel': MapperCase(
        'anthropic',
        _anthropic_body,
        _anthropic_model,
        anthropic_wire,
        _baseline_turns(assistant='assistant', results='user'),
        results_follow_their_call=True,
        rejects_empty_text=True,
        api='anthropic',
        marks=_needs(anthropic_imports_successful, 'anthropic'),
    ),
    'GoogleModel': MapperCase(
        'google',
        _google_body,
        _google_model,
        google_wire,
        _baseline_turns(assistant='model', results='user'),
        marks=_needs(google_imports_successful, 'google-genai'),
    ),
    'GoogleDecisionsModel': Exempt(_JUDGMENT_BODY),
    'TypeSafeModel': Exempt(_JUDGMENT_BODY),
    'SystemOneModel': Exempt(_JUDGMENT_BODY),
    'BedrockConverseModel': MapperCase(
        'bedrock',
        _bedrock_body,
        _bedrock_model,
        bedrock_wire,
        _baseline_turns(assistant='assistant', results='user'),
        results_follow_their_call=True,
        rejects_empty_text=True,
        marks=_needs(bedrock_imports_successful, 'boto3'),
    ),
    'GroqModel': MapperCase(
        'groq',
        _groq_body,
        _groq_model,
        chat_completions_wire,
        _baseline_turns(assistant='assistant', results='tool'),
        takes_empty_assistant_turns=True,
        marks=_needs(groq_imports_successful, 'groq'),
    ),
    'MistralModel': MapperCase(
        'mistral',
        _mistral_body,
        _mistral_model,
        chat_completions_wire,
        # Mistral takes only 9-character alphanumeric ids, so its profile rewrites `call_1` to a stable hash.
        _baseline_turns(assistant='assistant', results='tool', call_id='RksXkbv4v'),
        marks=_needs(mistral_imports_successful, 'mistralai'),
    ),
    'CohereModel': MapperCase(
        'cohere',
        _cohere_body,
        _cohere_model,
        chat_completions_wire,
        _baseline_turns(assistant='assistant', results='tool'),
        marks=_needs(cohere_imports_successful, 'cohere'),
    ),
    'HuggingFaceModel': MapperCase(
        'huggingface',
        _huggingface_body,
        _huggingface_model,
        chat_completions_wire,
        _baseline_turns(assistant='assistant', results='tool'),
        takes_empty_user_turns=True,
        takes_empty_assistant_turns=True,
        marks=_needs(huggingface_imports_successful, 'huggingface-hub'),
    ),
    'XaiModel': MapperCase(
        'xai',
        _xai_body,
        _xai_model,
        xai_wire,
        _baseline_turns(assistant='assistant', results='tool'),
        marks=_needs(xai_imports_successful, 'xai-sdk'),
    ),
}
"""Each model class's mapper case, keyed by class name, or why it has none."""

BODY_CASES: list[MapperCase] = [case for case in MAPPER_CASES.values() if isinstance(case, MapperCase)]
"""The cases whose bodies the body properties check."""
