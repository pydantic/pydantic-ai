"""Weekly live probes of the provider facts the Pydantic AI Gateway reaches (proposal R32).

A fact set here is a set of model names the library writes by hand: the models that accept a feature, or that reject
something the library therefore withholds. `FACT_SETS` lists the sets whose provider API the gateway serves, and gives
each member a probe: a request through the gateway, to a model the member matches, that passes while the provider still
behaves as the set says. A member the gateway can't reach is `Exempt`, with the reason.

`test_every_member_has_a_probe` runs offline in every CI run: a member added to a set fails until it gets a probe or an
exemption, and a probe whose member left its set fails as stale. `test_the_fact_still_holds` makes the paid requests,
and runs only under `--run-gateway-live`, each Monday in `.github/workflows/provider-facts-health.yml`.

A probe sends a minimal prompt and caps no output: `max_tokens` stays unset. It asserts what the provider answers, not
what the library sends: the recorded test each check names pins the request.
"""

from __future__ import annotations as _annotations

import ast
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pytest
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    CompactionPart,
    ModelRequest,
    ModelResponse,
    NativeOutput,
    SystemPromptPart,
    TextPart,
    ToolAvailabilityDeltaPart,
    UserPromptPart,
)
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters, infer_model
from pydantic_ai.native_tools import CodeExecutionTool
from pydantic_ai.native_tools._tool_search import ToolSearchTool
from pydantic_ai.profiles.google import GOOGLE_THINKING_LEVEL_SCALE, GOOGLE_THINKING_LEVELS
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers._bedrock_model_names import split_bedrock_model_id
from pydantic_ai.providers.gateway import gateway_provider
from pydantic_ai.tools import ToolDefinition

from ...conftest import try_import
from ...manifests import Exempt, assert_classified

with try_import() as imports_successful:
    from google.genai.types import ThinkingLevel

    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.models.openai._reasoning import supports_reasoning_effort_none
    from pydantic_ai.providers.bedrock import BedrockModelProfile

_REPO = Path(__file__).parents[3]
_PACKAGE = 'pydantic_ai_slim/pydantic_ai'

_PROMPT = f'Reply with OK. {uuid4().hex[:8]}'
"""The prompt every probe sends. The nonce keeps the gateway's response cache from answering a probe."""

GatewayRoute = Literal['anthropic', 'openai', 'bedrock', 'google-cloud']


class _Answer(BaseModel):
    answer: str


@dataclass(frozen=True)
class Probe:
    """A member's probe: the model it sends to, on its set's gateway route."""

    model: str
    """The model name the request names. The set's members are matched against it the way the library matches them."""

    @property
    def reason(self) -> str:
        return f'probed live on {self.model}'


@dataclass(frozen=True)
class FactSet:
    where: str
    """The set, as `path::symbol`, the symbol dotted for a set local to a function."""
    route: GatewayRoute
    check: Callable[[str], Awaitable[None]]
    """Sends a member's probe to the model it names, and fails unless the provider answers as the set says."""
    probes: Mapping[str, Probe | Exempt]
    """Each member's probe, or why it has none."""
    exact: bool = False
    """Whether the library matches a model to the members by name, rather than by prefix."""


def _gateway(route: GatewayRoute, model: str) -> Model:
    return infer_model(f'gateway/{route}:{model}')


async def _rejects(request: Awaitable[object], *, mentions: str | None = None, status: int = 400) -> None:
    """Fail unless the provider rejects `request` with `status`, and with a body that names `mentions`."""
    with pytest.raises(ModelHTTPError) as exc_info:
        await request
    assert exc_info.value.status_code == status, exc_info.value
    assert mentions is None or mentions in str(exc_info.value.body), exc_info.value.body


async def _request(model: Model, history: list[ModelRequest | ModelResponse], params: ModelRequestParameters) -> None:
    """Send `history` as one request through the model's public path, as the agent graph does."""
    await model.request(model.prepare_messages(history, params), None, params)


async def _anthropic_accepts_tool_addition(model_name: str) -> None:
    """Mirrors `tests/models/test_anthropic_mid_conversation_system.py::test_native_tool_availability_delta`."""
    tool = ToolDefinition(
        name='lookup_refund_policy',
        description='Look up the refund policy for an order.',
        parameters_json_schema={
            'type': 'object',
            'properties': {'order_id': {'type': 'string'}},
            'required': ['order_id'],
        },
        defer_loading=True,
        with_native='tool_search',
    )
    history: list[ModelRequest | ModelResponse] = [
        ModelRequest.user_text_prompt(_PROMPT),
        ModelResponse(parts=[TextPart('OK')]),
        ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['lookup_refund_policy'])]),
    ]
    params = ModelRequestParameters(function_tools=[tool], native_tools=[ToolSearchTool()])
    await _request(_gateway('anthropic', model_name), history, params)


async def _anthropic_accepts_request_scoped_system_prompt(model_name: str) -> None:
    """Mirrors `tests/models/test_request_scoped_system_prompts.py::test_anthropic_leading_request_scoped_prompt_is_not_hoisted`."""
    history: list[ModelRequest | ModelResponse] = [
        ModelRequest(
            parts=[SystemPromptPart('Answer in one word this turn.', scope='request'), UserPromptPart(_PROMPT)]
        )
    ]
    await _request(_gateway('anthropic', model_name), history, ModelRequestParameters())


async def _anthropic_accepts_effort_change(model_name: str) -> None:
    """Mirrors `tests/models/test_model_settings_delta.py::test_anthropic_effort_change_keeps_the_cache`."""
    agent = Agent(_gateway('anthropic', model_name))
    first = await agent.run(_PROMPT, model_settings={'thinking': 'low'})
    await agent.run(_PROMPT, message_history=first.all_messages(), model_settings={'thinking': 'medium'})


async def _anthropic_compacts(model_name: str) -> None:
    """Mirrors `tests/models/anthropic/test_compact_messages.py::test_compact_messages` and `::test_compact_messages_replay`."""
    model = _gateway('anthropic', model_name)
    history: list[ModelRequest | ModelResponse] = [
        ModelRequest.user_text_prompt(f'Record decision DB-1: use PostgreSQL. {_PROMPT}'),
        ModelResponse(parts=[TextPart('OK')], provider_name='anthropic'),
        ModelRequest.user_text_prompt(_PROMPT),
    ]
    compacted = await model.compact_messages(
        ModelRequestContext(
            model=model, messages=[*history], model_settings=None, model_request_parameters=ModelRequestParameters()
        )
    )
    assert [type(part) for part in compacted.parts] == [CompactionPart]
    await Agent(model).run(_PROMPT, message_history=[*history, compacted])


async def _anthropic_accepts_inline_system_prompt(model_name: str) -> None:
    """Mirrors `tests/models/test_anthropic_mid_conversation_system.py::test_mid_conversation_system_prompt`.

    It shows only that the model accepts the entry. Whether it honors it is a measured quality the set's docstring
    records, which no single request shows.
    """
    history: list[ModelRequest | ModelResponse] = [
        ModelRequest.user_text_prompt(_PROMPT),
        ModelResponse(parts=[TextPart('OK')]),
        ModelRequest(parts=[SystemPromptPart('Answer in one word.'), UserPromptPart(_PROMPT)]),
    ]
    await _request(_gateway('anthropic', model_name), history, ModelRequestParameters())


async def _anthropic_accepts_code_execution(model_name: str) -> None:
    """Mirrors `tests/models/test_anthropic.py::test_anthropic_code_execution_tool`, which sends `code_execution_20260120`."""
    await Agent(_gateway('anthropic', model_name), capabilities=[NativeTool(CodeExecutionTool())]).run(_PROMPT)


async def _accepts_native_output(route: GatewayRoute, model_name: str) -> None:
    result = await Agent(_gateway(route, model_name), output_type=NativeOutput(_Answer)).run(_PROMPT)
    assert isinstance(result.output, _Answer)


async def _anthropic_accepts_native_output(model_name: str) -> None:
    """Mirrors `tests/models/anthropic/test_output.py::test_no_tools_native_output_strict_true`."""
    await _accepts_native_output('anthropic', model_name)


async def _openai_chat_rejects_function_tools_while_reasoning(model_name: str) -> None:
    """Mirrors `tests/models/test_openai.py::test_chat_completions_function_tools_with_reasoning_provider_400_without_flag`.

    The probe turns off the guard the set enables, so the request goes out with a function tool and default reasoning.
    """
    model = OpenAIChatModel(
        model_name,
        provider=gateway_provider('openai'),
        profile=OpenAIModelProfile(openai_chat_function_tools_require_reasoning_off=False),
    )
    tool = ToolDefinition(name='get_temperature', parameters_json_schema={'type': 'object', 'properties': {}})
    params = ModelRequestParameters(function_tools=[tool])
    await _rejects(
        _request(model, [ModelRequest.user_text_prompt(_PROMPT)], params),
        mentions='Function tools with reasoning_effort are not supported',
    )


async def _openai_reasoning_off_as_profiled(model_name: str) -> None:
    """Sends `reasoning.effort='none'`, which the profile says the model accepts or rejects; checks it both ways.

    Mirrors `tests/models/test_openai_responses.py::test_openai_responses_gpt_5_6_reasoning_off_keeps_sampling_params`.
    """
    model = OpenAIResponsesModel(model_name, provider=gateway_provider('openai'))
    request = Agent(model).run(_PROMPT, model_settings=OpenAIResponsesModelSettings(openai_reasoning_effort='none'))
    if supports_reasoning_effort_none(model.profile):
        await request
    else:
        await _rejects(request, mentions='effort')


async def _openai_does_not_serve(model_name: str) -> None:
    """The model is gone: OpenAI answers `model_not_found`. Shaped like `tests/test_model_name_suggestions.py`'s recordings.

    The gateway answers a model it can't price with a 404 of its own, so the probe checks OpenAI's error code.
    """
    model = OpenAIChatModel(model_name, provider=gateway_provider('openai'))
    await _rejects(Agent(model).run(_PROMPT), status=404, mentions='model_not_found')


async def _bedrock_accepts_one_hour_cache(model_name: str) -> None:
    """Sends `cachePoint.ttl='1h'`, as `tests/models/test_bedrock.py::test_unified_cache_writes_then_reads_real_api` does.

    A prompt under the model's cache minimum is accepted but not cached, so the probe shows that Converse takes the
    TTL, not that it writes a one-hour entry.
    """
    await Agent(_gateway('bedrock', model_name), model_settings={'cache': '1h'}).run(_PROMPT)


async def _bedrock_rejects_native_output(model_name: str) -> None:
    """Turns on the structured output the set turns off, after a plain request shows the model answers at all."""
    model = BedrockConverseModel(
        model_name,
        provider=gateway_provider('bedrock'),
        profile=BedrockModelProfile(supports_json_schema_output=True, bedrock_supports_strict_tool_definition=True),
    )
    await Agent(model).run(_PROMPT)
    await _rejects(Agent(model, output_type=NativeOutput(_Answer)).run(_PROMPT))


async def _bedrock_answers(model_name: str) -> None:
    """Mirrors `tests/models/test_bedrock.py::test_gateway_bedrock_gpt_5_6_converse`, without its `max_tokens`."""
    await Agent(_gateway('bedrock', model_name)).run(_PROMPT)


async def _bedrock_accepts_native_output(model_name: str) -> None:
    """Mirrors `tests/models/bedrock/test_native_output.py::test_bedrock_native_output_supported_model`."""
    await _accepts_native_output('bedrock', model_name)


async def _bedrock_rejects_stop_sequences(model_name: str) -> None:
    """Sends the `stopSequences` the set withholds. The message is the one https://github.com/pydantic/pydantic-ai/pull/13762 recorded live."""
    model = BedrockConverseModel(
        model_name,
        provider=gateway_provider('bedrock'),
        profile=BedrockModelProfile(bedrock_supports_stop_sequences=True),
    )
    await _rejects(
        Agent(model, model_settings={'stop_sequences': ['5']}).run(_PROMPT),
        mentions="This model doesn't support the stopSequences field",
    )


async def _google_thinking_levels_as_profiled(model_name: str) -> None:
    """Sends each standard thinking level the profile leaves out, and the lowest it lists; checks each answer.

    Mirrors `tests/test_thinking_wire_contract.py::test_reasoning_wire_contract`'s Google cases, on Vertex.
    `XHIGH` is left out: only a model that lists it takes it, at the highest cost.
    """
    model = GoogleModel(model_name, provider=gateway_provider('google-cloud'))
    levels = model.profile.get('google_thinking_levels', GOOGLE_THINKING_LEVELS)
    agent = Agent(model)
    for level in GOOGLE_THINKING_LEVEL_SCALE:
        if level in GOOGLE_THINKING_LEVELS and level not in levels:
            settings = GoogleModelSettings(google_thinking_config={'thinking_level': ThinkingLevel(level)})
            await _rejects(agent.run(_PROMPT, model_settings=settings))
    lowest = next(level for level in GOOGLE_THINKING_LEVEL_SCALE if level in levels)
    await agent.run(
        _PROMPT, model_settings=GoogleModelSettings(google_thinking_config={'thinking_level': ThinkingLevel(lowest)})
    )


_MYTHOS = Exempt('Claude Mythos is not reachable with our credentials (`providers/anthropic.py` docstrings).')


def _retired(date: str) -> Exempt:
    return Exempt(f'Every model the member matches retired by {date} (`providers/_model_deprecations.py`).')


FACT_SETS: tuple[FactSet, ...] = (
    FactSet(
        where=f'{_PACKAGE}/providers/anthropic.py::_TOOL_AVAILABILITY_DELTA_MODEL_PREFIXES',
        route='anthropic',
        check=_anthropic_accepts_tool_addition,
        probes={
            'claude-fable-5': Probe('claude-fable-5-2'),
            'claude-mythos-5': _MYTHOS,
            'claude-opus-4-8': Probe('claude-opus-4-8'),
            'claude-opus-5': Probe('claude-opus-5-5'),
            'claude-sonnet-5-5': Probe('claude-sonnet-5-5'),
            'claude-haiku-5-5': Probe('claude-haiku-5-5'),
        },
    ),
    FactSet(
        where=f'{_PACKAGE}/providers/anthropic.py::_REQUEST_SCOPED_SYSTEM_PROMPT_MODEL_PREFIXES',
        route='anthropic',
        check=_anthropic_accepts_request_scoped_system_prompt,
        probes={
            'claude-fable-5': Probe('claude-fable-5-2'),
            'claude-mythos-5': _MYTHOS,
            'claude-opus-4-8': Probe('claude-opus-4-8'),
            'claude-opus-5': Probe('claude-opus-5-5'),
            'claude-sonnet-5-5': Probe('claude-sonnet-5-5'),
            'claude-haiku-5-5': Probe('claude-haiku-5-5'),
        },
    ),
    FactSet(
        where=f'{_PACKAGE}/providers/anthropic.py::_MID_CONVERSATION_EFFORT_MODEL_PREFIXES',
        route='anthropic',
        check=_anthropic_accepts_effort_change,
        probes={
            'claude-fable-5-1': Probe('claude-fable-5-1'),
            'claude-fable-5-2': Probe('claude-fable-5-2'),
            'claude-mythos-5-1': _MYTHOS,
            'claude-mythos-5-2': _MYTHOS,
            'claude-opus-5': Probe('claude-opus-5-5'),
            'claude-sonnet-5-5': Probe('claude-sonnet-5-5'),
            'claude-haiku-5-5': Probe('claude-haiku-5-5'),
        },
    ),
    FactSet(
        where=f'{_PACKAGE}/providers/anthropic.py::_ON_DEMAND_COMPACTION_MODEL_PREFIXES',
        route='anthropic',
        check=_anthropic_compacts,
        probes={
            'claude-fable-5': Probe('claude-fable-5-2'),
            'claude-mythos-5': _MYTHOS,
            'claude-mythos-preview': _MYTHOS,
            'claude-opus-4-6': Probe('claude-opus-4-6'),
            'claude-opus-4-7': Probe('claude-opus-4-7'),
            'claude-opus-4-8': Probe('claude-opus-4-8'),
            'claude-opus-5': Probe('claude-opus-5-5'),
            'claude-sonnet-4-6': Probe('claude-sonnet-4-6'),
            'claude-sonnet-5': Probe('claude-sonnet-5-5'),
            'claude-haiku-5-5': Probe('claude-haiku-5-5'),
        },
    ),
    FactSet(
        where=f'{_PACKAGE}/profiles/anthropic.py::_INLINE_SYSTEM_PROMPT_MODEL_PREFIXES',
        route='anthropic',
        check=_anthropic_accepts_inline_system_prompt,
        probes={
            'claude-fable-5': Probe('claude-fable-5-2'),
            'claude-mythos-5': _MYTHOS,
            'claude-opus-4-8': Probe('claude-opus-4-8'),
            'claude-opus-5': Probe('claude-opus-5-5'),
        },
    ),
    FactSet(
        where=f'{_PACKAGE}/profiles/anthropic.py::_ANTHROPIC_CODE_EXECUTION_20260120_MODEL_PREFIXES',
        route='anthropic',
        check=_anthropic_accepts_code_execution,
        probes={
            'claude-fable-5': Probe('claude-fable-5-2'),
            'claude-mythos-5': _MYTHOS,
            'claude-opus-4-5': Probe('claude-opus-4-5'),
            'claude-opus-4-6': Probe('claude-opus-4-6'),
            'claude-opus-4-7': Probe('claude-opus-4-7'),
            'claude-opus-4-8': Probe('claude-opus-4-8'),
            'claude-opus-5': Probe('claude-opus-5-5'),
            'claude-sonnet-4-5': Probe('claude-sonnet-4-5'),
            'claude-sonnet-4-6': Probe('claude-sonnet-4-6'),
            'claude-sonnet-5': Probe('claude-sonnet-5-5'),
            'claude-haiku-5-5': Probe('claude-haiku-5-5'),
        },
    ),
    FactSet(
        where=f'{_PACKAGE}/profiles/anthropic.py::anthropic_model_profile.models_that_support_json_schema_output',
        route='anthropic',
        check=_anthropic_accepts_native_output,
        probes={
            'claude-fable-5': Probe('claude-fable-5-2'),
            'claude-mythos-5': _MYTHOS,
            'claude-haiku-4-5': Probe('claude-haiku-4-5'),
            'claude-sonnet-4-5': Probe('claude-sonnet-4-5'),
            'claude-sonnet-4-6': Probe('claude-sonnet-4-6'),
            'claude-opus-4-1': _retired('2026-08-05'),
            'claude-opus-4-5': Probe('claude-opus-4-5'),
            'claude-opus-4-6': Probe('claude-opus-4-6'),
            'claude-opus-4-7': Probe('claude-opus-4-7'),
            'claude-opus-4-8': Probe('claude-opus-4-8'),
            'claude-opus-5': Probe('claude-opus-5-5'),
            'claude-sonnet-5': Probe('claude-sonnet-5-5'),
            'claude-haiku-5-5': Probe('claude-haiku-5-5'),
        },
    ),
    FactSet(
        where=f'{_PACKAGE}/profiles/openai.py::_GPT_6_MODEL_PREFIXES',
        route='openai',
        check=_openai_chat_rejects_function_tools_while_reasoning,
        probes={
            'gpt-6-astra': Probe('gpt-6-astra'),
            'gpt-6-sol': Probe('gpt-6-sol'),
            'gpt-6-luna': Probe('gpt-6-luna'),
            'gpt-6.1-sol': Probe('gpt-6.1-sol'),
            'gpt-6.1-terra': Probe('gpt-6.1-terra'),
            'gpt-6.2-sol': Probe('gpt-6.2-sol'),
            'gpt-6.2-astra': Probe('gpt-6.2-astra'),
        },
    ),
    FactSet(
        where=f'{_PACKAGE}/profiles/openai.py::_REASONING_SUPPORT_BY_PREFIX',
        route='openai',
        check=_openai_reasoning_off_as_profiled,
        probes={
            'gpt-6-astra': Probe('gpt-6-astra'),
            'gpt-6.1-sol': Probe('gpt-6.1-sol'),
            'gpt-6.2-sol': Probe('gpt-6.2-sol'),
            'gpt-6.2-astra': Probe('gpt-6.2-astra'),
            'gpt-6.1-terra': Probe('gpt-6.1-terra'),
            'gpt-6-sol': Probe('gpt-6-sol'),
            'gpt-6-luna': Probe('gpt-6-luna'),
            'gpt-5.6': Probe('gpt-5.6-luna'),
            'gpt-5.5-pro': Probe('gpt-5.5-pro'),
            'gpt-5.5': Probe('gpt-5.5'),
            'gpt-5.4-pro': Probe('gpt-5.4-pro'),
            'gpt-5.4': Probe('gpt-5.4-mini'),
            'gpt-5.3-chat': _retired('2026-08-10'),
            'gpt-5.3': Probe('gpt-5.3-codex'),
            'gpt-5.2-pro': Probe('gpt-5.2-pro'),
            'gpt-5.2-chat': _retired('2026-08-10'),
            'gpt-5.2': Probe('gpt-5.2'),
            'gpt-5.1-codex': _retired('2026-07-23'),
            'gpt-5.1-chat': _retired('2026-07-23'),
            'gpt-5.1': Probe('gpt-5.1'),
            'gpt-5-chat': _retired('2026-07-23'),
            'gpt-5-pro': Probe('gpt-5-pro'),
            'gpt-5-codex': _retired('2026-07-23'),
            'gpt-5': Probe('gpt-5-mini'),
            'o1': _retired('2026-10-23'),
            'o3': Probe('o3'),
            'o4': _retired('2026-10-23'),
        },
    ),
    FactSet(
        where=f'{_PACKAGE}/models/openai/_model_names.py::DEPRECATED_OPENAI_MODELS',
        route='openai',
        check=_openai_does_not_serve,
        exact=True,
        probes={
            name: Probe(name)
            for name in (
                'chatgpt-4o-latest',
                'codex-mini-latest',
                'gpt-3.5-turbo-0613',
                'gpt-3.5-turbo-16k-0613',
                'gpt-4-0125-preview',
                'gpt-4-1106-preview',
                'gpt-4-turbo-preview',
                'gpt-4-32k',
                'gpt-4-32k-0314',
                'gpt-4-32k-0613',
                'gpt-4-vision-preview',
                'gpt-4o-audio-preview-2024-10-01',
                'gpt-5.1-mini',
                'o1-mini',
                'o1-mini-2024-09-12',
                'o1-preview',
                'o1-preview-2024-09-12',
            )
        },
    ),
    FactSet(
        where=f'{_PACKAGE}/providers/_bedrock_model_names.py::_ONE_HOUR_CACHE_CLAUDE_MODEL_PREFIXES',
        route='bedrock',
        check=_bedrock_accepts_one_hour_cache,
        probes={
            'claude-fable-5': Probe('global.anthropic.claude-fable-5-2'),
            'claude-haiku-4-5': Probe('global.anthropic.claude-haiku-4-5-20251001-v1:0'),
            'claude-haiku-5-5': Probe('global.anthropic.claude-haiku-5-5'),
            'claude-mythos-5': _MYTHOS,
            'claude-opus-4-5': Probe('global.anthropic.claude-opus-4-5-20251101-v1:0'),
            'claude-opus-4-6': Probe('global.anthropic.claude-opus-4-6-v1'),
            'claude-opus-4-7': Probe('us.anthropic.claude-opus-4-7'),
            'claude-opus-4-8': Probe('global.anthropic.claude-opus-4-8'),
            'claude-opus-5': Probe('global.anthropic.claude-opus-5-5'),
            'claude-sonnet-4-5': Probe('us.anthropic.claude-sonnet-4-5-20250929-v1:0'),
            'claude-sonnet-4-6': Probe('global.anthropic.claude-sonnet-4-6'),
            'claude-sonnet-5': Probe('global.anthropic.claude-sonnet-5-5'),
        },
    ),
    FactSet(
        where=f'{_PACKAGE}/providers/_bedrock_model_names.py::_STRUCTURED_OUTPUT_UNSUPPORTED_CLAUDE_MODEL_PREFIXES',
        route='bedrock',
        check=_bedrock_rejects_native_output,
        probes={
            'claude-opus-4-1': Probe('us.anthropic.claude-opus-4-1-20250805-v1:0'),
            'claude-fable-5': Probe('global.anthropic.claude-fable-5-2'),
        },
    ),
    FactSet(
        where=f'{_PACKAGE}/providers/bedrock.py::_BEDROCK_CONVERSE_GPT_MODELS',
        route='bedrock',
        check=_bedrock_answers,
        exact=True,
        probes={
            name: Probe(f'global.openai.{name}')
            for name in (
                'gpt-5.6-sol',
                'gpt-5.6-terra',
                'gpt-5.6-luna',
                'gpt-6-sol',
                'gpt-6-luna',
                'gpt-6-astra',
                'gpt-6.1-sol',
                'gpt-6.1-terra',
            )
        },
    ),
    FactSet(
        where=f'{_PACKAGE}/providers/bedrock.py::bedrock_mistral_model_profile.models_that_support_structured_output',
        route='bedrock',
        check=_bedrock_accepts_native_output,
        probes={
            'magistral-small': Probe('mistral.magistral-small-2509'),
            'ministral-3': Probe('mistral.ministral-3-14b-instruct'),
            'mistral-large-3': Probe('mistral.mistral-large-3-675b-instruct'),
            'voxtral': Probe('mistral.voxtral-small-24b-2507'),
        },
    ),
    FactSet(
        where=f'{_PACKAGE}/providers/bedrock.py::bedrock_mistral_model_profile.models_that_reject_stop_sequences',
        route='bedrock',
        check=_bedrock_rejects_stop_sequences,
        probes={
            'devstral': Probe('mistral.devstral-2-123b'),
            'magistral': Probe('mistral.magistral-small-2509'),
            'ministral-3': Probe('mistral.ministral-3-14b-instruct'),
            'mistral-large-3': Probe('mistral.mistral-large-3-675b-instruct'),
            'voxtral': Probe('mistral.voxtral-small-24b-2507'),
        },
    ),
    FactSet(
        where=f'{_PACKAGE}/providers/bedrock.py::bedrock_qwen_model_profile.models_that_support_structured_output',
        route='bedrock',
        check=_bedrock_accepts_native_output,
        probes={'qwen3': Probe('qwen.qwen3-next-80b-a3b')},
    ),
    FactSet(
        where=f'{_PACKAGE}/providers/bedrock.py::bedrock_google_model_profile.models_that_support_structured_output',
        route='bedrock',
        check=_bedrock_accepts_native_output,
        probes={'gemma-3-12b-it': Probe('google.gemma-3-12b-it'), 'gemma-3-27b-it': Probe('google.gemma-3-27b-it')},
    ),
    FactSet(
        where=f'{_PACKAGE}/providers/bedrock.py::bedrock_minimax_model_profile.models_that_support_structured_output',
        route='bedrock',
        check=_bedrock_accepts_native_output,
        probes={'minimax-m2': Probe('minimax.minimax-m2.5')},
    ),
    FactSet(
        where=f'{_PACKAGE}/providers/bedrock.py::bedrock_nvidia_model_profile.models_that_support_structured_output',
        route='bedrock',
        check=_bedrock_accepts_native_output,
        probes={'nemotron-nano': Probe('nvidia.nemotron-nano-3-30b')},
    ),
    FactSet(
        where=f'{_PACKAGE}/profiles/google.py::_MODEL_THINKING_LEVELS',
        route='google-cloud',
        check=_google_thinking_levels_as_profiled,
        probes={
            'gemini-3.1-flash-lite-image': Probe('gemini-3.1-flash-lite-image'),
            'gemini-3.7-flash': Probe('gemini-3.7-flash'),
            'gemini-3.8-flash': Probe('gemini-3.8-flash'),
            'gemini-3.1-pro-preview': Probe('gemini-3.1-pro-preview'),
            'gemini-3-pro-preview': Probe('gemini-3-pro-preview'),
            'gemini-4-argon': Probe('gemini-4-argon'),
            'gemini-4-carbon': Probe('gemini-4-carbon'),
            'gemini-4-hydrogen': Probe('gemini-4-hydrogen'),
            'gemma-4': Probe('gemma-4-26b-a4b-it-maas'),
        },
    ),
)
"""The fact sets the gateway's routes reach: the Anthropic API, OpenAI, Bedrock Converse and Vertex AI."""


def _member_names(value: ast.expr) -> list[str]:
    """The members a set's literal value writes: its strings, a dict's keys, or each pair's first item."""
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'frozenset':
        return _member_names(value.args[0])
    elements = (
        value.keys if isinstance(value, ast.Dict) else value.elts if isinstance(value, ast.Tuple | ast.Set) else []
    )
    names: list[str] = []
    for element in elements:
        if isinstance(element, ast.Tuple):
            element = element.elts[0]
        assert isinstance(element, ast.Constant) and isinstance(element.value, str), ast.dump(value)
        names.append(element.value)
    return names


def _members(where: str) -> list[str]:
    """The members of the set at `where`, in source order, read from the module without importing it."""
    path, symbol = where.split('::')
    function, _, name = symbol.rpartition('.')
    body = ast.parse((_REPO / path).read_text(encoding='utf-8')).body
    if function:
        body = next(node.body for node in body if isinstance(node, ast.FunctionDef) and node.name == function)
    for node in body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return _member_names(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            assert node.value is not None, where
            return _member_names(node.value)
    raise AssertionError(f'{where} is not assigned in {path}')


def _matched_member(fact_set: FactSet, members: list[str], model: str) -> str | None:
    """The member the library matches `model` to: the first prefix it starts with, or its exact name."""
    name = split_bedrock_model_id(model)[1] if fact_set.route == 'bedrock' else model
    if fact_set.exact:
        return name if name in members else None
    return next((member for member in members if name.startswith(member)), None)


@pytest.mark.parametrize('fact_set', FACT_SETS, ids=[s.where.split('::')[1] for s in FACT_SETS])
def test_every_member_has_a_probe(fact_set: FactSet):
    """Each member of the set has a probe or an exemption, and each entry names a member."""
    members = _members(fact_set.where)
    assert_classified(members, fact_set.probes, complete_walk=True, table_name=fact_set.where)


@pytest.mark.parametrize('fact_set', FACT_SETS, ids=[s.where.split('::')[1] for s in FACT_SETS])
def test_each_probe_reaches_its_member(fact_set: FactSet):
    """A probe's model is one the library matches to the probe's own member, so the request carries its fact."""
    members = _members(fact_set.where)
    mismatched = {
        member: probe.model
        for member, probe in fact_set.probes.items()
        if isinstance(probe, Probe) and _matched_member(fact_set, members, probe.model) != member
    }
    assert mismatched == {}


_LIVE_PROBES = [
    pytest.param(fact_set, probe, id=f'{fact_set.where.split("::")[1]}[{member}]')
    for fact_set in FACT_SETS
    for member, probe in fact_set.probes.items()
    if isinstance(probe, Probe)
]


@pytest.mark.skipif(not imports_successful(), reason='provider packages not installed')
@pytest.mark.filterwarnings('ignore::pydantic_ai.exceptions.DeprecatedModelWarning')
@pytest.mark.parametrize(('fact_set', 'probe'), _LIVE_PROBES)
async def test_the_fact_still_holds(
    fact_set: FactSet, probe: Probe, allow_model_requests: None, gateway_live_api_key: str
):
    await fact_set.check(probe.model)
