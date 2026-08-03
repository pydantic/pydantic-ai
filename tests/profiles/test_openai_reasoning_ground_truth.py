"""Ground-truth probes for the OpenAI reasoning-capability profile flags.

`test_openai.py::test_reasoning_matrix` pins `openai_model_profile()` against a hand-written
table. That catches an accidental change to `_REASONING_SUPPORT_BY_PREFIX`, but it cannot catch
the map being *wrong*, because both sides of the assertion are ours. This file closes the loop by
asking the Responses API itself and cross-checking its answer against the profile: the API's
accept/reject is the assertion.

The probes deliberately use a raw `AsyncOpenAI` client instead of `Agent`/`OpenAIResponsesModel`.
The model consults the very flags under test to decide whether to send `reasoning.effort` and
whether to drop sampling parameters, so probing through it would assert the profile against
itself.

| probe                                | the API's answer means                                    |
| ------------------------------------ | --------------------------------------------------------- |
| `reasoning={'effort': 'none'}`       | accepted <=> `openai_supports_reasoning_effort_none`       |
| `temperature=0.5`, no `reasoning`    | rejected <=> `openai_reasoning_enabled_by_default`         |
| `reasoning={'mode': ...}`            | BOTH values accepted <=> `openai_responses_supports_reasoning_mode` |
| `reasoning={'context': 'all_turns'}` | accepted <=> `openai_responses_supports_reasoning_context` |

`mode` has to be probed with `'standard'` AND `'pro'`, and the flag only holds when both are
accepted, because on most models `reasoning.mode` is not a choice but an assertion of what the
model already is: every `-pro` model accepts `'pro'` and rejects `'standard'`, and every non-pro
reasoning model does the reverse. Accepting one value therefore says nothing about the flag —
which means the model can be *told* which mode to use — and only GPT-5.6 accepts both.

`mode` and `context` are probed only on the GPT-5.4/5.5/5.6 families, the only ones where either
flag is ever set; that covers both sides of `openai_responses_supports_reasoning_mode`. The
negative side of `openai_responses_supports_reasoning_context` is already recorded live against
`o3` by `test_openai_responses.py::test_openai_responses_reasoning_context_default_wire_contract`.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass

import pytest
from pydantic import BaseModel, TypeAdapter

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from openai import APIStatusError, AsyncOpenAI, Omit, omit
    from openai.types.shared_params import Reasoning

    from pydantic_ai.profiles.openai import openai_model_profile

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


@dataclass
class Probe:
    """What the Responses API said about one request shape."""

    accepted: bool
    error_code: str | None = None
    error_message: str | None = None


class _ErrorBody(BaseModel):
    code: str | None = None
    message: str | None = None


_ERROR_BODY = TypeAdapter(_ErrorBody)


async def _probe(
    client: AsyncOpenAI,
    model: str,
    *,
    reasoning: Reasoning | Omit = omit,
    temperature: float | Omit = omit,
) -> Probe:
    """Send one minimal request and report whether the API accepted the shape.

    `max_output_tokens` is bounded because the suite records ~100 of these against models as
    expensive as `gpt-5.5-pro`; the API validates the request shape before generating anything,
    so the accept/reject verdict is unaffected by the cap.
    """
    try:
        await client.responses.create(
            model=model,
            input='hi',
            max_output_tokens=16,
            store=False,
            reasoning=reasoning,
            temperature=temperature,
        )
    except APIStatusError as e:
        # The structured body, not `e.message`, which carries SDK formatting that differs
        # between a live call and VCR replay.
        error = _ERROR_BODY.validate_python(e.body)
        return Probe(accepted=False, error_code=error.code, error_message=error.message)
    return Probe(accepted=True)


@dataclass
class Case:
    """One model's recorded answers. `mode`/`context` are only probed where the flags apply."""

    model: str
    effort_none: Probe
    no_effort_temperature: Probe
    reasoning_mode_standard: Probe | None = None
    reasoning_mode_pro: Probe | None = None
    reasoning_context_all_turns: Probe | None = None


CASES = [
    # --- non-reasoning ---
    Case(
        model='gpt-4.1',
        effort_none=snapshot(
            Probe(
                accepted=False,
                error_code='unsupported_parameter',
                error_message="Unsupported parameter: 'reasoning.effort' is not supported with this model.",
            )
        ),
        no_effort_temperature=snapshot(Probe(accepted=True)),
    ),
    Case(
        model='gpt-4o-2024-08-06',
        effort_none=snapshot(
            Probe(
                accepted=False,
                error_code='unsupported_parameter',
                error_message="Unsupported parameter: 'reasoning.effort' is not supported with this model.",
            )
        ),
        no_effort_temperature=snapshot(Probe(accepted=True)),
    ),
    # --- the original GPT-5 family: always reasons, no off switch ---
    Case(
        model='gpt-5',
        effort_none=snapshot(
            Probe(
                accepted=False,
                error_code='unsupported_value',
                error_message="Unsupported value: 'none' is not supported with the 'gpt-5' model. Supported values are: 'minimal', 'low', 'medium', and 'high'.",
            )
        ),
        no_effort_temperature=snapshot(
            Probe(
                accepted=False, error_message="Unsupported parameter: 'temperature' is not supported with this model."
            )
        ),
    ),
    Case(
        model='gpt-5-mini',
        effort_none=snapshot(
            Probe(
                accepted=False,
                error_code='unsupported_value',
                error_message="Unsupported value: 'none' is not supported with the 'gpt-5-mini' model. Supported values are: 'minimal', 'low', 'medium', and 'high'.",
            )
        ),
        no_effort_temperature=snapshot(
            Probe(
                accepted=False, error_message="Unsupported parameter: 'temperature' is not supported with this model."
            )
        ),
    ),
    Case(
        model='gpt-5-nano',
        effort_none=snapshot(
            Probe(
                accepted=False,
                error_code='unsupported_value',
                error_message="Unsupported value: 'none' is not supported with the 'gpt-5-nano' model. Supported values are: 'minimal', 'low', 'medium', and 'high'.",
            )
        ),
        no_effort_temperature=snapshot(
            Probe(
                accepted=False, error_message="Unsupported parameter: 'temperature' is not supported with this model."
            )
        ),
    ),
    Case(
        model='gpt-5-pro',
        effort_none=snapshot(
            Probe(
                accepted=False,
                error_code='unsupported_value',
                error_message="Unsupported value: 'none' is not supported with the 'gpt-5-pro' model. Supported values are: 'high'.",
            )
        ),
        no_effort_temperature=snapshot(
            Probe(
                accepted=False, error_message="Unsupported parameter: 'temperature' is not supported with this model."
            )
        ),
    ),
    # --- the o-series: always reasons, no off switch ---
    Case(
        model='o1',
        effort_none=snapshot(
            Probe(
                accepted=False,
                error_code='unsupported_value',
                error_message="Unsupported value: 'none' is not supported with the 'o1' model. Supported values are: 'low', 'medium', and 'high'.",
            )
        ),
        no_effort_temperature=snapshot(
            Probe(
                accepted=False, error_message="Unsupported parameter: 'temperature' is not supported with this model."
            )
        ),
    ),
    Case(
        model='o1-pro',
        effort_none=snapshot(
            Probe(
                accepted=False,
                error_code='unsupported_value',
                error_message="Unsupported value: 'none' is not supported with the 'o1-pro' model. Supported values are: 'low', 'medium', and 'high'.",
            )
        ),
        no_effort_temperature=snapshot(
            Probe(
                accepted=False, error_message="Unsupported parameter: 'temperature' is not supported with this model."
            )
        ),
    ),
    Case(
        model='o3',
        effort_none=snapshot(
            Probe(
                accepted=False,
                error_code='unsupported_value',
                error_message="Unsupported value: 'none' is not supported with the 'o3' model. Supported values are: 'low', 'medium', and 'high'.",
            )
        ),
        no_effort_temperature=snapshot(
            Probe(
                accepted=False, error_message="Unsupported parameter: 'temperature' is not supported with this model."
            )
        ),
    ),
    Case(
        model='o3-mini',
        effort_none=snapshot(
            Probe(
                accepted=False,
                error_code='unsupported_value',
                error_message="Unsupported value: 'none' is not supported with the 'o3-mini' model. Supported values are: 'low', 'medium', and 'high'.",
            )
        ),
        no_effort_temperature=snapshot(
            Probe(
                accepted=False, error_message="Unsupported parameter: 'temperature' is not supported with this model."
            )
        ),
    ),
    Case(
        model='o4-mini',
        effort_none=snapshot(
            Probe(
                accepted=False,
                error_code='unsupported_value',
                error_message="Unsupported value: 'none' is not supported with the 'o4-mini' model. Supported values are: 'low', 'medium', and 'high'.",
            )
        ),
        no_effort_temperature=snapshot(
            Probe(
                accepted=False, error_message="Unsupported parameter: 'temperature' is not supported with this model."
            )
        ),
    ),
    # --- GPT-5.1..5.3 mainline: reasoning off by default, opt in via effort ---
    Case(
        model='gpt-5.1',
        effort_none=snapshot(Probe(accepted=True)),
        no_effort_temperature=snapshot(Probe(accepted=True)),
    ),
    Case(
        model='gpt-5.2',
        effort_none=snapshot(Probe(accepted=True)),
        no_effort_temperature=snapshot(Probe(accepted=True)),
    ),
    Case(
        model='gpt-5.3-codex',
        effort_none=snapshot(Probe(accepted=True)),
        no_effort_temperature=snapshot(Probe(accepted=True)),
    ),
    # --- GPT-5.1+ chat variants and -pro: always reason at a fixed effort ---
    Case(
        model='gpt-5.2-chat-latest',
        effort_none=snapshot(
            Probe(
                accepted=False,
                error_code='unsupported_value',
                error_message="Unsupported value: 'none' is not supported with the 'gpt-5.2-chat-latest' model. Supported values are: 'medium'.",
            )
        ),
        no_effort_temperature=snapshot(
            Probe(
                accepted=False, error_message="Unsupported parameter: 'temperature' is not supported with this model."
            )
        ),
    ),
    Case(
        model='gpt-5.3-chat-latest',
        effort_none=snapshot(
            Probe(
                accepted=False,
                error_code='unsupported_value',
                error_message="Unsupported value: 'none' is not supported with the 'gpt-5.3-chat-latest' model. Supported values are: 'medium'.",
            )
        ),
        no_effort_temperature=snapshot(
            Probe(
                accepted=False, error_message="Unsupported parameter: 'temperature' is not supported with this model."
            )
        ),
    ),
    Case(
        model='gpt-5.2-pro',
        effort_none=snapshot(
            Probe(
                accepted=False,
                error_code='unsupported_value',
                error_message="Unsupported value: 'none' is not supported with the 'gpt-5.2-pro' model. Supported values are: 'medium', 'high', and 'xhigh'.",
            )
        ),
        no_effort_temperature=snapshot(
            Probe(
                accepted=False, error_message="Unsupported parameter: 'temperature' is not supported with this model."
            )
        ),
    ),
    # --- GPT-5.4: opt-in reasoning, and the first family to accept `context='all_turns'` ---
    Case(
        model='gpt-5.4',
        effort_none=snapshot(Probe(accepted=True)),
        no_effort_temperature=snapshot(Probe(accepted=True)),
        reasoning_mode_standard=snapshot(Probe(accepted=True)),
        reasoning_mode_pro=snapshot(
            Probe(
                accepted=False,
                error_code='unsupported_value',
                error_message='`reasoning.mode` is not supported with this model.',
            )
        ),
        reasoning_context_all_turns=snapshot(Probe(accepted=True)),
    ),
    Case(
        model='gpt-5.4-mini',
        effort_none=snapshot(Probe(accepted=True)),
        no_effort_temperature=snapshot(Probe(accepted=True)),
        reasoning_mode_standard=snapshot(Probe(accepted=True)),
        reasoning_mode_pro=snapshot(
            Probe(
                accepted=False,
                error_code='unsupported_value',
                error_message='`reasoning.mode` is not supported with this model.',
            )
        ),
        reasoning_context_all_turns=snapshot(Probe(accepted=True)),
    ),
    Case(
        model='gpt-5.4-nano',
        effort_none=snapshot(Probe(accepted=True)),
        no_effort_temperature=snapshot(Probe(accepted=True)),
        reasoning_mode_standard=snapshot(Probe(accepted=True)),
        reasoning_mode_pro=snapshot(
            Probe(
                accepted=False,
                error_code='unsupported_value',
                error_message='`reasoning.mode` is not supported with this model.',
            )
        ),
        reasoning_context_all_turns=snapshot(Probe(accepted=True)),
    ),
    Case(
        model='gpt-5.4-pro',
        effort_none=snapshot(
            Probe(
                accepted=False,
                error_code='unsupported_value',
                error_message="Unsupported value: 'none' is not supported with the 'gpt-5.4-pro' model. Supported values are: 'medium', 'high', and 'xhigh'.",
            )
        ),
        no_effort_temperature=snapshot(
            Probe(
                accepted=False, error_message="Unsupported parameter: 'temperature' is not supported with this model."
            )
        ),
        reasoning_mode_standard=snapshot(
            Probe(
                accepted=False,
                error_code='unsupported_value',
                error_message='`reasoning.mode` is not supported with this model.',
            )
        ),
        reasoning_mode_pro=snapshot(Probe(accepted=True)),
        reasoning_context_all_turns=snapshot(Probe(accepted=True)),
    ),
    # --- GPT-5.5: reasons by default AND can be turned off ---
    Case(
        model='gpt-5.5',
        effort_none=snapshot(Probe(accepted=True)),
        no_effort_temperature=snapshot(
            Probe(
                accepted=False, error_message="Unsupported parameter: 'temperature' is not supported with this model."
            )
        ),
        reasoning_mode_standard=snapshot(Probe(accepted=True)),
        reasoning_mode_pro=snapshot(
            Probe(
                accepted=False,
                error_code='unsupported_value',
                error_message='`reasoning.mode` is not supported with this model.',
            )
        ),
        reasoning_context_all_turns=snapshot(Probe(accepted=True)),
    ),
    Case(
        model='gpt-5.5-pro',
        effort_none=snapshot(
            Probe(
                accepted=False,
                error_code='unsupported_value',
                error_message="Unsupported value: 'none' is not supported with the 'gpt-5.5-pro' model. Supported values are: 'medium', 'high', and 'xhigh'.",
            )
        ),
        no_effort_temperature=snapshot(
            Probe(
                accepted=False, error_message="Unsupported parameter: 'temperature' is not supported with this model."
            )
        ),
        reasoning_mode_standard=snapshot(
            Probe(
                accepted=False,
                error_code='unsupported_value',
                error_message='`reasoning.mode` is not supported with this model.',
            )
        ),
        reasoning_mode_pro=snapshot(Probe(accepted=True)),
        reasoning_context_all_turns=snapshot(Probe(accepted=True)),
    ),
    # --- GPT-5.6: the only family that accepts `reasoning.mode='pro'` ---
    Case(
        model='gpt-5.6-sol',
        effort_none=snapshot(Probe(accepted=True)),
        no_effort_temperature=snapshot(
            Probe(
                accepted=False, error_message="Unsupported parameter: 'temperature' is not supported with this model."
            )
        ),
        reasoning_mode_standard=snapshot(Probe(accepted=True)),
        reasoning_mode_pro=snapshot(Probe(accepted=True)),
        reasoning_context_all_turns=snapshot(Probe(accepted=True)),
    ),
    Case(
        model='gpt-5.6-terra',
        effort_none=snapshot(Probe(accepted=True)),
        no_effort_temperature=snapshot(
            Probe(
                accepted=False, error_message="Unsupported parameter: 'temperature' is not supported with this model."
            )
        ),
        reasoning_mode_standard=snapshot(Probe(accepted=True)),
        reasoning_mode_pro=snapshot(Probe(accepted=True)),
        reasoning_context_all_turns=snapshot(Probe(accepted=True)),
    ),
    Case(
        model='gpt-5.6-luna',
        effort_none=snapshot(Probe(accepted=True)),
        no_effort_temperature=snapshot(
            Probe(
                accepted=False, error_message="Unsupported parameter: 'temperature' is not supported with this model."
            )
        ),
        reasoning_mode_standard=snapshot(Probe(accepted=True)),
        reasoning_mode_pro=snapshot(Probe(accepted=True)),
        reasoning_context_all_turns=snapshot(Probe(accepted=True)),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.model) for c in CASES])
async def test_reasoning_flags_match_the_api(case: Case, openai_api_key: str):
    """Cross-check every reasoning flag the profile sets for this model against the live API."""
    async with AsyncOpenAI(api_key=openai_api_key) as client:
        effort_none = await _probe(client, case.model, reasoning={'effort': 'none'})
        no_effort_temperature = await _probe(client, case.model, temperature=0.5)

        assert effort_none == case.effort_none
        assert no_effort_temperature == case.no_effort_temperature

        profile = openai_model_profile(case.model)
        assert isinstance(profile, dict)
        assert profile.get('openai_supports_reasoning_effort_none', False) is effort_none.accepted
        # A model reasons by default exactly when it rejects sampling params with no effort set.
        assert profile.get('openai_reasoning_enabled_by_default', False) is not no_effort_temperature.accepted
        assert profile.get('openai_supports_reasoning', False) is (
            effort_none.accepted or not no_effort_temperature.accepted
        )

        if case.reasoning_mode_standard is not None and case.reasoning_mode_pro is not None:
            mode_standard = await _probe(client, case.model, reasoning={'mode': 'standard'})
            mode_pro = await _probe(client, case.model, reasoning={'mode': 'pro'})
            assert mode_standard == case.reasoning_mode_standard
            assert mode_pro == case.reasoning_mode_pro
            # Only accepting both values means the mode is selectable rather than fixed.
            assert profile.get('openai_responses_supports_reasoning_mode', False) is (
                mode_standard.accepted and mode_pro.accepted
            )

        if case.reasoning_context_all_turns is not None:
            context_all_turns = await _probe(client, case.model, reasoning={'context': 'all_turns'})
            assert context_all_turns == case.reasoning_context_all_turns
            assert profile.get('openai_responses_supports_reasoning_context', False) is context_all_turns.accepted


@dataclass
class MissingCase:
    """An id the profile resolves that the Responses API won't answer for."""

    model: str
    response: Probe


# The profile keeps resolving these — through `_REASONING_SUPPORT_BY_PREFIX` and the
# `test_openai.py` matrix — but nothing can prove those cells right. The retired ones are still
# listed by `/v1/models`, so a listing is not enough to detect them; only a request is.
MISSING_CASES = [
    # Retired: Chat Completions reports these as deprecated.
    MissingCase(
        model='gpt-5-chat-latest',
        response=snapshot(Probe(accepted=False, error_message='Model not found gpt-5-chat-latest')),
    ),
    MissingCase(
        model='gpt-5-codex', response=snapshot(Probe(accepted=False, error_message='Model not found gpt-5-codex'))
    ),
    MissingCase(
        model='gpt-5.1-chat-latest',
        response=snapshot(Probe(accepted=False, error_message='Model not found gpt-5.1-chat-latest')),
    ),
    MissingCase(
        model='gpt-5.1-codex', response=snapshot(Probe(accepted=False, error_message='Model not found gpt-5.1-codex'))
    ),
    MissingCase(
        model='gpt-5.1-codex-max',
        response=snapshot(Probe(accepted=False, error_message='Model not found gpt-5.1-codex-max')),
    ),
    MissingCase(
        model='gpt-5.1-codex-mini',
        response=snapshot(Probe(accepted=False, error_message='Model not found gpt-5.1-codex-mini')),
    ),
    MissingCase(
        model='gpt-5.2-codex', response=snapshot(Probe(accepted=False, error_message='Model not found gpt-5.2-codex'))
    ),
    MissingCase(
        model='o4-mini-deep-research',
        response=snapshot(Probe(accepted=False, error_message='Model not found o4-mini-deep-research')),
    ),
    # Live, but the Responses API doesn't serve it.
    MissingCase(
        model='gpt-5-search-api',
        response=snapshot(
            Probe(
                accepted=False,
                error_code='model_not_found',
                error_message="The requested model 'gpt-5-search-api' is not supported with the Responses API.",
            )
        ),
    ),
    # Never existed: prefix-coverage rows in the `test_openai.py` matrix, not real model ids.
    MissingCase(
        model='gpt-5-chat',
        response=snapshot(
            Probe(
                accepted=False,
                error_code='model_not_found',
                error_message="The requested model 'gpt-5-chat' does not exist.",
            )
        ),
    ),
    MissingCase(
        model='gpt-5-turbo',
        response=snapshot(
            Probe(
                accepted=False,
                error_code='model_not_found',
                error_message="The requested model 'gpt-5-turbo' does not exist.",
            )
        ),
    ),
    MissingCase(
        model='gpt-5.1-mini',
        response=snapshot(
            Probe(
                accepted=False,
                error_code='model_not_found',
                error_message="The requested model 'gpt-5.1-mini' does not exist.",
            )
        ),
    ),
    MissingCase(
        model='gpt-5.1-turbo',
        response=snapshot(
            Probe(
                accepted=False,
                error_code='model_not_found',
                error_message="The requested model 'gpt-5.1-turbo' does not exist.",
            )
        ),
    ),
    MissingCase(
        model='gpt-5.2-mini',
        response=snapshot(
            Probe(
                accepted=False,
                error_code='model_not_found',
                error_message="The requested model 'gpt-5.2-mini' does not exist.",
            )
        ),
    ),
    MissingCase(
        model='gpt-5.2-turbo',
        response=snapshot(
            Probe(
                accepted=False,
                error_code='model_not_found',
                error_message="The requested model 'gpt-5.2-turbo' does not exist.",
            )
        ),
    ),
    MissingCase(
        model='gpt-5.3-mini',
        response=snapshot(
            Probe(
                accepted=False,
                error_code='model_not_found',
                error_message="The requested model 'gpt-5.3-mini' does not exist.",
            )
        ),
    ),
    MissingCase(
        model='o1-mini',
        response=snapshot(
            Probe(
                accepted=False,
                error_code='model_not_found',
                error_message="The requested model 'o1-mini' does not exist.",
            )
        ),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.model) for c in MISSING_CASES])
async def test_ids_without_live_ground_truth(case: MissingCase, openai_api_key: str):
    """Record why these ids can't be cross-checked, so the gap is documented rather than assumed.

    A prefix entry outliving its model is harmless, so the profile is right to keep resolving them
    — but they stay out of the matrix above, because no request can confirm what it claims.
    """
    async with AsyncOpenAI(api_key=openai_api_key) as client:
        probe = await _probe(client, case.model)

    assert probe == case.response
    assert probe.accepted is False
