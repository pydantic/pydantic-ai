"""Tests for `ModelProfile.prompt_cache_retention` and the `prompt_cache_outlook()` helper.

These are pure-data / pure-function tests (no provider requests), so they're unit tests rather
than VCR tests: the retention floors are documented model-family facts and the outlook is a
deterministic function of a message history and an injected `now`, neither of which a recorded
request would exercise.
"""

from __future__ import annotations as _annotations

from datetime import datetime, timedelta, timezone

import pytest

from pydantic_ai import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.profiles import ModelProfile, prompt_cache_outlook
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.openai import openai_model_profile

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _history(idle: timedelta) -> list[ModelMessage]:
    """A minimal request/response exchange whose last timestamp is `idle` before `NOW`."""
    at = NOW - idle
    return [
        ModelRequest(parts=[UserPromptPart(content='Hi')], timestamp=at),
        ModelResponse(parts=[TextPart(content='Hello!')], timestamp=at),
    ]


# ---- Documented profile retention floors -------------------------------------------------


def test_anthropic_profile_retention_floor():
    profile = anthropic_model_profile('claude-sonnet-4-5')
    assert profile is not None
    assert profile.get('prompt_cache_retention') == timedelta(minutes=5)


def test_openai_profile_retention_floor():
    profile = openai_model_profile('gpt-5.2')
    assert profile.get('prompt_cache_retention') == timedelta(minutes=5)


@pytest.mark.parametrize(
    'profile',
    [
        deepseek_model_profile('deepseek-reasoner'),
        google_model_profile('gemini-3-flash-preview'),
    ],
)
def test_undocumented_providers_leave_retention_unset(profile: ModelProfile | None):
    assert profile is not None
    assert profile.get('prompt_cache_retention') is None


# ---- Outlook: warm / cold boundaries -----------------------------------------------------


def test_outlook_warm_within_retention():
    retention = timedelta(minutes=5)
    assert prompt_cache_outlook(_history(timedelta(minutes=2)), retention=retention, now=NOW) == 'warm'


def test_outlook_cold_past_retention():
    retention = timedelta(minutes=5)
    assert prompt_cache_outlook(_history(timedelta(minutes=30)), retention=retention, now=NOW) == 'cold'


def test_outlook_boundary_exactly_at_retention_is_warm():
    # Idle == retention is still a (borderline) hit; only strictly-past is cold.
    retention = timedelta(minutes=5)
    assert prompt_cache_outlook(_history(retention), retention=retention, now=NOW) == 'warm'


def test_outlook_just_past_boundary_is_cold():
    retention = timedelta(minutes=5)
    assert prompt_cache_outlook(_history(retention + timedelta(seconds=1)), retention=retention, now=NOW) == 'cold'


# ---- Outlook: retention resolution -------------------------------------------------------


def test_outlook_uses_profile_retention():
    profile = anthropic_model_profile('claude-sonnet-4-5')
    assert prompt_cache_outlook(_history(timedelta(minutes=30)), profile=profile, now=NOW) == 'cold'
    assert prompt_cache_outlook(_history(timedelta(minutes=2)), profile=profile, now=NOW) == 'warm'


def test_explicit_retention_overrides_profile():
    profile = anthropic_model_profile('claude-sonnet-4-5')  # 5m floor
    # A 30-minute idle gap is cold under the 5m floor but warm under an explicit 1h override.
    assert prompt_cache_outlook(_history(timedelta(minutes=30)), profile=profile, now=NOW) == 'cold'
    assert (
        prompt_cache_outlook(_history(timedelta(minutes=30)), profile=profile, retention=timedelta(hours=1), now=NOW)
        == 'warm'
    )


# ---- Outlook: unknown cases --------------------------------------------------------------


def test_outlook_unknown_without_retention():
    profile = deepseek_model_profile('deepseek-reasoner')  # no documented retention
    assert prompt_cache_outlook(_history(timedelta(minutes=30)), profile=profile, now=NOW) == 'unknown'


def test_outlook_unknown_without_profile_or_retention():
    assert prompt_cache_outlook(_history(timedelta(minutes=30)), now=NOW) == 'unknown'


def test_outlook_unknown_for_empty_history():
    assert prompt_cache_outlook([], retention=timedelta(minutes=5), now=NOW) == 'unknown'


def test_outlook_unknown_when_history_has_no_timestamps():
    # A request that hasn't been sent (or was deserialized from an old format) has timestamp=None.
    history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='Hi')])]
    assert history[0].timestamp is None
    assert prompt_cache_outlook(history, retention=timedelta(minutes=5), now=NOW) == 'unknown'


def test_outlook_uses_latest_timestamp_scanning_from_end():
    # A trailing unsent request (timestamp=None) is skipped; the prior response's timestamp is used.
    at = NOW - timedelta(minutes=2)
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Hi')], timestamp=at),
        ModelResponse(parts=[TextPart(content='Hello!')], timestamp=at),
        ModelRequest(parts=[UserPromptPart(content='Again')]),  # not yet sent
    ]
    assert prompt_cache_outlook(history, retention=timedelta(minutes=5), now=NOW) == 'warm'


# ---- Outlook: timezone handling ----------------------------------------------------------


def test_outlook_handles_naive_timestamps_as_utc():
    # Historical/deserialized messages may carry naive datetimes; treat them as UTC.
    naive_at = datetime(2026, 1, 1, 11, 58, 0)  # 2 minutes before NOW, no tzinfo
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Hi')], timestamp=naive_at),
        ModelResponse(parts=[TextPart(content='Hello!')], timestamp=naive_at),
    ]
    assert prompt_cache_outlook(history, retention=timedelta(minutes=5), now=NOW) == 'warm'


def test_outlook_handles_naive_now_as_utc():
    # A caller-supplied naive `now` is also assumed UTC, matching the timestamp handling.
    naive_now = datetime(2026, 1, 1, 12, 0, 0)
    assert prompt_cache_outlook(_history(timedelta(minutes=2)), retention=timedelta(minutes=5), now=naive_now) == 'warm'
    assert (
        prompt_cache_outlook(_history(timedelta(minutes=30)), retention=timedelta(minutes=5), now=naive_now) == 'cold'
    )


def test_outlook_defaults_now_to_current_time():
    # Without an injected `now`, a very recent exchange is warm and a very old one is cold.
    assert prompt_cache_outlook(_recent_history(), retention=timedelta(minutes=5)) == 'warm'
    assert prompt_cache_outlook(_history(timedelta(days=1)), retention=timedelta(minutes=5)) == 'cold'


def _recent_history() -> list[ModelMessage]:
    now = datetime.now(timezone.utc)
    return [ModelResponse(parts=[TextPart(content='Hello!')], timestamp=now)]
