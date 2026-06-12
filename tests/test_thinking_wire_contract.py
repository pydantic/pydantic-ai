"""Wire-contract tests for the unified `thinking` setting.

Each case asserts on the actual request wire body (`vcr.requests[0].body`), NOT on mock
kwargs or `_translate_thinking` return values. The cassette matcher isn't sensitive to the
request body, so asserting the body directly is what catches disable-signal regressions on
the wire (the methodology that surfaced the OpenRouter `enabled: True` miss).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from vcr.cassette import Cassette

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings, ThinkingLevel

from .cassette_utils import single_request_body
from .conftest import try_import

with try_import() as groq_imports:
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

with try_import() as cerebras_imports:
    from pydantic_ai.models.cerebras import CerebrasModel
    from pydantic_ai.providers.cerebras import CerebrasProvider

if TYPE_CHECKING:
    from pydantic_ai.models import Model

pytestmark = [pytest.mark.anyio, pytest.mark.vcr]


@dataclass(frozen=True)
class WireCase:
    id: str
    provider: str
    model_name: str
    thinking: ThinkingLevel
    extra_body: dict[str, object] | None = None
    """User-supplied `extra_body` passed via `ModelSettings`, to exercise the disable-signal merge path."""
    present: dict[str, object] = field(default_factory=dict[str, object])
    """Keys that must appear in the request body with exactly these values."""
    absent: tuple[str, ...] = ()
    """Keys that must NOT appear in the request body."""
    marks: tuple[pytest.MarkDecorator, ...] = ()


CASES = [
    WireCase(
        id='groq-qwen3-disable',
        provider='groq',
        model_name='qwen/qwen3-32b',
        thinking=False,
        present={'reasoning_effort': 'none'},
        absent=('reasoning_format',),
        marks=(pytest.mark.skipif(not groq_imports(), reason='groq not installed'),),
    ),
    WireCase(
        id='groq-qwen3-disable-merges-user-extra-body',
        provider='groq',
        model_name='qwen/qwen3-32b',
        thinking=False,
        extra_body={'service_tier': 'on_demand'},
        # The user's own `extra_body` must survive the merge that injects `reasoning_effort='none'`.
        present={'reasoning_effort': 'none', 'service_tier': 'on_demand'},
        absent=('reasoning_format',),
        marks=(pytest.mark.skipif(not groq_imports(), reason='groq not installed'),),
    ),
    WireCase(
        id='cerebras-zai-clear-thinking',
        provider='cerebras',
        model_name='zai-glm-4.7',
        thinking=False,
        # GLM disables via the standard `reasoning_effort='none'`, not the upstream-deprecated
        # `extra_body['disable_reasoning']` (https://inference-docs.cerebras.ai/resources/glm-47-migration).
        # `clear_thinking=false` is injected by default for the `zai` `<think>`-replay path so Cerebras
        # doesn't strip replayed reasoning — no user setting needed.
        present={'reasoning_effort': 'none', 'clear_thinking': False},
        absent=('disable_reasoning',),
        marks=(pytest.mark.skipif(not cerebras_imports(), reason='cerebras (openai) not installed'),),
    ),
    WireCase(
        id='cerebras-gpt-oss-always-on',
        provider='cerebras',
        model_name='gpt-oss-120b',
        thinking=False,
        # gpt-oss can't disable reasoning on Cerebras (`disable_reasoning=True` → 400, and the spec
        # rejects `reasoning_effort='none'` for gpt-oss too), so `thinking=False` must be silently
        # ignored: no disable signal of any kind on the wire, request accepted.
        absent=('disable_reasoning', 'reasoning_effort'),
        marks=(pytest.mark.skipif(not cerebras_imports(), reason='cerebras (openai) not installed'),),
    ),
]


def _build_model(case: WireCase, *, groq_api_key: str, cerebras_api_key: str) -> Model:
    if case.provider == 'groq':
        return GroqModel(case.model_name, provider=GroqProvider(api_key=groq_api_key))
    if case.provider == 'cerebras':
        return CerebrasModel(case.model_name, provider=CerebrasProvider(api_key=cerebras_api_key))
    raise ValueError(f'unknown provider {case.provider!r}')  # pragma: no cover


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id, marks=c.marks) for c in CASES])
async def test_thinking_disable_wire_contract(
    case: WireCase,
    allow_model_requests: None,
    groq_api_key: str,
    cerebras_api_key: str,
    vcr: Cassette,
):
    """`thinking=False` produces the correct wire behavior: a true disable signal where the model supports it, and its absence where reasoning is always on."""
    model = _build_model(case, groq_api_key=groq_api_key, cerebras_api_key=cerebras_api_key)
    settings = ModelSettings(thinking=case.thinking)
    if case.extra_body is not None:
        settings['extra_body'] = case.extra_body
    agent = Agent(model, model_settings=settings)
    await agent.run('What is 2+2? Reply with just the number.')

    body = single_request_body(vcr)
    for key, value in case.present.items():
        assert body.get(key) == value, f'expected {key}={value!r} on the wire, got {body.get(key)!r}'
    for key in case.absent:
        assert key not in body, f'expected {key!r} absent from the wire, got {body[key]!r}'
