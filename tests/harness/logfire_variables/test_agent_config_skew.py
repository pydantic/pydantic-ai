"""What an `AgentConfig` written by a newer Logfire UI or contract costs a *run* of an older SDK.

How leniently a published value validates is the contract's own business, and `logfire.agent_control`
tests it there: a value it cannot act on costs one setting, one instruction entry, or one tool
override rather than the whole config, because an `AgentConfig` that fails validation is reverted
*whole* by Logfire's resolution fallback. What is tested here is the other end of that promise --
that an agent keeps running on the managed config around the piece that was dropped, and says where
the gap is when the drop is only visible at the point the config is applied.
"""

from __future__ import annotations

import pytest
from logfire.testing import CaptureLogfire

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.logfire import AgentControl

from ._helpers import Publish, published_value, variables_provider


async def test_agent_keeps_managed_config_around_a_dropped_setting(publish: Publish) -> None:
    seen: list[dict[str, object]] = []

    def capture(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(dict(info.model_settings or {}))
        return ModelResponse(parts=[TextPart('done')])

    publish('skew', {'settings': {'temperature': 0.4, 'thinking': 'ultra'}})
    with pytest.warns(UserWarning, match=r"sets 'thinking' to 'ultra'"):
        await Agent(FunctionModel(capture), capabilities=[AgentControl('skew', label='production')]).run('hello')
    assert seen == [{'temperature': 0.4}]


async def test_agent_reports_a_settings_key_it_has_no_field_for_when_applying(publish: Publish) -> None:
    # The same drop as above, one contract version earlier: the key is not wrong, this release simply
    # has no setting for it. Validation stays quiet (it also builds the baseline, where an unknown key
    # is the agent's own `extra_headers`); the run says so where the patch is applied, once per
    # process.
    seen: list[dict[str, object]] = []

    def capture(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(dict(info.model_settings or {}))
        return ModelResponse(parts=[TextPart('done')])

    publish('skew_key', {'settings': {'temperature': 0.4, 'service_tier': 'flex'}})
    with pytest.warns(UserWarning, match=r"sets 'service_tier', which this version of the Agent Control") as caught:
        agent = Agent(FunctionModel(capture), capabilities=[AgentControl('skew_key', label='production')])
        await agent.run('hello')
        await agent.run('again')
    assert len(caught) == 1
    assert seen == [{'temperature': 0.4}, {'temperature': 0.4}]


async def test_an_unusable_timeout_costs_only_that_key(publish: Publish) -> None:
    # A negative budget cancels the request before it is sent and an infinite one silently means "no
    # limit", so a timeout the contract cannot install is dropped rather than clamped -- and the
    # settings published beside it still apply.
    seen: list[dict[str, object]] = []

    def capture(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(dict(info.model_settings or {}))
        return ModelResponse(parts=[TextPart('done')])

    publish('bad_timeout', {'settings': {'timeout': -5, 'temperature': 0.4}})
    with pytest.warns(UserWarning, match='which is not a budget a request can be given'):
        await Agent(FunctionModel(capture), capabilities=[AgentControl('bad_timeout', label='production')]).run('hello')
    assert seen == [{'temperature': 0.4}]


async def test_an_empty_model_costs_only_the_model_section(capfire: CaptureLogfire) -> None:
    # `{'model': ''}` is what two clicks in the Logfire UI once produced, and it is the one value that
    # cannot be degraded field by field: Pydantic AI raises `UserError: Unknown model:` on every run
    # of an agent that has it. The contract refuses it as a *section*, so the agent keeps its code
    # model and the instructions published alongside it still apply -- which is the blast radius every
    # other malformed piece already had.
    published = {'model': '', 'instructions': 'MANAGED: be brief.'}
    agent = Agent(TestModel(), instructions='code', capabilities=[AgentControl('empty_model', label='production')])
    with variables_provider(capfire, published_value('agent__empty_model', published)):
        with pytest.warns(UserWarning, match="selects invalid model ''"):
            result = await agent.run('hello')
    instructions = [m.instructions for m in result.all_messages() if isinstance(m, ModelRequest)]
    assert instructions == ['code\n\nMANAGED: be brief.']


async def test_a_config_written_by_a_newer_ui_still_reaches_the_run(publish: Publish) -> None:
    # Keys this release has never heard of, at every level a newer UI could add one, and none of them
    # costs the sections beside it. A whole section it has none for is the one that is also *named*:
    # ignoring it is what keeps a future section readable by an older SDK, and saying so is what stops
    # the first person who publishes one from getting a silently degraded agent. A key inside an entry
    # is not named, because that is the parser being lenient about a value rather than a request
    # failing to apply one.
    seen: list[dict[str, object]] = []

    def capture(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(dict(info.model_settings or {}))
        return ModelResponse(parts=[TextPart('done')])

    publish(
        'newer_ui',
        {
            'future_section': {'anything': 1},
            'settings': {'temperature': 0.4},
            'instructions': [{'instructions': 'MANAGED: be brief.', 'future_field': 1}],
        },
    )
    with pytest.warns(UserWarning, match=r"publishes a 'future_section' section") as caught:
        result = await Agent(
            FunctionModel(capture), instructions='code', capabilities=[AgentControl('newer_ui', label='production')]
        ).run('hello')
    assert len(caught) == 1
    assert seen == [{'temperature': 0.4}]
    assert [m.instructions for m in result.all_messages() if isinstance(m, ModelRequest)] == [
        'code\n\nMANAGED: be brief.'
    ]
