"""Tests for the BrowserUse capability and BrowserUseToolset."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Awaitable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Literal, TypeVar, overload

import anyio
import pytest

# `browser_use.Agent` is imported from its defining module: the test package
# `tests/browser_use` shadows the top-level `browser_use` name in pyright's
# tests execution environment, while submodule imports resolve correctly.
from browser_use.agent.service import Agent as BrowserUseAgent
from browser_use.agent.service import Tools  # pyright: ignore[reportPrivateImportUsage]
from browser_use.browser import BrowserProfile, BrowserSession
from browser_use.llm.messages import BaseMessage
from browser_use.llm.views import ChatInvokeCompletion
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.browser_use import (
    BrowserAgent,
    BrowserAgentHistory,
    BrowserAgentSettings,
    BrowserTask,
    BrowserUse,
    BrowserUseToolset,
    PydanticAIChatModel,
    default_browser_agent,
)

T = TypeVar('T', bound=BaseModel)


@pytest.fixture
def kill_calls(monkeypatch: pytest.MonkeyPatch) -> list[BrowserSession]:
    """Record `BrowserSession.kill` calls instead of running the real teardown."""
    calls: list[BrowserSession] = []

    async def record_kill(self: BrowserSession) -> None:
        calls.append(self)

    monkeypatch.setattr(BrowserSession, 'kill', record_kill)
    return calls


class _FakeChatModel:
    """A `BaseChatModel` double, passed through opaquely and never invoked in these tests.

    Structural conformance rather than inheritance: the protocol declares
    `provider`/`name` as properties and `ainvoke` with overloads, so a subclass
    would have to restate all three to satisfy the override checks.
    """

    _verified_api_keys: bool = True
    model: str = 'fake-model'
    provider: str = 'fake'
    name: str = 'fake-model'
    model_name: str = 'fake-model'

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: type, handler: object) -> object:
        raise NotImplementedError('the fake chat model is never validated')  # pragma: no cover

    @overload
    async def ainvoke(  # pragma: no cover - overload is enforced by static type checking
        self, messages: list[BaseMessage], output_format: None = None, **kwargs: object
    ) -> ChatInvokeCompletion[str]: ...

    @overload
    async def ainvoke(  # pragma: no cover - overload is enforced by static type checking
        self, messages: list[BaseMessage], output_format: type[T], **kwargs: object
    ) -> ChatInvokeCompletion[T]: ...

    async def ainvoke(
        self, messages: list[BaseMessage], output_format: type[T] | None = None, **kwargs: object
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        raise NotImplementedError('the fake chat model is never invoked')  # pragma: no cover


class _Facts(BaseModel):
    name: str
    price_usd: int


def _validation_error() -> ValidationError:
    """A real pydantic `ValidationError` for `_Facts`."""
    try:
        _Facts.model_validate({})
    except ValidationError as error:
        return error
    raise AssertionError('unreachable')  # pragma: no cover


@dataclass
class _FakeHistory:
    """A `BrowserAgentHistory` double with canned outcomes."""

    result: str | None = None
    step_errors: list[str | None] = field(default_factory=list[str | None])
    success: bool | None = None
    structured: BaseModel | None = None
    structured_error: ValidationError | None = None

    def final_result(self) -> None | str:
        return self.result

    def errors(self) -> list[str | None]:
        return self.step_errors

    def is_successful(self) -> bool | None:
        return self.success

    @property
    def structured_output(self) -> BaseModel | None:
        if self.structured_error is not None:
            raise self.structured_error
        return self.structured


class _FakeBrowserAgent:
    """A `BrowserAgent` double: records `run` calls, returns a canned history."""

    def __init__(self, history: _FakeHistory, error: Exception | None = None) -> None:
        self.history = history
        self.error = error
        self.run_calls: list[int] = []

    async def run(self, max_steps: int = 500) -> _FakeHistory:
        self.run_calls.append(max_steps)
        if self.error is not None:
            raise self.error
        return self.history


class _FakeFactory:
    """A `BrowserAgentFactory` double: records the `BrowserTask` requests."""

    def __init__(self, agent: _FakeBrowserAgent) -> None:
        self.agent = agent
        self.requests: list[BrowserTask] = []

    def __call__(self, request: BrowserTask) -> _FakeBrowserAgent:
        self.requests.append(request)
        return self.agent


def _success_factory(result: str = 'done') -> _FakeFactory:
    """A factory whose agent finishes successfully with `result`."""
    return _FakeFactory(_FakeBrowserAgent(_FakeHistory(result=result, success=True)))


class TestBrowserUseToolset:
    async def test_returns_final_result(self, kill_calls: list[BrowserSession]) -> None:
        factory = _success_factory('The Pro plan costs $20.')
        toolset = BrowserUse[None](browser_agent=factory).get_toolset()
        assert isinstance(toolset, BrowserUseToolset)

        result = await toolset.browse_web('find the price of the Pro plan')

        assert result == 'The Pro plan costs $20.'
        [request] = factory.requests
        assert request.task == 'find the price of the Pro plan'
        assert request.llm is None
        assert request.use_vision is True
        assert request.output_schema is None
        assert request.sensitive_data is None
        assert request.extend_system_message is None
        assert request.settings == BrowserAgentSettings()
        assert factory.agent.run_calls == [50]

    async def test_configuration_forwarded_to_session_and_agent(self, kill_calls: list[BrowserSession]) -> None:
        llm = _FakeChatModel()
        secrets: dict[str, str | dict[str, str]] = {'x_password': 'hunter2'}
        factory = _success_factory()
        capability = BrowserUse[None](
            llm=llm,
            allowed_domains=['example.com', 'example.org'],
            headless=False,
            max_steps=7,
            use_vision='auto',
            sensitive_data=secrets,
            extend_system_message='Never submit forms.',
            cdp_url='http://localhost:9222',
            browser_agent=factory,
        )

        await capability.get_toolset().browse_web('task')

        [request] = factory.requests
        assert request.llm is llm
        assert request.use_vision == 'auto'
        assert request.sensitive_data == secrets
        assert request.extend_system_message == 'Never submit forms.'
        session = request.browser_session
        assert session.browser_profile.headless is False
        assert session.browser_profile.allowed_domains == ['example.com', 'example.org']
        assert session.cdp_url == 'http://localhost:9222'
        assert factory.agent.run_calls == [7]

    async def test_defaults_to_headless_without_profile(self, kill_calls: list[BrowserSession]) -> None:
        factory = _success_factory()

        await BrowserUse[None](browser_agent=factory).get_toolset().browse_web('task')

        session_profile = factory.requests[0].browser_session.browser_profile
        assert session_profile.headless is True
        assert session_profile.block_ip_addresses is True
        assert session_profile.prohibited_domains == ['localhost', 'localhost.*', '*.localhost']
        # In 'call' scope the sub-agent may tear the session down itself; the
        # tool kills it in `finally` regardless.
        assert not session_profile.keep_alive

    async def test_browser_profile_forwarded_and_kept(self, kill_calls: list[BrowserSession]) -> None:
        profile = BrowserProfile(headless=False, allowed_domains=['docs.example.com'], user_agent='harness-test')
        factory = _success_factory()

        await BrowserUse[None](browser_profile=profile, browser_agent=factory).get_toolset().browse_web('task')

        session_profile = factory.requests[0].browser_session.browser_profile
        assert session_profile.headless is False
        assert session_profile.allowed_domains == ['docs.example.com']
        assert session_profile.user_agent == 'harness-test'
        assert session_profile.block_ip_addresses is True
        assert session_profile.prohibited_domains == ['localhost', 'localhost.*', '*.localhost']

    async def test_sensitive_data_disables_cross_origin_iframes_without_a_profile(
        self, kill_calls: list[BrowserSession]
    ) -> None:
        factory = _success_factory()

        await (
            BrowserUse[None](
                browser_agent=factory,
                sensitive_data={'https://example.com': {'x_password': 'hunter2'}},
            )
            .get_toolset()
            .browse_web('task')
        )

        assert factory.requests[0].browser_session.browser_profile.cross_origin_iframes is False

    async def test_sensitive_data_disables_cross_origin_iframes_from_a_profile(
        self, kill_calls: list[BrowserSession]
    ) -> None:
        factory = _success_factory()

        await (
            BrowserUse[None](
                browser_agent=factory,
                browser_profile=BrowserProfile(cross_origin_iframes=True),
                sensitive_data={'https://example.com': {'x_password': 'hunter2'}},
            )
            .get_toolset()
            .browse_web('task')
        )

        assert factory.requests[0].browser_session.browser_profile.cross_origin_iframes is False

    async def test_capability_fields_override_browser_profile(self, kill_calls: list[BrowserSession]) -> None:
        profile = BrowserProfile(headless=False, allowed_domains=['docs.example.com'])
        factory = _success_factory()
        capability = BrowserUse[None](
            browser_profile=profile,
            headless=True,
            allowed_domains=['example.com'],
            browser_agent=factory,
        )

        await capability.get_toolset().browse_web('task')

        session_profile = factory.requests[0].browser_session.browser_profile
        assert session_profile.headless is True
        assert session_profile.allowed_domains == ['example.com']

    @pytest.mark.parametrize(
        ('profile_allowed_domains', 'expected_allowed_domains'),
        [
            (
                [
                    'trusted.example',
                    'localhost',
                    'localhost.*',
                    '*.localhost',
                    '*',
                    'https://localhost/*',
                    'https://localhost/private',
                    'http://localhost:*/*',
                    'https://*.localhost:*/*',
                ],
                ['trusted.example'],
            ),
            ({'trusted.example', 'localhost'}, {'trusted.example'}),
        ],
    )
    async def test_localhost_is_removed_from_a_profile_allowlist(
        self,
        kill_calls: list[BrowserSession],
        profile_allowed_domains: list[str] | set[str],
        expected_allowed_domains: list[str] | set[str],
    ) -> None:
        factory = _success_factory()

        await (
            BrowserUse[None](
                browser_profile=BrowserProfile(allowed_domains=profile_allowed_domains),
                browser_agent=factory,
            )
            .get_toolset()
            .browse_web('task')
        )

        session_profile = factory.requests[0].browser_session.browser_profile
        assert session_profile.allowed_domains == expected_allowed_domains
        assert session_profile.prohibited_domains == ['localhost', 'localhost.*', '*.localhost']

    @pytest.mark.parametrize(
        'capability',
        [
            BrowserUse[None](allowed_domains=['localhost']),
            BrowserUse[None](browser_profile=BrowserProfile(allowed_domains=['localhost'])),
            BrowserUse[None](allowed_domains=['*://localhost:*/*']),
        ],
    )
    async def test_localhost_only_allowlist_requires_explicit_opt_in(
        self,
        capability: BrowserUse[None],
    ) -> None:
        with pytest.raises(ValueError, match='block_ip_addresses=False'):
            await capability.get_toolset().browse_web('task')

    async def test_bare_scheme_domain_has_a_path_boundary(self, kill_calls: list[BrowserSession]) -> None:
        factory = _success_factory()

        await (
            BrowserUse[None](
                allowed_domains=['https://trusted.example'],
                browser_agent=factory,
            )
            .get_toolset()
            .browse_web('task')
        )

        assert factory.requests[0].browser_session.browser_profile.allowed_domains == ['https://trusted.example/*']

    async def test_profile_bare_scheme_domain_has_a_path_boundary(self, kill_calls: list[BrowserSession]) -> None:
        factory = _success_factory()

        await (
            BrowserUse[None](
                browser_profile=BrowserProfile(allowed_domains=['https://trusted.example']),
                browser_agent=factory,
            )
            .get_toolset()
            .browse_web('task')
        )

        assert factory.requests[0].browser_session.browser_profile.allowed_domains == ['https://trusted.example/*']

    async def test_private_network_blocking_can_be_disabled(self, kill_calls: list[BrowserSession]) -> None:
        factory = _success_factory()

        await (
            BrowserUse[None](
                block_ip_addresses=False,
                browser_agent=factory,
            )
            .get_toolset()
            .browse_web('task')
        )

        session_profile = factory.requests[0].browser_session.browser_profile
        assert session_profile.block_ip_addresses is False
        assert session_profile.prohibited_domains is None

    async def test_private_network_setting_overrides_browser_profile(self, kill_calls: list[BrowserSession]) -> None:
        factory = _success_factory()

        await (
            BrowserUse[None](
                browser_profile=BrowserProfile(block_ip_addresses=True),
                block_ip_addresses=False,
                browser_agent=factory,
            )
            .get_toolset()
            .browse_web('task')
        )

        assert factory.requests[0].browser_session.browser_profile.block_ip_addresses is False

    async def test_session_killed_after_success(self, kill_calls: list[BrowserSession]) -> None:
        factory = _success_factory()

        await BrowserUse[None](browser_agent=factory).get_toolset().browse_web('task')

        [killed] = kill_calls
        assert killed is factory.requests[0].browser_session

    async def test_session_killed_when_run_raises(self, kill_calls: list[BrowserSession]) -> None:
        factory = _FakeFactory(_FakeBrowserAgent(_FakeHistory(), error=RuntimeError('browser crashed')))

        with pytest.raises(RuntimeError, match='browser crashed'):
            await BrowserUse[None](browser_agent=factory).get_toolset().browse_web('task')

        [killed] = kill_calls
        assert killed is factory.requests[0].browser_session

    @pytest.mark.parametrize('session_scope', ['call', 'agent'])
    async def test_session_killed_when_the_call_is_cancelled(
        self, monkeypatch: pytest.MonkeyPatch, session_scope: str
    ) -> None:
        """A cancelled `browse_web` still closes the browser.

        The real `kill` suspends several times over CDP, so the fake suspends
        too: without the shield around teardown, the first of those checkpoints
        raises inside the cancelled scope and the browser is never closed.
        """
        attempts: list[BrowserSession] = []

        async def suspending_kill(self: BrowserSession) -> None:
            await anyio.sleep(0)
            attempts.append(self)
            if len(attempts) == 1:
                raise TimeoutError('event bus stop timed out')

        monkeypatch.setattr(BrowserSession, 'kill', suspending_kill)

        running = anyio.Event()

        class _HangingAgent:
            async def run(self, max_steps: int = 500) -> _FakeHistory:
                running.set()
                await anyio.sleep_forever()
                raise AssertionError('unreachable')  # pragma: no cover

        requests: list[BrowserTask] = []

        def factory(request: BrowserTask) -> _HangingAgent:
            requests.append(request)
            return _HangingAgent()

        toolset = BrowserUse[None](
            session_scope='agent' if session_scope == 'agent' else 'call',
            browser_agent=factory,
        ).get_toolset()

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(toolset.browse_web, 'task')
            await running.wait()
            task_group.cancel_scope.cancel()

        await toolset.aclose()

        assert attempts == [requests[0].browser_session, requests[0].browser_session]

    async def test_no_result_reports_step_errors(self, kill_calls: list[BrowserSession]) -> None:
        history = _FakeHistory(step_errors=[None, 'timeout on step 2', 'element not found'])
        factory = _FakeFactory(_FakeBrowserAgent(history))

        result = await BrowserUse[None](browser_agent=factory).get_toolset().browse_web('task')

        assert result == (
            'The browser agent stopped without producing a result (timeout on step 2; element not found).'
        )

    async def test_no_result_without_errors(self, kill_calls: list[BrowserSession]) -> None:
        factory = _FakeFactory(_FakeBrowserAgent(_FakeHistory()))

        result = await BrowserUse[None](browser_agent=factory).get_toolset().browse_web('task')

        assert result == 'The browser agent stopped without producing a result (no further details).'

    async def test_unsuccessful_result_is_flagged(self, kill_calls: list[BrowserSession]) -> None:
        history = _FakeHistory(result='I could not log in.', success=False)
        factory = _FakeFactory(_FakeBrowserAgent(history))

        result = await BrowserUse[None](browser_agent=factory).get_toolset().browse_web('task')

        assert result == ('The browser agent could not fully complete the task. Its final result: I could not log in.')

    async def test_unsuccessful_structured_result_is_flagged(self, kill_calls: list[BrowserSession]) -> None:
        """A schema parses regardless of the agent's verdict, so the verdict has to survive it.

        browser-use's `structured_output` validates the final result whether or not the
        sub-agent called `done` with `success=False`, so returning the JSON alone would
        present a run it gave up on as a clean answer.
        """
        facts = _Facts(name='Pro', price_usd=0)
        history = _FakeHistory(result='{"name": "Pro", "price_usd": 0}', success=False, structured=facts)
        factory = _FakeFactory(_FakeBrowserAgent(history))
        capability = BrowserUse[None](output_schema=_Facts, browser_agent=factory)

        result = await capability.get_toolset().browse_web('task')

        assert result == (
            'The browser agent could not fully complete the task. Its final result: {"name":"Pro","price_usd":0}'
        )

    async def test_structured_output_returned_as_json(self, kill_calls: list[BrowserSession]) -> None:
        facts = _Facts(name='Pro', price_usd=20)
        history = _FakeHistory(result='{"name": "Pro", "price_usd": 20}', success=True, structured=facts)
        factory = _FakeFactory(_FakeBrowserAgent(history))
        capability = BrowserUse[None](output_schema=_Facts, browser_agent=factory)

        result = await capability.get_toolset().browse_web('task')

        assert json.loads(result) == {'name': 'Pro', 'price_usd': 20}
        assert factory.requests[0].output_schema is _Facts

    async def test_structured_output_mismatch_raises_model_retry(self, kill_calls: list[BrowserSession]) -> None:
        history = _FakeHistory(result='not json', success=True, structured_error=_validation_error())
        factory = _FakeFactory(_FakeBrowserAgent(history))
        capability = BrowserUse[None](output_schema=_Facts, browser_agent=factory)

        with pytest.raises(ModelRetry, match='did not match the configured output schema'):
            await capability.get_toolset().browse_web('task')

        [killed] = kill_calls
        assert killed is factory.requests[0].browser_session

    async def test_structured_output_missing_falls_back_to_text(self, kill_calls: list[BrowserSession]) -> None:
        history = _FakeHistory(result='prose result', success=True)
        factory = _FakeFactory(_FakeBrowserAgent(history))
        capability = BrowserUse[None](output_schema=_Facts, browser_agent=factory)

        result = await capability.get_toolset().browse_web('task')

        assert result == 'prose result'

    async def test_default_factory_builds_real_browser_use_agent(
        self, monkeypatch: pytest.MonkeyPatch, kill_calls: list[BrowserSession]
    ) -> None:
        monkeypatch.setenv('ANONYMIZED_TELEMETRY', 'false')
        seen: dict[str, object] = {}

        async def record_run(self: object, max_steps: int = 500, **kwargs: object) -> _FakeHistory:
            seen['agent'] = self
            seen['max_steps'] = max_steps
            return _FakeHistory(result='browsed', success=True)

        monkeypatch.setattr(BrowserUseAgent, 'run', record_run)
        llm = _FakeChatModel()
        secrets: dict[str, str | dict[str, str]] = {'x_password': 'hunter2'}
        toolset = BrowserUse[None](
            llm=llm,
            allowed_domains=['example.com'],
            max_steps=3,
            sensitive_data=secrets,
            extend_system_message='Never buy anything.',
            agent_settings=BrowserAgentSettings(use_judge=False, flash_mode=True, judge_llm='test'),
        ).get_toolset()

        result = await toolset.browse_web('check example.com')

        assert result == 'browsed'
        agent = seen['agent']
        assert isinstance(agent, BrowserUseAgent)
        assert agent.task == 'check example.com'
        assert agent.llm is llm
        assert agent.sensitive_data == secrets
        assert agent.settings.use_judge is False
        assert agent.settings.flash_mode is True
        assert isinstance(agent.judge_llm, PydanticAIChatModel)
        assert seen['max_steps'] == 3
        assert len(kill_calls) == 1

    async def test_pydantic_ai_model_string_is_wrapped(self, kill_calls: list[BrowserSession]) -> None:
        factory = _success_factory()

        await BrowserUse[None](llm='test', browser_agent=factory).get_toolset().browse_web('task')

        assert isinstance(factory.requests[0].llm, PydanticAIChatModel)

    async def test_settings_chat_models_arrive_resolved(self, kill_calls: list[BrowserSession]) -> None:
        factory = _success_factory()
        capability = BrowserUse[None](
            agent_settings=BrowserAgentSettings(judge_llm='test'),
            browser_agent=factory,
        )

        await capability.get_toolset().browse_web('task')

        assert isinstance(factory.requests[0].settings.judge_llm, PydanticAIChatModel)


class TestBrowserAgentSettings:
    def test_mirrors_browser_use_defaults(self) -> None:
        """Every setting is a real `browser_use.Agent` option at browser-use's own default.

        The promise of `BrowserAgentSettings` is that an empty instance changes
        nothing, so a browser-use upgrade that renames an option or moves a
        default has to fail here rather than quietly alter how the sub-agent
        behaves.
        """
        # browser-use's own constructor is partially untyped (`**kwargs`, a bare
        # `dict` schema), so pyright cannot see it as fully known.
        parameters = inspect.signature(
            BrowserUseAgent.__init__  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        ).parameters
        for setting in fields(BrowserAgentSettings):
            parameter = parameters.get(setting.name)
            assert parameter is not None, f'{setting.name} is not a browser_use.Agent option'
            assert parameter.default == setting.default, f'{setting.name} default drifted from browser-use'

    def test_every_setting_reaches_the_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each setting arrives with its own value, not just its own key.

        `default_browser_agent` forwards forty-odd settings by hand, which is the archetypal
        copy-paste-swap surface. Asserting only that the key arrived leaves any swap between
        two type-compatible fields passing: `loop_detection_window=settings.step_timeout`
        reads fine and is wrong. Distinct values per field turn a swap into a failure.
        """
        seen: dict[str, object] = {}

        def record_init(self: object, **kwargs: object) -> None:
            seen.update(kwargs)

        settings = _distinctly_valued_settings()
        settings.tools = Tools[None]()
        monkeypatch.setattr(BrowserUseAgent, '__init__', record_init)
        default_browser_agent(
            BrowserTask(
                task='task',
                llm=None,
                browser_session=BrowserSession(),
                use_vision=True,
                output_schema=None,
                sensitive_data=None,
                extend_system_message=None,
                settings=settings,
            )
        )

        missing = [setting.name for setting in fields(BrowserAgentSettings) if setting.name not in seen]
        assert missing == []
        mismatched = {
            setting.name: (getattr(settings, setting.name), seen[setting.name])
            for setting in fields(BrowserAgentSettings)
            if seen[setting.name] != getattr(settings, setting.name)
        }
        assert mismatched == {}
        assert 'available_file_paths' not in seen

    def test_disables_file_actions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The browser agent cannot parse downloads or upload host files."""
        seen: dict[str, object] = {}

        def record_init(self: object, **kwargs: object) -> None:
            seen.update(kwargs)

        monkeypatch.setattr(BrowserUseAgent, '__init__', record_init)
        default_browser_agent(
            BrowserTask(
                task='task',
                llm=None,
                browser_session=BrowserSession(),
                use_vision=True,
                output_schema=None,
                sensitive_data=None,
                extend_system_message=None,
                settings=BrowserAgentSettings(),
            )
        )

        tools = seen['tools']
        assert isinstance(tools, Tools)
        assert 'read_file' not in tools.registry.registry.actions  # pyright: ignore[reportUnknownMemberType]
        assert 'upload_file' not in tools.registry.registry.actions  # pyright: ignore[reportUnknownMemberType]

    def test_disables_file_actions_from_custom_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Custom action registries cannot restore either restricted action."""
        seen: dict[str, object] = {}

        def record_init(self: object, **kwargs: object) -> None:
            seen.update(kwargs)

        custom_tools = Tools[None]()
        monkeypatch.setattr(BrowserUseAgent, '__init__', record_init)
        default_browser_agent(
            BrowserTask(
                task='task',
                llm=None,
                browser_session=BrowserSession(),
                use_vision=True,
                output_schema=None,
                sensitive_data=None,
                extend_system_message=None,
                settings=BrowserAgentSettings(tools=custom_tools),
            )
        )

        assert seen['tools'] is custom_tools
        assert 'read_file' not in custom_tools.registry.registry.actions  # pyright: ignore[reportUnknownMemberType]
        assert 'upload_file' not in custom_tools.registry.registry.actions  # pyright: ignore[reportUnknownMemberType]


def _distinctly_valued_settings() -> BrowserAgentSettings:
    """`BrowserAgentSettings` with a value per field that no other field shares.

    Derived from each field's default rather than written out, so a setting added upstream is
    covered without anyone remembering to add a value for it. Fields whose default is `None`
    and whose type is not a scalar keep it: identity still holds for them, they just cannot
    catch a swap.
    """
    overrides: dict[str, object] = {}
    for index, setting in enumerate(fields(BrowserAgentSettings)):
        default = setting.default
        if isinstance(default, bool):
            overrides[setting.name] = not default
        elif isinstance(default, int):
            overrides[setting.name] = 1_000 + index
        elif isinstance(default, str):
            overrides[setting.name] = f'value-{index}'
    return BrowserAgentSettings(**overrides)  # type: ignore[arg-type]


class TestTeardownFailure:
    """`_kill` runs in a `finally`, where a raise replaces whatever was unwinding through it."""

    @pytest.fixture
    def failing_kill(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def raise_on_kill(self: BrowserSession) -> None:
            raise TimeoutError('event bus stop timed out')

        monkeypatch.setattr(BrowserSession, 'kill', raise_on_kill)

    @pytest.mark.parametrize('scope', ['call', 'agent'])
    async def test_a_failed_teardown_does_not_replace_the_result(
        self, failing_kill: None, scope: Literal['call', 'agent']
    ) -> None:
        toolset = BrowserUse[None](browser_agent=_success_factory('the answer'), session_scope=scope).get_toolset()
        assert isinstance(toolset, BrowserUseToolset)

        assert await toolset.browse_web('go') == 'the answer'

        await toolset.aclose()

    async def test_a_failed_teardown_does_not_replace_the_run_error(self, failing_kill: None) -> None:
        """The original cause is what the caller needs; a teardown error hides it."""

        class _Boom:
            def run(self, max_steps: int = 500) -> Awaitable[BrowserAgentHistory]:
                raise RuntimeError('the browser agent itself failed')

        def factory(request: BrowserTask) -> BrowserAgent:
            return _Boom()  # type: ignore[return-value]

        toolset = BrowserUse[None](browser_agent=factory, session_scope='agent').get_toolset()
        assert isinstance(toolset, BrowserUseToolset)

        with pytest.raises(RuntimeError, match='the browser agent itself failed'):
            await toolset.browse_web('go')

        await toolset.aclose()

    async def test_a_teardown_timeout_is_reported(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        async def hanging_kill(self: BrowserSession) -> None:
            await anyio.sleep_forever()

        monkeypatch.setattr(BrowserSession, 'kill', hanging_kill)
        monkeypatch.setattr('pydantic_ai_harness.browser_use._toolset._TEARDOWN_TIMEOUT', 0)
        toolset = BrowserUse[None](browser_agent=_success_factory()).get_toolset()

        with caplog.at_level(logging.WARNING):
            assert await toolset.browse_web('go') == 'done'

        assert 'browser-use session teardown timed out after 0 seconds' in caplog.text

    async def test_a_failed_teardown_is_retried_before_the_next_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        attempts: list[BrowserSession] = []

        async def fail_once(self: BrowserSession) -> None:
            attempts.append(self)
            if len(attempts) == 1:
                raise TimeoutError('event bus stop timed out')

        monkeypatch.setattr(BrowserSession, 'kill', fail_once)
        factory = _success_factory()
        toolset = BrowserUse[None](browser_agent=factory).get_toolset()

        assert await toolset.browse_web('first') == 'done'
        assert await toolset.browse_web('second') == 'done'

        assert attempts == [
            factory.requests[0].browser_session,
            factory.requests[0].browser_session,
            factory.requests[1].browser_session,
        ]

    async def test_session_build_failure_releases_the_call_slot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail_to_build(self: BrowserUseToolset[None]) -> BrowserSession:
            raise RuntimeError('browser session could not be created')

        monkeypatch.setattr(BrowserUseToolset, '_build_session', fail_to_build)
        toolset = BrowserUse[None]().get_toolset()

        with pytest.raises(RuntimeError, match='could not be created'):
            await toolset.browse_web('task')

        await toolset.aclose()

    async def test_aclose_cannot_finish_while_a_call_is_registering(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        retry_started = anyio.Event()
        second_retry = anyio.Event()
        release_retry = anyio.Event()
        running = anyio.Event()
        release_run = anyio.Event()
        retry_count = 0

        async def pause_first_retry(self: BrowserUseToolset[None]) -> None:
            nonlocal retry_count
            retry_count += 1
            if retry_count == 1:
                retry_started.set()
                await release_retry.wait()
            else:
                second_retry.set()

        class _BlockingAgent:
            async def run(self, max_steps: int = 500) -> _FakeHistory:
                running.set()
                await release_run.wait()
                return _FakeHistory(result='done', success=True)

        monkeypatch.setattr(BrowserUseToolset, '_retry_pending_cleanup', pause_first_retry)
        toolset = BrowserUse[None](browser_agent=lambda request: _BlockingAgent()).get_toolset()

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(toolset.browse_web, 'task')
            await retry_started.wait()
            task_group.start_soon(toolset.aclose)
            with anyio.move_on_after(0.1):
                await second_retry.wait()
            assert not second_retry.is_set()
            release_retry.set()
            await running.wait()
            assert not second_retry.is_set()
            release_run.set()

        assert second_retry.is_set()

    @pytest.mark.parametrize('scope', ['call', 'agent'])
    async def test_aclose_retries_cleanup_after_an_in_flight_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
        scope: Literal['call', 'agent'],
    ) -> None:
        running = anyio.Event()
        release = anyio.Event()
        closing = anyio.Event()
        attempts: list[BrowserSession] = []

        async def fail_once(self: BrowserSession) -> None:
            attempts.append(self)
            if len(attempts) == 1:
                raise TimeoutError('event bus stop timed out')

        class _BlockingAgent:
            async def run(self, max_steps: int = 500) -> _FakeHistory:
                running.set()
                await release.wait()
                return _FakeHistory(result='done', success=True)

        requests: list[BrowserTask] = []

        def factory(request: BrowserTask) -> _BlockingAgent:
            requests.append(request)
            return _BlockingAgent()

        async def close(toolset: BrowserUseToolset[None]) -> None:
            closing.set()
            await toolset.aclose()

        monkeypatch.setattr(BrowserSession, 'kill', fail_once)
        toolset = BrowserUse[None](browser_agent=factory, session_scope=scope).get_toolset()

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(toolset.browse_web, 'go')
            await running.wait()
            task_group.start_soon(close, toolset)
            await closing.wait()
            await anyio.sleep(0)
            release.set()

        assert attempts == [requests[0].browser_session, requests[0].browser_session]


class TestUntrustedContent:
    def test_the_instructions_say_the_result_is_untrusted(self) -> None:
        """The point of the capability is piping attacker-controllable page text into the host context."""
        instructions = BrowserUse[None]().get_instructions()

        assert isinstance(instructions, str)
        assert 'untrusted' in instructions


class TestCredentialsStayOutOfRepr:
    """Keeping values from the sub-agent's model and then printing them in a repr is an odd place to stop."""

    def test_the_capability_does_not_print_its_secrets(self) -> None:
        capability = BrowserUse[None](
            allowed_domains=['example.com'],
            sensitive_data={'x_password': 'hunter2'},
        )

        assert 'hunter2' not in repr(capability)

    def test_the_capability_does_not_print_its_browser_profile(self) -> None:
        """A profile carries proxy credentials and `storage_state` cookies."""
        capability = BrowserUse[None](browser_profile=BrowserProfile(user_data_dir='/tmp/secret-profile'))

        assert 'secret-profile' not in repr(capability)

    def test_the_capability_does_not_print_its_cdp_credentials(self) -> None:
        capability = BrowserUse[None](cdp_url='https://browser.example?token=secret-token')

        assert 'secret-token' not in repr(capability)

    def test_a_task_does_not_print_its_secrets(self) -> None:
        task = BrowserTask(
            task='t',
            llm=None,
            browser_session=BrowserSession(),
            use_vision=True,
            output_schema=None,
            sensitive_data={'x_password': 'hunter2'},
            extend_system_message=None,
            settings=BrowserAgentSettings(),
        )

        assert 'hunter2' not in repr(task)


class TestSensitiveDataSafety:
    def test_flat_secrets_without_an_allowlist_raise(self) -> None:
        with pytest.raises(ValueError, match='Flat `sensitive_data` values require'):
            BrowserUse[None](sensitive_data={'x_password': 'hunter2'})

    @pytest.mark.parametrize(
        'capability',
        [
            BrowserUse[None](
                allowed_domains=['example.com'],
                sensitive_data={'x_password': 'hunter2'},
            ),
            BrowserUse[None](
                browser_profile=BrowserProfile(allowed_domains=['example.com']),
                sensitive_data={'x_password': 'hunter2'},
            ),
            BrowserUse[None](
                sensitive_data={'https://example.com': {'x_password': 'hunter2'}},
            ),
        ],
    )
    def test_restricted_or_domain_scoped_secrets_are_allowed(self, capability: BrowserUse[None]) -> None:
        assert capability.sensitive_data is not None

    def test_empty_capability_allowlist_raises(self) -> None:
        with pytest.raises(ValueError, match='Flat `sensitive_data` values require'):
            BrowserUse[None](
                allowed_domains=[],
                sensitive_data={'x_password': 'hunter2'},
            )

    def test_empty_profile_allowlist_raises(self) -> None:
        with pytest.raises(ValueError, match='Flat `sensitive_data` values require'):
            BrowserUse[None](
                browser_profile=BrowserProfile(allowed_domains=[]),
                sensitive_data={'x_password': 'hunter2'},
            )

    @pytest.mark.parametrize('allowed_domains', [['*'], ['*.*'], ['*.example.com'], ['https://*'], ['*://*']])
    def test_host_glob_capability_allowlist_raises(self, allowed_domains: list[str]) -> None:
        with pytest.raises(ValueError, match='Flat `sensitive_data` values require'):
            BrowserUse[None](
                allowed_domains=allowed_domains,
                sensitive_data={'x_password': 'hunter2'},
            )

    def test_host_glob_profile_allowlist_raises(self) -> None:
        with pytest.raises(ValueError, match='Flat `sensitive_data` values require'):
            BrowserUse[None](
                browser_profile=BrowserProfile(allowed_domains=['*.example.com']),
                sensitive_data={'x_password': 'hunter2'},
            )

    def test_host_glob_allowlist_without_flat_secrets_is_allowed(self) -> None:
        capability = BrowserUse[None](allowed_domains=['*.example.com'])

        assert capability.allowed_domains == ['*.example.com']

    def test_mutating_an_allowlist_before_toolset_construction_raises(self) -> None:
        capability = BrowserUse[None](
            allowed_domains=['safe.example'],
            sensitive_data={'x_password': 'hunter2'},
        )
        assert capability.allowed_domains is not None
        capability.allowed_domains.clear()

        with pytest.raises(ValueError, match='Flat `sensitive_data` values require'):
            capability.get_toolset()

    def test_adding_a_flat_secret_before_toolset_construction_raises(self) -> None:
        capability = BrowserUse[None](
            sensitive_data={'https://safe.example': {'x_password': 'hunter2'}},
        )
        assert capability.sensitive_data is not None
        capability.sensitive_data['x_token'] = 'abc123'

        with pytest.raises(ValueError, match='Flat `sensitive_data` values require'):
            capability.get_toolset()

    async def test_toolset_snapshots_flat_secret_configuration(self, kill_calls: list[BrowserSession]) -> None:
        factory = _success_factory()
        secrets: dict[str, str | dict[str, str]] = {'x_password': 'hunter2'}
        capability = BrowserUse[None](
            browser_agent=factory,
            allowed_domains=['safe.example'],
            sensitive_data=secrets,
        )
        toolset = capability.get_toolset()
        assert capability.allowed_domains is not None
        capability.allowed_domains.clear()
        secrets['x_token'] = 'abc123'

        await toolset.browse_web('task')

        request = factory.requests[0]
        assert request.browser_session.browser_profile.allowed_domains == ['safe.example']
        assert request.sensitive_data == {'x_password': 'hunter2'}

    async def test_toolset_snapshots_profile_allowlist_with_flat_secrets(
        self, kill_calls: list[BrowserSession]
    ) -> None:
        factory = _success_factory()
        profile = BrowserProfile(allowed_domains=['safe.example'])
        capability = BrowserUse[None](
            browser_agent=factory,
            browser_profile=profile,
            sensitive_data={'x_password': 'hunter2'},
        )
        toolset = capability.get_toolset()
        assert profile.allowed_domains is not None
        profile.allowed_domains.clear()

        await toolset.browse_web('task')

        assert factory.requests[0].browser_session.browser_profile.allowed_domains == ['safe.example']

    def test_empty_capability_allowlist_overrides_profile_allowlist_and_raises(self) -> None:
        with pytest.raises(ValueError, match='Flat `sensitive_data` values require'):
            BrowserUse[None](
                browser_profile=BrowserProfile(allowed_domains=['safe.example']),
                allowed_domains=[],
                sensitive_data={'x_password': 'hunter2'},
            )


class TestSessionScope:
    async def test_agent_scope_reuses_one_session(self, kill_calls: list[BrowserSession]) -> None:
        factory = _success_factory()
        capability = BrowserUse[None](session_scope='agent', browser_agent=factory)
        toolset = capability.get_toolset()

        await toolset.browse_web('first task')
        await toolset.browse_web('second task')

        first, second = factory.requests
        assert first.browser_session is second.browser_session
        # keep_alive stops browser_use.Agent from killing the shared session
        # at the end of each of its runs.
        assert first.browser_session.browser_profile.keep_alive is True
        assert kill_calls == []

        await capability.aclose()
        assert kill_calls == [first.browser_session]
        await capability.aclose()
        assert len(kill_calls) == 1

    async def test_agent_scope_error_resets_session(self, kill_calls: list[BrowserSession]) -> None:
        agent = _FakeBrowserAgent(_FakeHistory(result='done', success=True), error=RuntimeError('crash'))
        factory = _FakeFactory(agent)
        toolset = BrowserUse[None](session_scope='agent', browser_agent=factory).get_toolset()

        with pytest.raises(RuntimeError, match='crash'):
            await toolset.browse_web('first task')
        assert kill_calls == [factory.requests[0].browser_session]

        agent.error = None
        await toolset.browse_web('second task')
        assert factory.requests[1].browser_session is not factory.requests[0].browser_session

    async def test_agent_scope_schema_retry_keeps_session(self, kill_calls: list[BrowserSession]) -> None:
        agent = _FakeBrowserAgent(_FakeHistory(result='bad', success=True, structured_error=_validation_error()))
        factory = _FakeFactory(agent)
        capability = BrowserUse[None](output_schema=_Facts, session_scope='agent', browser_agent=factory)
        toolset = capability.get_toolset()

        with pytest.raises(ModelRetry):
            await toolset.browse_web('first task')
        # The run itself finished; only the result was rejected, so the shared
        # browser survives for the follow-up call.
        assert kill_calls == []

        agent.history = _FakeHistory(result='{"name": "Pro", "price_usd": 20}', success=True)
        agent.history.structured = _Facts(name='Pro', price_usd=20)
        await toolset.browse_web('second task')
        assert factory.requests[1].browser_session is factory.requests[0].browser_session
        await capability.aclose()

    async def test_capability_as_async_context_manager(self, kill_calls: list[BrowserSession]) -> None:
        factory = _success_factory()
        async with BrowserUse[None](session_scope='agent', browser_agent=factory) as capability:
            await capability.get_toolset().browse_web('task')
            assert kill_calls == []
        assert kill_calls == [factory.requests[0].browser_session]

    async def test_aclose_before_any_call_is_a_no_op(self, kill_calls: list[BrowserSession]) -> None:
        capability = BrowserUse[None](session_scope='agent')
        await capability.aclose()
        assert kill_calls == []

    async def test_browse_web_after_aclose_does_not_reopen(self, kill_calls: list[BrowserSession]) -> None:
        """`aclose()` closes the shared session for good, rather than letting the next call reopen it.

        A reopened session is created with `keep_alive`, and the `aclose()` that would
        have closed it has already returned, so it would outlive the run.
        """
        factory = _success_factory()
        capability = BrowserUse[None](session_scope='agent', browser_agent=factory)
        toolset = capability.get_toolset()

        await toolset.browse_web('first task')
        await capability.aclose()

        with pytest.raises(RuntimeError, match='closed'):
            await toolset.browse_web('second task')

        assert len(factory.requests) == 1
        assert kill_calls == [factory.requests[0].browser_session]

    async def test_call_queued_behind_aclose_does_not_reopen(self, kill_calls: list[BrowserSession]) -> None:
        """The same guard, reached the way it actually happens: `aclose()` and a call race for the lock.

        A `browse_web` that started while a call was in flight waits on the session lock,
        so it reaches the shared session only after `aclose()` has taken it away.
        """
        running = anyio.Event()
        release = anyio.Event()

        class _BlockingAgent:
            async def run(self, max_steps: int = 500) -> _FakeHistory:
                running.set()
                await release.wait()
                return _FakeHistory(result='done', success=True)

        requests: list[BrowserTask] = []

        def factory(request: BrowserTask) -> _BlockingAgent:
            requests.append(request)
            return _BlockingAgent()

        capability = BrowserUse[None](session_scope='agent', browser_agent=factory)
        toolset = capability.get_toolset()
        queued_error: list[RuntimeError] = []

        async def queued_call() -> None:
            with pytest.raises(RuntimeError, match='closed') as caught:
                await toolset.browse_web('second task')
            queued_error.append(caught.value)

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(toolset.browse_web, 'first task')
            await running.wait()
            # Both queue on the session lock the in-flight call holds, `aclose` first.
            task_group.start_soon(capability.aclose)
            task_group.start_soon(queued_call)
            release.set()

        assert len(queued_error) == 1
        assert len(requests) == 1
        assert kill_calls == [requests[0].browser_session]

    def test_toolset_is_cached(self) -> None:
        capability = BrowserUse[None]()
        assert capability.get_toolset() is capability.get_toolset()


class TestBrowserUse:
    def test_instructions_reference_the_tool(self) -> None:
        instructions = BrowserUse[None]().get_instructions()
        assert isinstance(instructions, str)
        assert '`browse_web`' in instructions

    def test_custom_guidance_retains_the_untrusted_content_rule(self) -> None:
        instructions = BrowserUse[None](guidance='Delegate web tasks.').get_instructions()
        assert isinstance(instructions, str)
        assert instructions.startswith('Delegate web tasks.')
        assert 'untrusted' in instructions

    def test_empty_guidance_disables_instructions(self) -> None:
        assert BrowserUse[None](guidance='').get_instructions() is None

    async def test_agent_run_returns_tool_result(self, kill_calls: list[BrowserSession]) -> None:
        factory = _success_factory('The answer is 42.')
        agent = Agent(TestModel(), capabilities=[BrowserUse(browser_agent=factory)])

        result = await agent.run('Find the answer on example.com.')

        parts = [
            part
            for message in result.all_messages()
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'browse_web'
        ]
        assert [part.content for part in parts] == ['The answer is 42.']


class TestAgentSpec:
    def test_spec_schema_includes_browser_use(self) -> None:
        schema = AgentSpec.model_json_schema_with_capabilities([BrowserUse])
        assert 'BrowserUse' in json.dumps(schema)

    def test_from_spec_builds_capability(self) -> None:
        capability = BrowserUse[None].from_spec(
            allowed_domains=['example.com'],
            block_ip_addresses=False,
            headless=False,
            max_steps=10,
            use_vision='auto',
            sensitive_data={'x_user': 'kacper'},
            extend_system_message='Stay on the English site.',
            session_scope='agent',
            cdp_url='http://localhost:9222',
            guidance='Delegate.',
        )
        assert capability.allowed_domains == ['example.com']
        assert capability.block_ip_addresses is False
        assert capability.headless is False
        assert capability.max_steps == 10
        assert capability.use_vision == 'auto'
        assert capability.sensitive_data == {'x_user': 'kacper'}
        assert capability.extend_system_message == 'Stay on the English site.'
        assert capability.session_scope == 'agent'
        assert capability.cdp_url == 'http://localhost:9222'
        assert capability.guidance == 'Delegate.'
        assert capability.llm is None
        assert capability.browser_profile is None
        assert capability.output_schema is None
        assert capability.agent_settings is None
        assert capability.browser_agent is None

    def test_agent_loads_from_spec_file(self, tmp_path: Path) -> None:
        spec = tmp_path / 'agent.yaml'
        spec.write_text('model: test\ncapabilities:\n  - BrowserUse:\n      max_steps: 5\n')
        agent = Agent.from_file(spec, custom_capability_types=[BrowserUse])
        assert isinstance(agent, Agent)
