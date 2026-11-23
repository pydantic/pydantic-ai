"""Tests for the `CDPBrowser` capability.

Chrome is faked throughout: `FakeChrome` answers the DevTools Protocol over a real in-process websocket, and a fake
process stands in for the browser `launch_chrome` starts, so no Chrome binary runs. What a real Chrome does with
these commands is checked by the hand-run `scripts/cdp_browser_smoke.py`.

None of these is a VCR test: `CDPBrowser` drives Chrome over a websocket rather than a model provider's HTTP API, so
there is no provider request for a cassette to record. The agents run on `FunctionModel`, which scripts the tool calls.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import anyio
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.abstract import CapabilityOrdering
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness import CDPBrowser
from pydantic_ai_harness.cdp_browser import (
    DEFAULT_ALLOWED_CDP_METHODS,
    CDPBrowserToolset,
    EgressPolicy,
    _client as client_module,
    _toolset as toolset_module,
)
from tests.conftest import no_leaked_tasks

from ._fake_chrome import PAGE_SESSION, FakeChrome, pause_expression


def calls_then_reply(*calls: dict[str, object]) -> FunctionModel:
    """A model that makes each `browser_cdp` call in turn, then replies."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = len(tool_returns(messages))
        if step < len(calls):
            return ModelResponse(parts=[ToolCallPart('browser_cdp', calls[step])])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(respond)


def tool_returns(messages: list[ModelMessage]) -> list[object]:
    return [part.content for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]


def evaluate(expression: str, **extra: object) -> dict[str, object]:
    return {'method': 'Runtime.evaluate', 'params': {'expression': expression, 'returnByValue': True}, **extra}


def profile_of(chrome: FakeChrome) -> Path:
    return Path(next(arg for arg in chrome.launches[-1] if arg.startswith('--user-data-dir=')).split('=', 1)[1])


def commands(chrome: FakeChrome, session_id: str | None) -> list[str]:
    return [method for session, method, _ in chrome.received if session == session_id]


class TestCDPBrowser:
    async def test_each_run_gets_its_own_browser(self, chrome: FakeChrome):
        agent = Agent(
            calls_then_reply(
                {
                    'method': 'Page.navigate',
                    'params': {'url': 'https://example.com'},
                    'wait_for_event': 'Page.loadEventFired',
                },
                evaluate('document.title'),
            ),
            capabilities=[CDPBrowser()],
        )
        result = await agent.run('title?')
        assert tool_returns(result.all_messages()) == snapshot(
            [
                {'result': {'frameId': 'PAGE-TARGET'}, 'event': {'timestamp': 1.0}},
                {'result': {'type': 'string', 'value': 'Example Domain'}},
            ]
        )
        assert chrome.verdicts == [('https://example.com', 'continue')]
        assert not profile_of(chrome).exists()
        await agent.run('again')
        assert len(chrome.launches) == 2
        assert chrome.launches[0][:2] == ['/fake/chrome', '--headless=new'] and '--no-sandbox' not in chrome.launches[0]
        assert commands(chrome, None) == snapshot(
            [
                'Browser.setDownloadBehavior',
                'Target.setAutoAttach',
                'Browser.close',
                'Browser.setDownloadBehavior',
                'Target.setAutoAttach',
                'Browser.close',
            ]
        )
        assert commands(chrome, PAGE_SESSION)[:5] == snapshot(
            [
                'Fetch.enable',
                'Page.enable',
                'Page.addScriptToEvaluateOnNewDocument',
                'Target.setAutoAttach',
                'Runtime.runIfWaitingForDebugger',
            ]
        )
        assert [params for _, method, params in chrome.received if method == 'Browser.setDownloadBehavior'] == [
            {'behavior': 'deny'}
        ] * 2
        # Chrome exited when it was asked to, so there was no process group left to kill.
        assert chrome.killed_groups == []

    async def test_launch_options_reach_chrome(self, chrome: FakeChrome):
        capability = CDPBrowser(headless=False, chrome_sandbox=False, chrome_path='/opt/chrome/chrome')
        await Agent(calls_then_reply(), capabilities=[capability]).run('hi')
        args = chrome.launches[0]
        assert args[0] == '/opt/chrome/chrome'
        assert '--no-sandbox' in args and '--headless=new' not in args

    async def test_disallowed_commands_and_urls_are_refused(self, chrome: FakeChrome):
        agent = Agent(
            calls_then_reply(
                {'method': 'Browser.close'},
                {'method': 'Page.navigate', 'params': {'url': 'file:///etc/passwd'}},
                # a success between refusals, so they stay under the three retries allowed in a row
                evaluate('document.title'),
                {'method': 'Page.navigate', 'params': {'url': 'http://169.254.169.254/latest/meta-data/'}},
                {'method': 'Page.navigate', 'params': {'url': 'https://elsewhere.org/'}},
            ),
            capabilities=[CDPBrowser(allowed_domains=['example.com'])],
        )
        result = await agent.run('go')
        refusals = tool_returns(result.all_messages())
        assert str(refusals[0]).startswith('Browser.close is not allowed, allowed commands: DOM.describeNode, ')
        assert refusals[1:] == snapshot(
            [
                "Only http and https URLs can be opened, got 'file:///etc/passwd'",
                {'result': {'type': 'string', 'value': 'Example Domain'}},
                "'http://169.254.169.254/latest/meta-data/' is refused by the egress policy: blocked private or link-local address",
                "'https://elsewhere.org/' is refused by the egress policy: domain not in allowed_domains",
            ]
        )
        assert 'Page.navigate' not in commands(chrome, PAGE_SESSION)

    async def test_allowed_methods_can_be_narrowed(self, chrome: FakeChrome):
        agent = Agent(
            calls_then_reply({'method': 'Input.insertText', 'params': {'text': 'hi'}}),
            capabilities=[CDPBrowser(allowed_methods=['Runtime.evaluate'])],
        )
        result = await agent.run('go')
        assert tool_returns(result.all_messages()) == [
            'Input.insertText is not allowed, allowed commands: Runtime.evaluate'
        ]

    async def test_instructions_describe_the_tool_and_its_reach(self, chrome: FakeChrome):
        seen: list[str | None] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info.instructions)
            assert [t.name for t in info.function_tools] == ['browser_cdp']
            return ModelResponse(parts=[TextPart('ok')])

        await Agent(FunctionModel(respond), capabilities=[CDPBrowser(allowed_domains=['example.com'])]).run('hi')
        assert seen[0] is not None and 'Chrome DevTools Protocol' in seen[0]
        assert seen[0].splitlines()[-1] == snapshot(
            "Hosts that top-level navigation and page scripts' data requests (`fetch`, XHR, EventSource, `sendBeacon`) "
            'may reach: example.com (and their subdomains) (private/internal addresses blocked). '
            'A refused request fails with `net::ERR_BLOCKED_BY_CLIENT`.'
        )

    async def test_instructions_name_a_non_default_reach(self, chrome: FakeChrome):
        """A policy that bounds other kinds of request is described by the policy, not by the default sentence."""
        seen: list[str | None] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info.instructions)
            return ModelResponse(parts=[TextPart('ok')])

        policy = EgressPolicy(allowed_domains=['example.com'], allowlist_reach=frozenset({'subresource'}))
        await Agent(FunctionModel(respond), capabilities=[CDPBrowser(policy=policy)]).run('hi')
        assert seen[0] is not None
        assert seen[0].splitlines()[-1] == snapshot(
            'Hosts the browser may reach: example.com (and their subdomains) (allowlist bounds subresource) '
            '(private/internal addresses blocked). A refused request fails with `net::ERR_BLOCKED_BY_CLIENT`.'
        )

    async def test_every_target_is_guarded(self, chrome: FakeChrome):
        """Requests from the page, its frames, a popup and workers all wait for the egress policy's verdict.

        A dedicated worker has no `Fetch` domain, so its requests are paused on its page's session; any other target
        whose requests cannot be paused is never resumed.
        """
        chrome.child_targets = [
            {'sessionId': 'iframe-session', 'targetId': 'IFRAME-TARGET', 'type': 'iframe'},
            {'sessionId': 'popup-session', 'targetId': 'POPUP-TARGET', 'type': 'page'},
            {'sessionId': 'worker-session', 'targetId': 'WORKER-TARGET', 'type': 'worker'},
            {'sessionId': 'shared-session', 'targetId': 'SHARED-TARGET', 'type': 'shared_worker'},
            # stands in for any target type Chrome offers no `Fetch` domain on
            {'sessionId': 'worklet-session', 'targetId': 'WORKLET-TARGET', 'type': 'auction_worklet'},
        ]
        chrome.error_targets = {'auction_worklet'}
        requests = [
            pause_expression('https://example.com/'),
            pause_expression('https://other.org/'),
            pause_expression('https://other.org/frame', frame_id='CHILD-FRAME'),
            pause_expression('https://cdn.other.org/logo.png', resource_type='Image'),
            pause_expression('https://other.org/api', resource_type='Fetch'),
            pause_expression(
                'http://10.0.0.1/', resource_type='Document', frame_id='IFRAME-TARGET', session_id='iframe-session'
            ),
            pause_expression('https://other.org/', frame_id='POPUP-TARGET', session_id='popup-session'),
            pause_expression('https://other.org/frame', session_id='unknown-session'),
            pause_expression(
                'http://10.0.0.2/', resource_type='Fetch', frame_id='SHARED-TARGET', session_id='shared-session'
            ),
        ]
        agent = Agent(
            calls_then_reply(*(evaluate(request) for request in requests)),
            capabilities=[CDPBrowser(allowed_domains=['example.com'])],
        )
        await agent.run('go')
        assert chrome.verdicts == snapshot(
            [
                ('https://example.com/', 'continue'),
                ('https://other.org/', 'fail'),
                ('https://other.org/frame', 'continue'),
                ('https://cdn.other.org/logo.png', 'continue'),
                ('https://other.org/api', 'fail'),
                ('http://10.0.0.1/', 'fail'),
                ('https://other.org/', 'fail'),
                ('https://other.org/frame', 'continue'),
                ('http://10.0.0.2/', 'fail'),
            ]
        )
        assert commands(chrome, 'iframe-session') == snapshot(
            [
                'Fetch.enable',
                'Page.enable',
                'Page.addScriptToEvaluateOnNewDocument',
                'Target.setAutoAttach',
                'Runtime.runIfWaitingForDebugger',
                'Fetch.failRequest',
            ]
        )
        assert commands(chrome, 'worker-session') == ['Target.setAutoAttach', 'Runtime.runIfWaitingForDebugger']
        assert commands(chrome, 'shared-session') == snapshot(
            ['Fetch.enable', 'Target.setAutoAttach', 'Runtime.runIfWaitingForDebugger', 'Fetch.failRequest']
        )
        assert commands(chrome, 'worklet-session') == ['Fetch.enable']
        assert {params['errorReason'] for _, method, params in chrome.received if method == 'Fetch.failRequest'} == {
            'BlockedByClient'
        }

    async def test_a_redirect_is_guarded_hop_by_hop(self, chrome: FakeChrome):
        """Chrome pauses each redirect hop as a request of its own, so a hop to a refused host fails the navigation."""
        chrome.redirects = {'https://example.com/out': 'http://10.0.0.1/'}
        result = await Agent(
            calls_then_reply({'method': 'Page.navigate', 'params': {'url': 'https://example.com/out'}}),
            capabilities=[CDPBrowser()],
        ).run('go')
        assert tool_returns(result.all_messages()) == snapshot(
            [{'frameId': 'PAGE-TARGET', 'errorText': 'net::ERR_BLOCKED_BY_CLIENT'}]
        )
        assert chrome.verdicts == [('https://example.com/out', 'continue'), ('http://10.0.0.1/', 'fail')]

    async def test_a_target_chrome_never_answers_stays_paused(
        self, chrome: FakeChrome, monkeypatch: pytest.MonkeyPatch
    ):
        """The guard never resumes a target it could not guard, and the run still ends with the handler stopped."""
        chrome.child_targets = [{'sessionId': 'frame-session', 'targetId': 'FRAME-TARGET', 'type': 'iframe'}]
        chrome.unanswered_targets = {'iframe'}
        # Chrome ignoring close keeps the connection open, so the waiting handler is still running at teardown.
        chrome.ignore_close = True
        monkeypatch.setattr(client_module, 'SHUTDOWN_GRACE_SECONDS', 0.01)
        result = await Agent(calls_then_reply(evaluate('document.title')), capabilities=[CDPBrowser()]).run('go')
        assert tool_returns(result.all_messages()) == [{'result': {'type': 'string', 'value': 'Example Domain'}}]
        assert commands(chrome, 'frame-session') == ['Fetch.enable']
        assert chrome.killed_groups == [chrome.process.pid]
        assert not profile_of(chrome).exists()

    async def test_a_cancelled_run_still_closes_chrome(self, chrome: FakeChrome):
        """Teardown is shielded, so cancelling a run mid-command still closes Chrome and removes its profile."""
        agent = Agent(calls_then_reply(evaluate('hang')), capabilities=[CDPBrowser()])
        with no_leaked_tasks(), anyio.CancelScope() as scope:
            async with anyio.create_task_group() as tg:
                tg.start_soon(agent.run, 'go')
                await chrome.hanging.wait()
                scope.cancel()
        assert scope.cancelled_caught
        assert chrome.process.returncode == 0
        assert commands(chrome, None)[-1] == 'Browser.close'
        assert not profile_of(chrome).exists()

    async def test_instructions_follow_the_policy(self, chrome: FakeChrome):
        """A host added to the allowlist after construction is both allowed and named to the model."""
        seen: list[str | None] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info.instructions)
            return ModelResponse(parts=[TextPart('ok')])

        domains = ['example.com']
        browser = CDPBrowser(allowed_domains=domains)
        domains.append('other.org')
        await Agent(FunctionModel(respond), capabilities=[browser]).run('hi')
        assert seen[0] is not None and 'other.org' in seen[0].splitlines()[-1]

    async def test_malformed_messages_are_ignored(self, chrome: FakeChrome):
        """Messages no real Chrome sends are skipped rather than failing the run."""
        chrome.browser_events = [
            ['not', 'an', 'object'],
            {'neither': 'an id nor a method'},
            {'id': 987654, 'result': {}},
            {'method': 'Target.attachedToTarget', 'params': {'targetInfo': {'type': 'page'}}},
            {'method': 'Fetch.requestPaused', 'params': {'requestId': 'r', 'request': {'url': 'https://x.org'}}},
            {'method': 'Fetch.requestPaused', 'params': {'request': {}}, 'sessionId': PAGE_SESSION},
            {'method': 'Fetch.requestPaused', 'params': {'requestId': 'r', 'request': 'x'}, 'sessionId': PAGE_SESSION},
            {'method': 'Page.javascriptDialogOpening', 'params': {'type': 'alert'}},
            {'method': 'Page.loadEventFired', 'params': ['not', 'an', 'object'], 'sessionId': PAGE_SESSION},
        ]
        chrome.bare_replies = {'DOM.getDocument'}
        result = await Agent(
            calls_then_reply({'method': 'DOM.getDocument'}, evaluate('document.title')), capabilities=[CDPBrowser()]
        ).run('go')
        assert tool_returns(result.all_messages()) == [{}, {'result': {'type': 'string', 'value': 'Example Domain'}}]
        assert chrome.verdicts == []

    async def test_dialogs_are_dismissed(self, chrome: FakeChrome):
        """A dialog would block the page until it is answered; a `beforeunload` one is accepted so navigation proceeds."""
        result = await Agent(
            calls_then_reply(
                evaluate(json.dumps({'dialog': 'alert'})), evaluate(json.dumps({'dialog': 'beforeunload'}))
            ),
            capabilities=[CDPBrowser()],
        ).run('go')
        assert tool_returns(result.all_messages()) == snapshot(
            [{'result': {'type': 'boolean', 'value': False}}, {'result': {'type': 'boolean', 'value': True}}]
        )

    async def test_a_command_that_never_answers_times_out(self, chrome: FakeChrome):
        """The tool owns its deadline, so a hung page script is a retry and the tab is stopped and answers again."""
        result = await Agent(
            calls_then_reply(evaluate('hang', timeout=0.05), evaluate('document.title')), capabilities=[CDPBrowser()]
        ).run('go')
        assert tool_returns(result.all_messages()) == snapshot(
            [
                'Runtime.evaluate did not answer within 0.05s, so the page script was stopped; try a simpler command or a longer timeout',
                {'result': {'type': 'string', 'value': 'Example Domain'}},
            ]
        )
        assert 'Runtime.terminateExecution' in commands(chrome, PAGE_SESSION)

    async def test_an_event_that_never_arrives_times_out(self, chrome: FakeChrome):
        """An abandoned waiter is dropped by the next event, or failed with the connection when none comes."""
        navigate = {'method': 'Page.navigate', 'params': {'url': 'https://example.com'}}
        stopped = {**navigate, 'wait_for_event': 'Page.frameStoppedLoading', 'timeout': 0.05}
        result = await Agent(
            calls_then_reply(stopped, {**navigate, 'wait_for_event': 'Page.loadEventFired'}, stopped),
            capabilities=[CDPBrowser()],
        ).run('go')
        timed_out = "Timed out after 0.05s waiting for 'Page.frameStoppedLoading'; the command returned {'frameId': 'PAGE-TARGET'}"
        assert tool_returns(result.all_messages()) == [
            timed_out,
            {'result': {'frameId': 'PAGE-TARGET'}, 'event': {'timestamp': 1.0}},
            timed_out,
        ]

    async def test_protocol_errors_are_retries(self, chrome: FakeChrome):
        chrome.errors = {'DOM.querySelector': 'Could not find node with given id'}
        result = await Agent(
            calls_then_reply({'method': 'DOM.querySelector', 'params': {'nodeId': 0, 'selector': 'x'}}),
            capabilities=[CDPBrowser()],
        ).run('go')
        assert tool_returns(result.all_messages()) == ['DOM.querySelector failed: Could not find node with given id']

    async def test_screenshots_are_image_content(self, chrome: FakeChrome, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(toolset_module, 'MAX_SCREENSHOT_BYTES', len(chrome.screenshot))
        result = await Agent(
            calls_then_reply(
                {'method': 'Page.captureScreenshot'},
                {'method': 'Page.captureScreenshot', 'params': {'format': 'jpeg', 'quality': 50}},
            ),
            capabilities=[CDPBrowser()],
        ).run('go')
        assert tool_returns(result.all_messages()) == snapshot(
            ['Screenshot attached (image/png, 15 bytes).', 'Screenshot attached (image/jpeg, 15 bytes).']
        )
        images = [
            item
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, UserPromptPart)
            for item in part.content
            if isinstance(item, BinaryContent)
        ]
        assert [(image.data, image.media_type) for image in images] == [
            (chrome.screenshot, 'image/png'),
            (chrome.screenshot, 'image/jpeg'),
        ]

    async def test_an_oversized_screenshot_is_an_error(self, chrome: FakeChrome, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(toolset_module, 'MAX_SCREENSHOT_BYTES', len(chrome.screenshot) - 1)
        result = await Agent(calls_then_reply({'method': 'Page.captureScreenshot'}), capabilities=[CDPBrowser()]).run(
            'go'
        )
        assert tool_returns(result.all_messages()) == snapshot(
            [
                "Error: screenshot is 15 bytes, over the 14 byte image limit; capture a smaller `clip`, or pass `format: 'jpeg'` with a lower `quality`."
            ]
        )

    async def test_a_lost_connection_ends_the_run(self, chrome: FakeChrome, monkeypatch: pytest.MonkeyPatch):
        """A crashed Chrome is not something the model can retry; teardown still kills it."""
        monkeypatch.setattr(client_module, 'SHUTDOWN_GRACE_SECONDS', 0.01)
        with pytest.raises(ConnectionError, match='Chrome DevTools connection closed'):
            await Agent(calls_then_reply(evaluate('disconnect')), capabilities=[CDPBrowser()]).run('go')
        assert chrome.killed_groups == [chrome.process.pid]
        assert not profile_of(chrome).exists()

    async def test_the_toolset_must_be_entered(self):
        with pytest.raises(RuntimeError, match='must be entered'):
            await CDPBrowserToolset[None]().browser_cdp('Runtime.evaluate')

    async def test_nested_entry_shares_one_browser(self, chrome: FakeChrome):
        toolset = CDPBrowserToolset[None]()
        async with toolset:
            async with toolset:
                assert await toolset.browser_cdp('Runtime.evaluate', {'expression': '1'}) == {
                    'result': {'type': 'string', 'value': 'Example Domain'}
                }
            assert commands(chrome, None).count('Browser.close') == 0
        assert len(chrome.launches) == 1 and commands(chrome, None).count('Browser.close') == 1

    def test_policy_excludes_the_shorthands(self):
        with pytest.raises(UserError, match='Pass either policy or the allowed_domains/block_private_addresses'):
            CDPBrowser(policy=EgressPolicy(), allowed_domains=['example.com'])

    async def test_a_policy_decides_alone(self, chrome: FakeChrome):
        policy = EgressPolicy(blocked_domains=['ads.example.com'], block_private_addresses=False)
        agent = Agent(
            calls_then_reply(
                evaluate(pause_expression('http://127.0.0.1:8000/', resource_type='Fetch')),
                evaluate(pause_expression('https://ads.example.com/pixel.gif', resource_type='Image')),
            ),
            capabilities=[CDPBrowser(policy=policy)],
        )
        await agent.run('go')
        assert chrome.verdicts == [
            ('http://127.0.0.1:8000/', 'continue'),
            ('https://ads.example.com/pixel.gif', 'fail'),
        ]

    def test_from_spec(self):
        capability = CDPBrowser[None].from_spec(
            chrome_sandbox=False, allowed_methods=['Runtime.evaluate'], allowed_domains=['example.com']
        )
        assert (capability.chrome_sandbox, capability.allowed_methods, capability.allowed_domains) == (
            False,
            ('Runtime.evaluate',),
            ['example.com'],
        )
        assert CDPBrowser[None].from_spec().allowed_methods == DEFAULT_ALLOWED_CDP_METHODS

    def test_methods_given_as_a_list_compare_equal(self):
        """The same commands as a list and as a tuple are one configuration, so `combine` keeps one browser."""
        assert CDPBrowser(allowed_methods=list(DEFAULT_ALLOWED_CDP_METHODS)) == CDPBrowser()


class TestCDPBrowserLaunch:
    async def test_chrome_that_exits_at_startup(self, chrome: FakeChrome):
        chrome.write_port, chrome.exit_on_start = False, True
        with pytest.raises(RuntimeError, match='Chrome exited with code 21 before opening its DevTools port'):
            await Agent(calls_then_reply(), capabilities=[CDPBrowser()]).run('hi')
        assert not profile_of(chrome).exists()

    async def test_chrome_that_never_opens_its_port(self, chrome: FakeChrome, monkeypatch: pytest.MonkeyPatch):
        chrome.write_port = False
        monkeypatch.setattr(client_module, 'STARTUP_TIMEOUT_SECONDS', 0.05)
        with pytest.raises(RuntimeError, match=r'Chrome did not open its DevTools port within 0\.05s'):
            await Agent(calls_then_reply(), capabilities=[CDPBrowser()]).run('hi')
        assert chrome.killed_groups == [chrome.process.pid]

    async def test_chrome_that_never_attaches_its_tab(self, chrome: FakeChrome, monkeypatch: pytest.MonkeyPatch):
        chrome.attach_page = False
        monkeypatch.setattr(toolset_module, 'SETUP_TIMEOUT_SECONDS', 0.05)
        with pytest.raises(RuntimeError, match=r'Chrome did not attach its first tab within 0\.05s'):
            await Agent(calls_then_reply(), capabilities=[CDPBrowser()]).run('hi')
        assert commands(chrome, None)[-1] == 'Browser.close'

    async def test_chrome_that_ignores_close_is_killed(self, chrome: FakeChrome, monkeypatch: pytest.MonkeyPatch):
        chrome.ignore_close = True
        monkeypatch.setattr(client_module, 'SHUTDOWN_GRACE_SECONDS', 0.01)
        await Agent(calls_then_reply(), capabilities=[CDPBrowser()]).run('hi')
        assert chrome.killed_groups == [chrome.process.pid]

    async def test_windows_kills_the_process_instead_of_its_group(
        self, chrome: FakeChrome, monkeypatch: pytest.MonkeyPatch
    ):
        chrome.ignore_close = True
        monkeypatch.setattr(client_module, 'SHUTDOWN_GRACE_SECONDS', 0.01)
        monkeypatch.setattr(sys, 'platform', 'win32')
        await Agent(calls_then_reply(), capabilities=[CDPBrowser()]).run('hi')
        assert chrome.process.returncode == -9
        assert chrome.killed_groups == []

    async def test_a_connection_lost_before_the_first_tab_fails_at_once(self, chrome: FakeChrome):
        """The wait for the first tab ends with the connection, instead of running out its setup timeout."""
        chrome.close_after_auto_attach = True
        with pytest.raises(ConnectionError, match='Chrome DevTools connection closed'):
            await Agent(calls_then_reply(), capabilities=[CDPBrowser()]).run('hi')
        assert not profile_of(chrome).exists()

    async def test_profile_removal_is_retried(self, chrome: FakeChrome, monkeypatch: pytest.MonkeyPatch):
        """A child process still writing to the profile makes the first removal leave files behind."""
        rmtree = shutil.rmtree
        calls: list[object] = []

        def flaky_rmtree(path: Path, ignore_errors: bool = False) -> None:
            calls.append(path)
            if len(calls) > 1:
                rmtree(path, ignore_errors=ignore_errors)

        monkeypatch.setattr(shutil, 'rmtree', flaky_rmtree)
        await Agent(calls_then_reply(), capabilities=[CDPBrowser()]).run('hi')
        assert len(calls) == 2 and not profile_of(chrome).exists()

    async def test_a_profile_that_cannot_be_removed_is_left(self, chrome: FakeChrome, monkeypatch: pytest.MonkeyPatch):
        """Teardown gives up rather than failing a run that already finished."""
        rmtree = shutil.rmtree

        def keep(path: Path, ignore_errors: bool = False) -> None:
            pass

        monkeypatch.setattr(shutil, 'rmtree', keep)
        monkeypatch.setattr(client_module, 'PROFILE_REMOVAL_ATTEMPTS', 2)
        await Agent(calls_then_reply(), capabilities=[CDPBrowser()]).run('hi')
        profile = profile_of(chrome)
        assert profile.exists()
        rmtree(profile)

    @pytest.mark.parametrize(
        ('platform', 'found', 'expected'),
        [
            ('linux', {'chromium'}, 'chromium'),
            ('darwin', set[str](), None),
            ('win32', set[str](), 'windows'),
        ],
    )
    async def test_chrome_is_found(
        self,
        chrome: FakeChrome,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        platform: str,
        found: set[str],
        expected: str | None,
    ):
        windows_chrome = tmp_path / r'programs\Google\Chrome\Application\chrome.exe'
        windows_chrome.touch()
        monkeypatch.delenv('CHROME_PATH', raising=False)
        monkeypatch.setenv('PROGRAMFILES', str(tmp_path / 'programs'))
        monkeypatch.delenv('PROGRAMFILES(X86)', raising=False)
        monkeypatch.delenv('LOCALAPPDATA', raising=False)
        monkeypatch.setattr(sys, 'platform', platform)

        def which(name: str) -> str | None:
            return f'/usr/bin/{name}' if name in found else None

        monkeypatch.setattr(shutil, 'which', which)
        agent = Agent(calls_then_reply(), capabilities=[CDPBrowser()])
        if expected is None:
            with pytest.raises(FileNotFoundError, match='Chrome not found, set `CHROME_PATH`'):
                await agent.run('hi')
            return
        await agent.run('hi')
        launched = chrome.launches[0][0]
        assert launched == (str(windows_chrome) if expected == 'windows' else '/usr/bin/chromium')

    async def test_a_bare_name_in_the_working_directory_is_not_chrome(
        self, chrome: FakeChrome, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        (tmp_path / 'chromium').touch()
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv('CHROME_PATH', raising=False)
        monkeypatch.setattr(sys, 'platform', 'linux')

        def nowhere(name: str) -> None:
            return None

        monkeypatch.setattr(shutil, 'which', nowhere)
        with pytest.raises(FileNotFoundError, match='Chrome not found'):
            await Agent(calls_then_reply(), capabilities=[CDPBrowser()]).run('hi')

    async def test_chrome_path_from_the_environment(self, chrome: FakeChrome, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv('CHROME_PATH', '/env/chrome')
        await Agent(calls_then_reply(), capabilities=[CDPBrowser()]).run('hi')
        assert chrome.launches[0][0] == '/env/chrome'


class TestDurabilityRejection:
    def test_accepts_innermost_non_durability_capability(self):
        class _Innermost(AbstractCapability[object]):
            def get_ordering(self) -> CapabilityOrdering:
                return CapabilityOrdering(position='innermost')

        Agent(TestModel(), capabilities=[CDPBrowser(), _Innermost()])

    def test_rejects_temporal_durability_at_construction(self):
        pytest.importorskip('temporalio')
        from pydantic_ai.durable_exec.temporal import TemporalDurability

        with pytest.raises(UserError, match='CDPBrowser does not support durable execution'):
            Agent(TestModel(), capabilities=[CDPBrowser(), TemporalDurability()])
