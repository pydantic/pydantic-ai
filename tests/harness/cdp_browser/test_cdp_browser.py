"""Tests for the `CDPBrowser` capability.

Chrome is faked throughout: `FakeChrome` answers the DevTools Protocol over a real in-process websocket, and a fake
process stands in for the browser `launch_chrome` starts, so no Chrome binary runs. What a real Chrome does with
these commands is checked by the hand-run `scripts/cdp_browser_smoke.py`.

None of these is a VCR test: `CDPBrowser` drives Chrome over a websocket rather than a model provider's HTTP API, so
there is no provider request for a cassette to record. The agents run on `FunctionModel`, which scripts the tool calls.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import sys
from collections.abc import AsyncIterator, Callable
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
from pydantic_ai_harness import CDPBrowser, _browser_egress as egress_module, _browser_proxy as proxy_module
from pydantic_ai_harness.cdp_browser import (
    DEFAULT_ALLOWED_CDP_METHODS,
    CDPBrowserToolset,
    EgressPolicy,
    EgressRequest,
    _client as client_module,
    _toolset as toolset_module,
)
from tests._ports import reserve_loopback_port
from tests._timeouts import READINESS_WAIT_TIMEOUT
from tests.conftest import no_leaked_tasks
from tests.harness._socks_client import (
    PUBLIC_ADDRESS,
    EchoServer,
    dial_public_on_loopback,
    ipv4,
    ipv6,
    name,
    serve_echo,
    socks_connect,
    socks_handshake,
)

from ._fake_chrome import PAGE_SESSION, FakeChrome, is_object_dict, pause_expression


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

    async def test_chrome_launches_with_the_guard_flags(self, chrome: FakeChrome):
        await Agent(calls_then_reply(), capabilities=[CDPBrowser()]).run('hi')
        assert '--disable-features=FedCm' in chrome.launches[0]

    async def test_disallowed_commands_and_urls_are_refused(self, chrome: FakeChrome):
        agent = Agent(
            calls_then_reply(
                {'method': 'Browser.close'},
                {'method': 'Page.navigate', 'params': {'url': 'file:///etc/passwd'}},
                # a success between refusals, so they stay under the three retries allowed in a row
                evaluate('document.title'),
                {'method': 'Page.navigate', 'params': {'url': 'http://169.254.169.254/latest/meta-data/'}},
                {'method': 'Page.navigate', 'params': {'url': 'https://elsewhere.org/'}},
                evaluate('document.title'),
                {'method': 'Page.navigate', 'params': {'url': 'https://[example.com/'}},
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
                {'result': {'type': 'string', 'value': 'Example Domain'}},
                "'https://[example.com/' is not a valid URL",
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
            "Hosts that top-level navigation and page scripts' data requests (`fetch`, XHR, EventSource, `sendBeacon`, "
            'WebSocket) may reach: example.com (and their subdomains) (private/internal addresses blocked). '
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
        assert seen[0] is not None
        assert seen[0].splitlines()[-1] == snapshot(
            "Hosts that top-level navigation and page scripts' data requests (`fetch`, XHR, EventSource, `sendBeacon`, "
            'WebSocket) may reach: example.com, other.org (and their subdomains) (private/internal addresses blocked). '
            'A refused request fails with `net::ERR_BLOCKED_BY_CLIENT`.'
        )

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

    async def test_a_timeout_must_be_positive(self, chrome: FakeChrome):
        """A deadline of zero would stop a page script that was never stuck."""
        result = await Agent(
            calls_then_reply(evaluate('document.title', timeout=0), evaluate('document.title')),
            capabilities=[CDPBrowser()],
        ).run('go')
        assert tool_returns(result.all_messages()) == snapshot(
            [
                [{'type': 'greater_than', 'loc': ['timeout'], 'msg': 'Input should be greater than 0', 'input': 0}],
                {'result': {'type': 'string', 'value': 'Example Domain'}},
            ]
        )
        assert commands(chrome, PAGE_SESSION).count('Runtime.evaluate') == 1
        assert 'Runtime.terminateExecution' not in commands(chrome, PAGE_SESSION)

    async def test_an_event_that_never_arrives_times_out(self, chrome: FakeChrome):
        """An abandoned waiter is dropped by the next event, or failed with the connection when none comes."""
        navigate: dict[str, object] = {'method': 'Page.navigate', 'params': {'url': 'https://example.com'}}
        stopped: dict[str, object] = {**navigate, 'wait_for_event': 'Page.frameStoppedLoading', 'timeout': 0.05}
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

    async def test_screenshots_are_image_content(self, chrome: FakeChrome):
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

    @pytest.mark.parametrize(
        ('extra_bytes', 'expected'),
        [
            pytest.param(0, 'Screenshot attached (image/png, 5000000 bytes).', id='at-the-limit'),
            pytest.param(
                1,
                'Error: screenshot is 5000001 bytes, over the 5000000 byte image limit; '
                "capture a smaller `clip`, or pass `format: 'jpeg'` with a lower `quality`.",
                id='refused-after-decoding',
            ),
            pytest.param(
                3,
                'Error: screenshot is over the 5000000 byte image limit; '
                "capture a smaller `clip`, or pass `format: 'jpeg'` with a lower `quality`.",
                id='refused-before-decoding',
            ),
        ],
    )
    async def test_screenshot_size_limit(self, chrome: FakeChrome, extra_bytes: int, expected: str):
        """One byte over the limit still has base64 no longer than the limit's, so only the decoded size refuses it."""
        chrome.screenshot = bytes(toolset_module.MAX_SCREENSHOT_BYTES + extra_bytes)
        result = await Agent(calls_then_reply({'method': 'Page.captureScreenshot'}), capabilities=[CDPBrowser()]).run(
            'go'
        )
        assert tool_returns(result.all_messages()) == [expected]

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

    @pytest.mark.parametrize(
        ('allowed_domains', 'block_private_addresses'),
        [
            pytest.param(['example.com'], True, id='allowed_domains'),
            pytest.param(None, False, id='block_private_addresses'),
        ],
    )
    def test_policy_excludes_the_shorthands(self, allowed_domains: list[str] | None, block_private_addresses: bool):
        with pytest.raises(UserError, match='Pass either policy or the allowed_domains/block_private_addresses'):
            CDPBrowser(
                policy=EgressPolicy(),
                allowed_domains=allowed_domains,
                block_private_addresses=block_private_addresses,
            )

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

    async def test_a_connection_lost_before_the_first_tab_fails_at_once(
        self, chrome: FakeChrome, monkeypatch: pytest.MonkeyPatch
    ):
        """The wait for the first tab ends with the connection, instead of running out its setup timeout."""
        monkeypatch.setattr(client_module, 'SHUTDOWN_GRACE_SECONDS', 0.01)
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
            # the macOS candidate is an absolute path, so a Chrome installed on the host would be found
            def not_a_file(self: Path) -> bool:
                return False

            monkeypatch.setattr(Path, 'is_file', not_a_file)
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


def proxy_server(chrome: FakeChrome) -> str:
    return next(arg for arg in chrome.launches[-1] if arg.startswith('--proxy-server='))


def proxy_port(chrome: FakeChrome) -> int:
    """The port of the proxy Chrome sends WebSockets to, its `socks` entry."""
    return int(proxy_server(chrome).rsplit(':', 1)[1])


def http_proxy_port(chrome: FakeChrome) -> int:
    """The port of the proxy Chrome sends `http` and `https` to."""
    return int(proxy_server(chrome).split(';')[0].rsplit(':', 1)[1])


def small_receive_buffer() -> socket.socket:
    """A socket whose peer runs out of room as soon as its owner stops reading."""
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    sock.setblocking(False)
    return sock


def stalling_listener() -> socket.socket:
    listener = small_receive_buffer()
    listener.bind(('127.0.0.1', 0))
    listener.listen()
    return listener


class AllowsEverything(EgressPolicy):
    """A policy override, whose hosts are still resolved so the proxy has addresses to connect to."""

    def refuse(self, request: EgressRequest) -> str | None:
        return None


@pytest.fixture
async def echo_server() -> AsyncIterator[EchoServer]:
    async with serve_echo() as server:
        yield server


def private_address(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer every lookup with loopback."""

    async def loopback(host: str) -> tuple[str, ...]:
        return ('127.0.0.1',)

    monkeypatch.setattr(egress_module, 'getaddrinfo', loopback)


async def guard_verdict(toolset: CDPBrowserToolset[None], url: str, resource_type: str = 'Document') -> object:
    """Have the fake page request `url`, and return what the egress guard answered."""
    expression = pause_expression(url, resource_type=resource_type)
    result = await toolset.browser_cdp('Runtime.evaluate', {'expression': expression, 'returnByValue': True})
    assert is_object_dict(result)
    value = result['result']
    assert is_object_dict(value)
    return value['value']


async def socks_reply(port: int, destination: bytes) -> bytes:
    """The proxy's answer to a CONNECT, with the connection closed again."""
    _, writer, reply = await socks_connect(port, destination)
    writer.close()
    await writer.wait_closed()
    return reply[:2]


class TestConnectionGuard:
    """With the private-address block on, Chrome sends `http` and `https` to a proxy that checks what it dials."""

    @pytest.mark.parametrize('block_private_addresses', [True, False], ids=['block', 'no-block'])
    async def test_chrome_is_launched_with_its_connections_on_the_proxies(
        self, chrome: FakeChrome, block_private_addresses: bool
    ):
        async with CDPBrowserToolset[None](policy=EgressPolicy(block_private_addresses=block_private_addresses)):
            socket = f'socks5://127.0.0.1:{proxy_port(chrome)}'
            http = f'socks5://127.0.0.1:{http_proxy_port(chrome)}' if block_private_addresses else 'direct://'
            assert http != socket
            assert [arg for arg in chrome.launches[-1] if 'proxy' in arg] == [
                f'--proxy-server=http={http};https={http};socks={socket}',
                '--proxy-bypass-list=<-loopback>',
            ]

    @pytest.mark.parametrize(
        ('address', 'host'),
        [
            pytest.param(ipv4, '127.0.0.1', id='ipv4'),
            pytest.param(ipv6, '::1', id='ipv6'),
            pytest.param(name, 'private.example', id='name-resolving-to-a-private-address'),
        ],
    )
    async def test_a_private_destination_is_refused_at_dial_time(
        self,
        chrome: FakeChrome,
        echo_server: EchoServer,
        monkeypatch: pytest.MonkeyPatch,
        address: Callable[[str, int], bytes],
        host: str,
    ):
        private_address(monkeypatch)
        async with CDPBrowserToolset[None]():
            reply = await socks_reply(http_proxy_port(chrome), address(host, echo_server.port))
        assert reply == b'\x05\x02'
        assert echo_server.connections == []

    @pytest.mark.parametrize(
        ('ttl', 'reply', 'dialed'),
        [
            pytest.param(30.0, b'\x05\x00', [PUBLIC_ADDRESS], id='within-the-cache'),
            pytest.param(0.0, b'\x05\x02', [], id='looked-up-again'),
        ],
    )
    async def test_a_name_that_rebinds_reaches_only_an_address_the_policy_checked(
        self,
        chrome: FakeChrome,
        echo_server: EchoServer,
        monkeypatch: pytest.MonkeyPatch,
        ttl: float,
        reply: bytes,
        dialed: list[str],
    ):
        """The guard's lookup answers with a public address and every later one with loopback, as a rebinding name can.

        While the guard's answer is cached, the proxy dials the address the guard checked. Once it has expired, the
        proxy looks the name up again and refuses the private answer it gets.
        """
        answers = iter([(PUBLIC_ADDRESS,)])

        async def rebinding(host: str) -> tuple[str, ...]:
            return next(answers, ('127.0.0.1',))

        monkeypatch.setattr(egress_module, 'getaddrinfo', rebinding)
        monkeypatch.setattr(egress_module, 'RESOLUTION_TTL_SECONDS', ttl)
        connections = dial_public_on_loopback(monkeypatch)
        async with CDPBrowserToolset[None]() as toolset:
            verdict = await guard_verdict(toolset, f'http://rebind.example:{echo_server.port}/', 'Fetch')
            got = await socks_reply(http_proxy_port(chrome), name('rebind.example', echo_server.port))
        assert (verdict, got, connections) == ('continue', reply, dialed)

    async def test_a_passive_subresource_of_an_allowlisted_page_still_loads(
        self, chrome: FakeChrome, echo_server: EchoServer, monkeypatch: pytest.MonkeyPatch
    ):
        """The allowlist leaves an image on another host alone, and the proxy applies no domain rule to its connection.

        A `fetch` to that host is refused by the guard, and a WebSocket to it by the socket proxy, which applies the
        whole policy.
        """
        connections = dial_public_on_loopback(monkeypatch)
        image = f'http://cdn.other.test:{echo_server.port}/logo.png'
        async with CDPBrowserToolset[None](policy=EgressPolicy(allowed_domains=['allowed.test'])) as toolset:
            verdicts = [await guard_verdict(toolset, image, 'Image'), await guard_verdict(toolset, image, 'Fetch')]
            reader, writer, reply = await socks_connect(
                http_proxy_port(chrome), name('cdn.other.test', echo_server.port)
            )
            writer.write(b'ping')
            await writer.drain()
            assert await reader.readexactly(4) == b'ping'
            writer.close()
            await writer.wait_closed()
            socket_reply = await socks_reply(proxy_port(chrome), name('cdn.other.test', echo_server.port))
        assert verdicts == ['continue', 'fail']
        assert (reply[:2], socket_reply) == (b'\x05\x00', b'\x05\x02')
        assert connections == [PUBLIC_ADDRESS]

    async def test_an_override_that_admits_a_private_address_reaches_it_only_over_a_websocket(
        self, chrome: FakeChrome, echo_server: EchoServer, monkeypatch: pytest.MonkeyPatch
    ):
        """The address block on `http` and `https` connections reads no override; the socket proxy asks the policy."""
        private_address(monkeypatch)
        async with CDPBrowserToolset[None](policy=AllowsEverything()) as toolset:
            verdict = await guard_verdict(toolset, f'http://intranet.example:{echo_server.port}/')
            http_reply = await socks_reply(http_proxy_port(chrome), name('intranet.example', echo_server.port))
            socket_reply = await socks_reply(proxy_port(chrome), name('intranet.example', echo_server.port))
        assert (verdict, http_reply, socket_reply) == ('continue', b'\x05\x02', b'\x05\x00')

    async def test_a_kind_resolved_kinds_leaves_out_is_still_checked_at_dial_time(
        self, chrome: FakeChrome, echo_server: EchoServer, monkeypatch: pytest.MonkeyPatch
    ):
        """The guard lets an image through without a lookup, and the proxy still resolves the name it dials."""
        private_address(monkeypatch)
        policy = EgressPolicy(resolved_kinds=frozenset({'navigation', 'subframe', 'data'}))
        async with CDPBrowserToolset[None](policy=policy) as toolset:
            verdict = await guard_verdict(toolset, f'http://private.example:{echo_server.port}/logo.png', 'Image')
            reply = await socks_reply(http_proxy_port(chrome), name('private.example', echo_server.port))
        assert (verdict, reply) == ('continue', b'\x05\x02')
        assert echo_server.connections == []


class TestWebSocketGuard:
    """Chrome sends every WebSocket to the proxy; these tests speak to it directly, as Chrome does."""

    async def test_an_allowed_socket_is_relayed(self, chrome: FakeChrome, echo_server: EchoServer):
        async with CDPBrowserToolset[None](policy=EgressPolicy(block_private_addresses=False)):
            reader, writer, reply = await socks_connect(proxy_port(chrome), ipv4('127.0.0.1', echo_server.port))
            assert reply[:2] == b'\x05\x00'
            writer.write(b'ping')
            await writer.drain()
            assert await reader.readexactly(4) == b'ping'
            writer.close()
            await writer.wait_closed()

    @pytest.mark.parametrize(
        ('address', 'host'),
        [
            pytest.param(ipv4, '127.0.0.1', id='ipv4'),
            pytest.param(ipv6, '::1', id='ipv6'),
            pytest.param(name, 'private.example', id='name-resolving-to-a-private-address'),
            pytest.param(name, 'a/b', id='name-that-does-not-parse-back'),
        ],
    )
    async def test_a_refused_socket_never_connects(
        self,
        chrome: FakeChrome,
        echo_server: EchoServer,
        monkeypatch: pytest.MonkeyPatch,
        address: Callable[[str, int], bytes],
        host: str,
    ):
        async def private_address(host: str) -> tuple[str, ...]:
            return ('127.0.0.1',)

        monkeypatch.setattr(egress_module, 'getaddrinfo', private_address)
        async with CDPBrowserToolset[None]():
            _, writer, reply = await socks_connect(proxy_port(chrome), address(host, echo_server.port))
            writer.close()
            await writer.wait_closed()
        assert reply[:2] == b'\x05\x02'
        assert echo_server.connections == []

    async def test_a_name_connects_to_the_addresses_the_policy_checked(
        self, chrome: FakeChrome, echo_server: EchoServer, monkeypatch: pytest.MonkeyPatch
    ):
        """The first address has nothing listening, so the relay moves on to the next one."""

        async def two_addresses(host: str) -> tuple[str, ...]:
            return ('127.0.0.2', '127.0.0.1')

        monkeypatch.setattr(egress_module, 'getaddrinfo', two_addresses)
        async with CDPBrowserToolset[None](policy=AllowsEverything()):
            reader, writer, reply = await socks_connect(proxy_port(chrome), name('pinned.example', echo_server.port))
            assert reply[:2] == b'\x05\x00'
            writer.write(b'ping')
            await writer.drain()
            assert await reader.readexactly(4) == b'ping'
            writer.close()
            await writer.wait_closed()

    async def test_a_name_that_does_not_resolve_is_refused(
        self, chrome: FakeChrome, echo_server: EchoServer, monkeypatch: pytest.MonkeyPatch
    ):
        async def no_answer(host: str) -> tuple[str, ...]:
            raise OSError('no such host')

        monkeypatch.setattr(egress_module, 'getaddrinfo', no_answer)
        async with CDPBrowserToolset[None](policy=AllowsEverything()):
            _, writer, reply = await socks_connect(proxy_port(chrome), name('gone.example', echo_server.port))
            writer.close()
            await writer.wait_closed()
        assert reply[:2] == b'\x05\x02'

    @pytest.mark.parametrize(
        ('greeting', 'command', 'destination'),
        [
            pytest.param(b'\x04\x01\x00', 1, None, id='not-socks5'),
            pytest.param(b'\x05\x01\x02', 1, None, id='no-unauthenticated-method'),
            pytest.param(b'\x05\x01\x00', 2, None, id='not-connect'),
            pytest.param(b'\x05\x01\x00', 1, b'\x09', id='unknown-address-type'),
            pytest.param(b'\x05\x01\x00', 1, b'\x03\x02\xff\xff\x00\x50', id='name-not-ascii'),
        ],
    )
    async def test_a_malformed_request_is_closed_unrelayed(
        self, chrome: FakeChrome, echo_server: EchoServer, greeting: bytes, command: int, destination: bytes | None
    ):
        async with CDPBrowserToolset[None](policy=EgressPolicy(block_private_addresses=False)):
            _, writer, reply = await socks_connect(
                proxy_port(chrome),
                destination or ipv4('127.0.0.1', echo_server.port),
                greeting=greeting,
                command=command,
            )
            writer.close()
            await writer.wait_closed()
        assert reply == b''
        assert echo_server.connections == []

    async def test_a_destination_that_refuses_closes_the_socket(self, chrome: FakeChrome):
        async with CDPBrowserToolset[None](policy=EgressPolicy(block_private_addresses=False)):
            _, writer, reply = await socks_connect(proxy_port(chrome), ipv4('127.0.0.1', reserve_loopback_port()))
            writer.close()
            await writer.wait_closed()
        assert reply == b''

    async def test_a_host_holding_a_nul_is_refused_before_the_policy_and_the_lookup(
        self, chrome: FakeChrome, echo_server: EchoServer, monkeypatch: pytest.MonkeyPatch
    ):
        class NeverAsked(EgressPolicy):
            def refuse(self, request: EgressRequest) -> str | None:  # pragma: no cover
                raise AssertionError(request.url)

        async def no_lookup(host: str) -> tuple[str, ...]:  # pragma: no cover
            raise AssertionError(host)

        monkeypatch.setattr(egress_module, 'getaddrinfo', no_lookup)
        async with CDPBrowserToolset[None](policy=NeverAsked(allowed_domains=['allowed.test'])):
            destination = name('192.0.2.1\0.allowed.test', echo_server.port)
            _, writer, reply = await socks_connect(proxy_port(chrome), destination)
            writer.close()
            await writer.wait_closed()
        assert reply[:2] == b'\x05\x02'
        assert echo_server.connections == []

    async def test_a_peer_that_stops_reading_cannot_hold_a_socket_open(
        self, chrome: FakeChrome, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(proxy_module, 'CLOSE_GRACE_SECONDS', 0)
        stalled: set[asyncio.StreamWriter] = set()
        both_sides_stalled = anyio.Event()
        real_drain = asyncio.StreamWriter.drain

        async def noting_drain(self: asyncio.StreamWriter) -> None:
            if self.transport.get_write_buffer_size() > self.transport.get_write_buffer_limits()[1]:
                stalled.add(self)
                if len(stalled) == 2:
                    both_sides_stalled.set()
            await real_drain(self)

        monkeypatch.setattr(asyncio.StreamWriter, 'drain', noting_drain)
        unread = bytes(4 * 1024 * 1024)
        peers: list[asyncio.StreamWriter] = []

        def stop_reading(_: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            peers.append(writer)
            writer.write(unread)

        server = await asyncio.start_server(stop_reading, sock=stalling_listener())
        async with CDPBrowserToolset[None](policy=EgressPolicy(block_private_addresses=False)):
            client = small_receive_buffer()
            await asyncio.get_running_loop().sock_connect(client, ('127.0.0.1', proxy_port(chrome)))
            reader, writer = await asyncio.open_connection(sock=client)
            destination = ipv4('127.0.0.1', server.sockets[0].getsockname()[1])
            assert (await socks_handshake(reader, writer, destination))[:2] == b'\x05\x00'
            writer.write(unread)
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await both_sides_stalled.wait()
        assert [held.transport.get_write_buffer_size() for held in stalled] == [0, 0]
        for held in (writer, *peers):
            held.transport.abort()
        server.close()
        await server.wait_closed()

    async def test_a_closing_socket_still_delivers_what_it_holds(
        self, chrome: FakeChrome, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(proxy_module, 'CLOSE_GRACE_SECONDS', READINESS_WAIT_TIMEOUT)
        unsent_at_close = anyio.Event()
        real_close = asyncio.StreamWriter.close

        def noting_close(self: asyncio.StreamWriter) -> None:
            if self.transport.get_write_buffer_size():
                unsent_at_close.set()
            real_close(self)

        monkeypatch.setattr(asyncio.StreamWriter, 'close', noting_close)
        real_open_connection = asyncio.open_connection

        async def small_send_buffer(host: str, port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
            reader, writer = await real_open_connection(host, port)
            writer.get_extra_info('socket').setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
            return reader, writer

        monkeypatch.setattr(asyncio, 'open_connection', small_send_buffer)
        payload = bytes(60_000)
        received = bytearray()
        delivered = anyio.Event()

        async def read_after_close(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await unsent_at_close.wait()
            received.extend(await reader.read())
            writer.close()
            delivered.set()

        server = await asyncio.start_server(read_after_close, sock=stalling_listener())
        async with CDPBrowserToolset[None](policy=EgressPolicy(block_private_addresses=False)):
            destination = ipv4('127.0.0.1', server.sockets[0].getsockname()[1])
            _, writer, reply = await socks_connect(proxy_port(chrome), destination)
            assert reply[:2] == b'\x05\x00'
            writer.write(payload)
            writer.write_eof()
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await delivered.wait()
            writer.close()
            await writer.wait_closed()
        assert len(received) == len(payload)
        server.close()
        await server.wait_closed()

    async def test_a_connection_landing_as_the_run_ends_gets_no_relay(
        self, chrome: FakeChrome, echo_server: EchoServer, monkeypatch: pytest.MonkeyPatch
    ):
        callbacks: list[Callable[[asyncio.StreamReader, asyncio.StreamWriter], None]] = []
        real_start_server = asyncio.start_server

        async def noting_start_server(
            callback: Callable[[asyncio.StreamReader, asyncio.StreamWriter], None], host: str, port: int
        ) -> asyncio.Server:
            callbacks.append(callback)
            return await real_start_server(callback, host, port)

        monkeypatch.setattr(asyncio, 'start_server', noting_start_server)
        async with CDPBrowserToolset[None]():
            pass
        reader, writer = await asyncio.open_connection('127.0.0.1', echo_server.port)
        with no_leaked_tasks():
            callbacks[-1](reader, writer)
        assert writer.is_closing()
        await writer.wait_closed()

    async def test_a_relay_cancelled_before_its_first_step_closes_its_connection(
        self, chrome: FakeChrome, echo_server: EchoServer, monkeypatch: pytest.MonkeyPatch
    ):
        callbacks: list[Callable[[asyncio.StreamReader, asyncio.StreamWriter], None]] = []
        real_start_server = asyncio.start_server

        async def noting_start_server(
            callback: Callable[[asyncio.StreamReader, asyncio.StreamWriter], None], host: str, port: int
        ) -> asyncio.Server:
            callbacks.append(callback)
            return await real_start_server(callback, host, port)

        monkeypatch.setattr(asyncio, 'start_server', noting_start_server)
        async with CDPBrowserToolset[None]():
            reader, writer = await asyncio.open_connection('127.0.0.1', echo_server.port)
            running = asyncio.all_tasks()
            callbacks[-1](reader, writer)
            [relay] = asyncio.all_tasks() - running
            relay.cancel()
            with pytest.raises(asyncio.CancelledError):
                await relay
            closed_by_the_proxy = writer.is_closing()
            writer.transport.abort()
        await writer.wait_closed()
        assert closed_by_the_proxy

    async def test_the_end_of_the_run_closes_open_sockets(self, chrome: FakeChrome, echo_server: EchoServer):
        with no_leaked_tasks():
            async with CDPBrowserToolset[None](policy=EgressPolicy(block_private_addresses=False)):
                reader, writer, reply = await socks_connect(proxy_port(chrome), ipv4('127.0.0.1', echo_server.port))
                assert reply[:2] == b'\x05\x00'
            assert await reader.read() == b''
            writer.close()
            await writer.wait_closed()


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
