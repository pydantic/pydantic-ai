"""ACP spec-conformance tests, driven over a real in-memory wire (`tests/experimental/acp/_wire.py`).

These tests are organized by spec clause, not by adapter method, and each pins the *protocol*
invariant with an oracle constructed independently of the adapter -- not the adapter's own output
compared back against itself. They run through the SDK's JSON-RPC router and serialization, so they
cover what direct-call tests cannot: a method the router gates before the adapter sees it, and a
frame whose serialized bytes overrun the client's read buffer.

Each test names the spec clause it enforces; the package README's feature sections and
"Cancellation and limitations" summarize the adapter's supported surface.
"""

from __future__ import annotations

import acp
import pytest
from acp import RequestError, schema
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.experimental.acp import InMemorySessionStore, PydanticAIACPAgent
from tests.experimental.acp._wire import WireClient, wire_agent  # pyright: ignore[reportMissingTypeStubs]

pytestmark = pytest.mark.anyio


def _agent(text: str = 'hello') -> PydanticAIACPAgent[None, str]:
    return PydanticAIACPAgent(Agent(TestModel(custom_output_text=text)))


def _method_not_found_code() -> int:
    return RequestError.method_not_found('probe').code


def _invalid_params_code() -> int:
    return RequestError.invalid_params({}).code


class TestVersionNegotiation:
    """initialization.md: the Agent echoes a supported version, else returns its latest."""

    async def test_supported_version_is_echoed_back(self) -> None:
        # Clause (MUST): "If the Agent supports the requested version, it MUST respond with the same
        # version." Oracle: the response echoes the *input*, asserted for every in-range version
        # rather than the literal 1 -- so a future protocol bump keeps the mid-range case covered.
        async with wire_agent(_agent()) as (conn, _client):
            for version in range(1, acp.PROTOCOL_VERSION + 1):
                response = await conn.initialize(protocol_version=version)
                assert response.protocol_version == version

    async def test_unsupported_version_negotiates_down_to_latest(self) -> None:
        # Clause (MUST): "Otherwise, the Agent MUST respond with the latest version it supports."
        async with wire_agent(_agent()) as (conn, _client):
            too_new = await conn.initialize(protocol_version=acp.PROTOCOL_VERSION + 1)
            too_old = await conn.initialize(protocol_version=0)
        assert too_new.protocol_version == acp.PROTOCOL_VERSION
        assert too_old.protocol_version == acp.PROTOCOL_VERSION


class TestCapabilityAdvertisement:
    """initialization.md: the Agent advertises exactly the optional methods it supports."""

    async def test_unsupported_methods_are_advertised_off(self) -> None:
        # Clause (MUST): a capability omitted/absent means the method is unsupported. The adapter
        # rejects list/fork/resume/set_mode/set_config; the advertisement side must match. Oracle:
        # read the advertised capabilities back off the wire, independent of the reject handlers.
        async with wire_agent(_agent()) as (conn, _client):
            caps = (await conn.initialize(protocol_version=1)).agent_capabilities
        assert caps is not None
        session_caps = caps.session_capabilities
        assert session_caps is not None
        assert session_caps.close is not None  # the one session method the adapter supports
        assert session_caps.list is None and session_caps.fork is None and session_caps.resume is None
        assert caps.load_session is False  # no store given
        assert caps.mcp_capabilities == schema.McpCapabilities(http=False, sse=False)

    async def test_no_authentication_is_advertised(self) -> None:
        # authentication.md: an agent advertises auth options in `authMethods`; none here means
        # no-auth. Oracle: the advertised list is empty, asserted directly off the wire.
        async with wire_agent(_agent()) as (conn, _client):
            response = await conn.initialize(protocol_version=1)
        assert response.auth_methods == []

    async def test_load_session_advertised_only_with_a_store(self) -> None:
        adapter = PydanticAIACPAgent(Agent(TestModel()), session_store=InMemorySessionStore())
        async with wire_agent(adapter) as (conn, _client):
            caps = (await conn.initialize(protocol_version=1)).agent_capabilities
        assert caps is not None and caps.load_session is True

    async def test_session_setup_without_models_advertises_no_modes_or_config_options(self) -> None:
        # session-modes.md / session-config-options.md: both are advertised per-session. The adapter
        # supports no modes and only advertises config options when a model switch-set is configured.
        async with wire_agent(_agent()) as (conn, _client):
            await conn.initialize(protocol_version=1)
            session = await conn.new_session(cwd='/ws', mcp_servers=[])
        assert session.modes is None
        assert session.config_options is None


class TestModelConfigRouting:
    """session-config-options.md: model switching is a stable config-option update."""

    async def test_set_config_option_routes_without_unstable_protocol(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()), models=['test'])
        async with wire_agent(adapter, unstable=False) as (conn, _client):
            await conn.initialize(protocol_version=1)
            session = await conn.new_session(cwd='/ws', mcp_servers=[])
            response = await conn.set_config_option(config_id='model', value='test', session_id=session.session_id)
        [option] = response.config_options
        assert isinstance(option, schema.SessionConfigOptionSelect)
        assert option.current_value == 'test'


class TestUnstableMethodRouting:
    """transports/schema: an UNSTABLE method is reachable only with `use_unstable_protocol`."""

    async def test_unstable_close_reaches_the_adapter_when_enabled(self) -> None:
        # session/close is UNSTABLE in the SDK router; with the flag on it must route through to
        # the adapter. (Direct-call tests bypass the router and cannot see this.)
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        async with wire_agent(adapter, unstable=True) as (conn, _client):
            await conn.initialize(protocol_version=1)
            session = await conn.new_session(cwd='/ws', mcp_servers=[])
            closed = await conn.close_session(session_id=session.session_id)
        assert closed == schema.CloseSessionResponse()

    # The SDK router emits a UserWarning before rejecting an unstable method; this suite promotes
    # warnings to errors (so the warning would otherwise surface as a wrapped Internal error in the
    # agent task). Letting it stay a warning lets the real production error code -- method_not_found
    # -- reach the client, which is the contract this test pins.
    @pytest.mark.filterwarnings('ignore::UserWarning')
    async def test_unstable_close_is_gated_off_without_the_flag(self) -> None:
        # The load-bearing reason `run_acp_stdio` passes use_unstable_protocol=True: without it the
        # SDK router rejects close with method_not_found before the adapter.
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()))
        async with wire_agent(adapter, unstable=False) as (conn, _client):
            await conn.initialize(protocol_version=1)
            session = await conn.new_session(cwd='/ws', mcp_servers=[])
            with pytest.raises(RequestError) as exc:
                await conn.close_session(session_id=session.session_id)
        assert exc.value.code == _method_not_found_code()


class TestErrorCodes:
    """The JSON-RPC error *code* distinguishes unsupported from malformed -- not just that it raised."""

    async def test_unsupported_method_uses_method_not_found(self) -> None:
        async with wire_agent(_agent()) as (conn, _client):
            await conn.initialize(protocol_version=1)
            session = await conn.new_session(cwd='/ws', mcp_servers=[])
            with pytest.raises(RequestError) as exc:
                await conn.set_session_mode(mode_id='whatever', session_id=session.session_id)
        assert exc.value.code == _method_not_found_code()

    async def test_unknown_model_uses_invalid_params(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()), models=['test'])
        async with wire_agent(adapter) as (conn, _client):
            await conn.initialize(protocol_version=1)
            session = await conn.new_session(cwd='/ws', mcp_servers=[])
            with pytest.raises(RequestError) as exc:
                # The model config option exists here, so a bad model id is malformed input.
                await conn.set_config_option(config_id='model', value='not-a-model', session_id=session.session_id)
        assert exc.value.code == _invalid_params_code()

    async def test_unknown_config_option_uses_invalid_params(self) -> None:
        adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(Agent(TestModel()), models=['test'])
        async with wire_agent(adapter) as (conn, _client):
            await conn.initialize(protocol_version=1)
            session = await conn.new_session(cwd='/ws', mcp_servers=[])
            with pytest.raises(RequestError) as exc:
                await conn.set_config_option(config_id='theme', value='dark', session_id=session.session_id)
        assert exc.value.code == _invalid_params_code()


class TestSessionLoadReplay:
    """session-setup.md (MUST): `session/load` replays the *entire* conversation, user turns included."""

    async def test_load_replays_the_user_turn_over_the_wire(self) -> None:
        store = InMemorySessionStore()
        adapter = PydanticAIACPAgent(Agent(TestModel(custom_output_text='hi there')), session_store=store)
        async with wire_agent(adapter) as (conn, _client):
            await conn.initialize(protocol_version=1)
            session = await conn.new_session(cwd='/ws', mcp_servers=[])
            await conn.prompt(prompt=[acp.text_block('remember this')], session_id=session.session_id)

        # Reopen on a fresh connection, as an editor would after a restart.
        reopened = WireClient()
        async with wire_agent(adapter, reopened) as (conn, _client):
            await conn.initialize(protocol_version=1)
            await conn.load_session(cwd='/ws', session_id=session.session_id, mcp_servers=[])

        # The replay must begin with the user's own message -- the half a transcript-only replay
        # would silently drop. Oracle built by hand from the known prompt, not from adapter output.
        kinds = [getattr(u, 'session_update', '') for u in reopened.updates]
        assert kinds[0] == 'user_message_chunk'
        first = reopened.updates[0]
        assert getattr(getattr(first, 'content', None), 'text', '') == 'remember this'
        # The agent's reply is replayed too (and `texts()` skips the leading user_message_chunk).
        assert 'agent_message_chunk' in kinds
        assert reopened.texts() == 'hi there'


class TestStreamedFrameBytes:
    """transports.md: a streamed frame must fit the client's read buffer -- bound by bytes, not chars."""

    async def test_large_non_ascii_output_reassembles_intact(self) -> None:
        # The client StreamReader keeps asyncio's default 64 KiB line limit. Under `ensure_ascii`
        # each emoji serializes to a 12-byte surrogate-pair escape, so a char-count chunker would
        # emit ~98 KiB frames and the read would overrun. Oracle: the client reassembles the exact
        # payload -- only possible if every frame stayed under the buffer.
        payload = '\U0001f600' * 20_000  # 20k astral chars -> ~240 KiB of escaped JSON bytes
        async with wire_agent(_agent(payload)) as (conn, client):
            await conn.initialize(protocol_version=1)
            session = await conn.new_session(cwd='/ws', mcp_servers=[])
            response = await conn.prompt(prompt=[acp.text_block('go')], session_id=session.session_id)
        assert response.stop_reason == 'end_turn'
        assert client.texts() == payload
