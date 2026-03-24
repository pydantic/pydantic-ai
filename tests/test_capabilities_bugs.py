"""Tests confirming bugs in the Capability abstraction (PR #4640).

Background
----------
PR #4640 introduced a **Capability** system for composable agent behaviors.
Key concepts referenced in these tests:

- **Capability**: A composable unit of agent behavior (e.g. thinking config,
  web search, hooks). Each capability can provide tools, model settings,
  instructions, and lifecycle hooks to the agent.

- **Hooks**: A concrete Capability that lets users register lifecycle callbacks
  via decorators (e.g. `@hooks.on.before_run`, `@hooks.on.after_run`).
  Multiple hooks can be registered on a single `Hooks` instance.

- **CombinedCapability**: When an agent has multiple capabilities, they're
  wrapped in a CombinedCapability that merges their behaviors. It defines the
  ordering contract: `before_*` hooks fire forward (cap1 -> cap2), `after_*`
  hooks fire reversed (cap2 -> cap1), creating a symmetric middleware stack.

- **DynamicToolset**: A toolset whose tools are created by a factory function.
  With `per_run_step=True`, the factory is re-called each agent step, allowing
  tools to change between model requests.

- **ToolManager**: Manages tool execution for a run. Tracks tool retry counts
  by tool name across run steps.

- **HistoryProcessor**: A capability that filters/modifies message history
  before each model request. Multiple processors compose in sequence.

- **for_run()**: Called once per agent run on each capability to create per-run
  state. If it returns `self`, the agent uses cached init-time values. If it
  returns a new instance, the agent re-extracts all configuration.

Naming convention
-----------------
Each test is named: `test_bug<N>_<actual_behavior>__<expected_behavior>`

- The part before `__` describes what ACTUALLY happens (the bug).
- The part after `__` describes what SHOULD happen (the fix).
- Tests PASS by asserting the BUGGY behavior.
- Comments explain what correct behavior would look like.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pytest

from pydantic_ai._run_context import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.capabilities.combined import CombinedCapability
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.usage import RunUsage

pytestmark = [pytest.mark.anyio]


def build_run_context(deps: Any = None, run_step: int = 0) -> RunContext[Any]:
    return RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=run_step,
    )


# ======================================================================
# Bug 1 — Hooks after_* fires forward, should fire reversed
# ======================================================================
#
# DOCUMENTED CONTRACT (docs/hooks.md lines 287-290):
#   "When multiple hooks are registered for the same event (either on
#    the same Hooks instance or across multiple capabilities):
#    * before_* hooks fire in registration/capability order
#    * after_* hooks fire in reverse order"
#
# The docs explicitly say this applies to a SINGLE Hooks instance too.
# CombinedCapability implements this correctly (reversed after_*), and
# wrap_* within Hooks also reverses. But after_* within a single Hooks
# instance iterates FORWARD, violating the documented contract.
#
# Why it matters: if h1 opens a DB connection in before_run and h2
# opens a cache in before_run, the stack contract says h2's after_run
# should close the cache first, then h1's after_run closes the DB.
# With forward ordering, h1's after_run fires first and may try to
# use the cache that h2 hasn't closed yet.
# ======================================================================


async def test_bug1_after_run_fires_forward__should_fire_reversed():
    """Register two before_run and two after_run hooks on one Hooks instance.

    Actual:   before=[h1, h2], after=[h1, h2] (both forward)
    Expected: before=[h1, h2], after=[h2, h1] (symmetric stack)
    """
    order: list[str] = []
    hooks = Hooks[None]()

    @hooks.on.before_run
    async def h1_before(ctx: RunContext[None]) -> None:
        order.append('h1:before')

    @hooks.on.before_run
    async def h2_before(ctx: RunContext[None]) -> None:
        order.append('h2:before')

    @hooks.on.after_run
    async def h1_after(ctx: RunContext[None], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
        order.append('h1:after')
        return result

    @hooks.on.after_run
    async def h2_after(ctx: RunContext[None], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
        order.append('h2:after')
        return result

    agent = Agent(TestModel(), capabilities=[hooks])
    await agent.run('hello')

    before_order = [x for x in order if 'before' in x]
    after_order = [x for x in order if 'after' in x]

    assert before_order == ['h1:before', 'h2:before']

    # BUG: after fires forward (h1 first), should fire reversed (h2 first)
    assert after_order == ['h1:after', 'h2:after']
    # EXPECTED: ['h2:after', 'h1:after']


async def test_bug1_after_model_request_fires_forward__should_fire_reversed():
    """Same bug but for model request hooks.

    before_model_request and after_model_request fire around each LLM call.
    The same forward-instead-of-reversed issue applies.
    """
    order: list[str] = []
    hooks = Hooks[None]()

    @hooks.on.before_model_request
    async def h1_before(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
        order.append('h1:before')
        return request_context

    @hooks.on.before_model_request
    async def h2_before(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
        order.append('h2:before')
        return request_context

    @hooks.on.after_model_request
    async def h1_after(
        ctx: RunContext[None], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        order.append('h1:after')
        return response

    @hooks.on.after_model_request
    async def h2_after(
        ctx: RunContext[None], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        order.append('h2:after')
        return response

    agent = Agent(TestModel(), capabilities=[hooks])
    await agent.run('hello')

    before_order = [x for x in order if 'before' in x]
    after_order = [x for x in order if 'after' in x]

    assert before_order == ['h1:before', 'h2:before']

    # BUG: forward instead of reversed
    assert after_order == ['h1:after', 'h2:after']
    # EXPECTED: ['h2:after', 'h1:after']


async def test_bug1_after_tool_execute_fires_forward__should_fire_reversed():
    """Same bug but for tool execution hooks.

    before/after_tool_execute fire around each tool call. A model that
    calls a tool is needed to trigger these hooks.
    """
    order: list[str] = []
    hooks = Hooks[None]()

    @hooks.on.before_tool_execute
    async def h1_before(ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any) -> Any:
        order.append('h1:before')
        return args

    @hooks.on.before_tool_execute
    async def h2_before(ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any) -> Any:
        order.append('h2:before')
        return args

    @hooks.on.after_tool_execute
    async def h1_after(
        ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, result: Any
    ) -> Any:
        order.append('h1:after')
        return result

    @hooks.on.after_tool_execute
    async def h2_after(
        ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, result: Any
    ) -> Any:
        order.append('h2:after')
        return result

    def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for msg in messages:
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    return ModelResponse(parts=[TextPart(content='done')])
        if info.function_tools:
            return ModelResponse(
                parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='c1')]
            )
        return ModelResponse(parts=[TextPart(content='no tools')])

    agent = Agent(FunctionModel(model_func), capabilities=[hooks])

    @agent.tool_plain
    def my_tool() -> str:
        return 'result'

    await agent.run('call the tool')

    before_order = [x for x in order if 'before' in x]
    after_order = [x for x in order if 'after' in x]

    assert before_order == ['h1:before', 'h2:before']

    # BUG: forward instead of reversed
    assert after_order == ['h1:after', 'h2:after']
    # EXPECTED: ['h2:after', 'h1:after']


async def test_bug1_contrast__combined_capability_correctly_reverses():
    """CONTRAST (not a bug): CombinedCapability reverses after_* correctly.

    When two separate capabilities (not hooks on one Hooks instance) are
    combined, CombinedCapability iterates after_* in reversed order.
    This test confirms the correct behavior to show the Hooks bug is
    specifically about multiple hooks on a SINGLE Hooks instance.
    """
    order: list[str] = []

    @dataclass
    class Cap1(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            order.append('cap1:before')

        async def after_run(self, ctx: RunContext[None], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            order.append('cap1:after')
            return result

    @dataclass
    class Cap2(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            order.append('cap2:before')

        async def after_run(self, ctx: RunContext[None], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            order.append('cap2:after')
            return result

    agent = Agent(TestModel(), capabilities=[Cap1(), Cap2()])
    await agent.run('hello')

    before_order = [x for x in order if 'before' in x]
    after_order = [x for x in order if 'after' in x]

    # CombinedCapability correctly reverses after_run
    assert before_order == ['cap1:before', 'cap2:before']
    assert after_order == ['cap2:after', 'cap1:after']  # reversed — correct!


# ======================================================================
# Bug 2 — DynamicToolset for_run_step() has no error recovery
# ======================================================================
#
# DynamicToolset wraps a factory function that creates a new toolset
# each step. In for_run_step(), it:
#   1. Calls the factory to get a new toolset
#   2. Exits the old toolset (__aexit__)
#   3. Assigns the new toolset
#   4. Enters the new toolset (__aenter__)
#
# If step 1 or 4 raises, there's no rollback. Two failure modes:
#
# (a) Factory raises: exception propagates, old toolset is still
#     referenced but the DynamicToolset is in an ambiguous state.
#     Not catastrophic, but the contract is unclear.
#
# (b) New toolset's __aenter__ raises AFTER old's __aexit__: the old
#     toolset is gone (exited), the new one is broken (never entered),
#     but self._toolset now points to the un-entered new toolset.
#     Any subsequent tool calls will use a toolset that was never
#     properly initialized.
# ======================================================================


@dataclass
class TrackableToolset(AbstractToolset[None]):
    """Minimal toolset that tracks enter/exit lifecycle for testing."""

    _id: str = 'trackable'
    entered: bool = False
    exited: bool = False

    @property
    def id(self) -> str:
        return self._id

    @property
    def label(self) -> str:
        return self._id

    @property
    def tool_name_conflict_hint(self) -> str:
        return ''

    async def __aenter__(self) -> TrackableToolset:
        self.entered = True
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        self.exited = True
        return None

    async def get_tools(self, ctx: RunContext[None]) -> dict[str, Any]:
        return {}

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[None], tool: ToolsetTool[None]
    ) -> Any:
        return None


async def test_bug2a_factory_error_leaves_stale_toolset__should_be_safe_to_retry():
    """When the factory raises, the old toolset stays referenced but the
    DynamicToolset is in an ambiguous state — no cleanup, no rollback.

    Actual:   factory raises, old toolset still assigned, not exited, but
              the DynamicToolset has no way to signal "I'm in a bad state"
    Expected: either rollback cleanly (old toolset stays valid) or mark
              the DynamicToolset as failed so callers know not to use it
    """
    call_count = 0
    first_toolset = TrackableToolset(_id='first')

    def factory(ctx: RunContext[None]) -> AbstractToolset[None]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return first_toolset
        raise RuntimeError('factory failed on 2nd call')

    dt = DynamicToolset[None](toolset_func=factory, per_run_step=True)
    ctx = build_run_context()
    run_dt = await dt.for_run(ctx)

    async with run_dt:
        await run_dt.for_run_step(ctx)
        assert first_toolset.entered
        assert not first_toolset.exited

        # Factory raises on step 2 — old toolset is NOT exited (good)
        # but the DynamicToolset has no way to indicate failure state
        with pytest.raises(RuntimeError, match='factory failed'):
            await run_dt.for_run_step(replace(ctx, run_step=1))

        assert not first_toolset.exited  # old toolset not cleaned up
        assert isinstance(run_dt, DynamicToolset)
        # old toolset still referenced
        assert run_dt._toolset is first_toolset  # pyright: ignore[reportPrivateUsage]


async def test_bug2b_enter_error_after_exit_loses_both__should_rollback_to_old():
    """When the new toolset's __aenter__ fails AFTER the old was exited,
    both toolsets are in a broken state.

    Actual:   old toolset exited, new toolset's __aenter__ raised,
              self._toolset points to the un-entered (broken) new toolset
    Expected: rollback — re-enter old toolset or at least set
              self._toolset = None to signal "no valid toolset"
    """

    @dataclass
    class BrokenEnterToolset(AbstractToolset[None]):
        """Toolset whose __aenter__ always raises."""

        @property
        def id(self) -> str:
            return 'broken'

        @property
        def label(self) -> str:
            return 'broken'

        @property
        def tool_name_conflict_hint(self) -> str:
            return ''

        async def __aenter__(self) -> BrokenEnterToolset:
            raise RuntimeError('__aenter__ failed')

        async def __aexit__(self, *args: Any) -> bool | None:
            return None

        async def get_tools(self, ctx: RunContext[None]) -> dict[str, Any]:
            return {}

        async def call_tool(
            self, name: str, tool_args: dict[str, Any], ctx: RunContext[None], tool: ToolsetTool[None]
        ) -> Any:
            return None

    old_toolset = TrackableToolset(_id='old')
    call_count = 0

    def factory(ctx: RunContext[None]) -> AbstractToolset[None]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return old_toolset
        return BrokenEnterToolset()

    dt = DynamicToolset[None](toolset_func=factory, per_run_step=True)
    ctx = build_run_context()
    run_dt = await dt.for_run(ctx)

    async with run_dt:
        await run_dt.for_run_step(ctx)
        assert old_toolset.entered and not old_toolset.exited

        with pytest.raises(RuntimeError, match='__aenter__ failed'):
            await run_dt.for_run_step(replace(ctx, run_step=1))

        # BUG: old toolset was exited, new one failed to enter
        assert old_toolset.exited
        # self._toolset now points to the broken, never-entered toolset
        assert isinstance(run_dt, DynamicToolset)
        assert run_dt._toolset is not old_toolset  # pyright: ignore[reportPrivateUsage]
        assert not isinstance(run_dt._toolset, TrackableToolset)  # pyright: ignore[reportPrivateUsage]
        # EXPECTED: self._toolset should be None or the old toolset re-entered


# ======================================================================
# Bug 3 — Error handlers chain-replace the original error
# ======================================================================
#
# When multiple on_*_error handlers are registered (either on one Hooks
# instance or across capabilities), they fire in sequence. If handler A
# raises a NEW exception, handler B receives handler A's exception —
# NOT the original error that triggered the chain.
#
# This means the original error context is silently lost. If handler A
# has a bug (accidentally raises), handler B cannot even see what the
# real problem was.
#
# The on_*_error mechanism: when an agent run fails (e.g. model raises,
# tool raises unrecoverable error), on_run_error handlers get a chance
# to recover by returning a result. If a handler can't recover, it
# re-raises the error. Handlers fire in reverse registration order.
# ======================================================================


async def test_bug3_error_handler_sees_transformed_error__should_see_original():
    """Register two on_run_error handlers on a Hooks instance.
    Handler 1 raises a new TypeError. Handler 2 should see the original
    ValueError but instead sees handler 1's TypeError.

    Actual:   handler2 receives TypeError (from handler1)
    Expected: handler2 receives ValueError (the original error)

    Tested directly on the Hooks instance (no agent needed).
    """
    seen_errors: list[str] = []
    hooks = Hooks[None]()

    @hooks.on.run_error
    async def handler1(ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
        seen_errors.append(f'handler1 saw: {type(error).__name__}: {error}')
        raise TypeError('handler1 accidentally raised')

    @hooks.on.run_error
    async def handler2(ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
        seen_errors.append(f'handler2 saw: {type(error).__name__}: {error}')
        raise error

    ctx = build_run_context()

    with pytest.raises(TypeError, match='handler1 accidentally raised'):
        await hooks.on_run_error(ctx, error=ValueError('original problem'))

    assert seen_errors[0] == 'handler1 saw: ValueError: original problem'

    # BUG: handler2 sees TypeError from handler1, not the original ValueError
    assert seen_errors[1] == 'handler2 saw: TypeError: handler1 accidentally raised'
    # EXPECTED: 'handler2 saw: ValueError: original problem'


async def test_bug3_combined_caps_error_chain_replaces__should_preserve_original():
    """Same bug across capabilities in a CombinedCapability.

    CombinedCapability.on_run_error iterates capabilities in REVERSED order.
    If ObservingHandler re-raises, FailingHandler sees the same error.
    But if FailingHandler raises something new, the FINAL error loses
    all connection to the original.

    Actual:   final raised error is RuntimeError('handler failed')
    Expected: original ValueError should be preserved (e.g. as __cause__)

    Tested directly on CombinedCapability (no agent needed).
    """
    seen_errors: list[str] = []

    @dataclass
    class FailingHandler(AbstractCapability[None]):
        async def on_run_error(self, ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
            seen_errors.append(f'FailingHandler saw: {type(error).__name__}')
            raise RuntimeError('handler failed')

    @dataclass
    class ObservingHandler(AbstractCapability[None]):
        async def on_run_error(self, ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
            seen_errors.append(f'ObservingHandler saw: {type(error).__name__}')
            raise error

    # Reversed iteration: [Failing, Observing] -> fires Observing first, then Failing
    combined = CombinedCapability([FailingHandler(), ObservingHandler()])
    ctx = build_run_context()

    with pytest.raises(RuntimeError, match='handler failed'):
        await combined.on_run_error(ctx, error=ValueError('original'))

    assert seen_errors[0] == 'ObservingHandler saw: ValueError'  # sees original
    assert seen_errors[1] == 'FailingHandler saw: ValueError'  # also sees original (Observing re-raised it)
    # But the final RuntimeError has no reference to the original ValueError


# ======================================================================
# Bug 4 — Capability mutating self in for_run() uses stale cached values
# ======================================================================
#
# When an agent is created, it calls get_instructions(), get_model_settings(),
# etc. on each capability and CACHES the results. On each run, it calls
# for_run() on the root capability. If for_run() returns a DIFFERENT object
# (identity check: `run_cap is not effective_cap`), the agent re-extracts
# all values from the new object.
#
# But if for_run() mutates `self` and returns `self`, the identity check
# says "same object, no change" and the agent uses the stale cached values.
# The capability's internal state was changed, but the agent never notices.
#
# This is a contract violation trap: a capability developer might reasonably
# expect that mutating self in for_run() is sufficient, but the identity
# check silently ignores the mutation.
# ======================================================================


async def test_bug4_mutating_self_uses_stale_cache__should_re_extract():
    """A capability mutates its instructions in for_run() but returns self.
    The agent ignores the mutation and sends the old instructions to the model.

    Actual:   model sees 'original instructions' (cached at agent init)
    Expected: model sees 'mutated instructions' (from for_run mutation)
    """
    captured_instructions: list[str | None] = []

    def capture_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured_instructions.append(info.instructions)
        return ModelResponse(parts=[TextPart(content='ok')])

    @dataclass
    class MutatingCap(AbstractCapability[None]):
        instr: str = 'original instructions'

        def get_instructions(self) -> str:
            return self.instr

        async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
            self.instr = 'mutated instructions'
            return self  # returns self -> identity check misses the change

    cap = MutatingCap()
    agent = Agent(FunctionModel(capture_model), capabilities=[cap])
    await agent.run('hello')

    # Capability WAS mutated
    assert cap.instr == 'mutated instructions'

    # BUG: model saw STALE cached instructions
    assert captured_instructions[0] is not None
    assert 'original instructions' in captured_instructions[0]
    assert 'mutated instructions' not in captured_instructions[0]
    # EXPECTED: captured_instructions[0] should contain 'mutated instructions'


async def test_bug4_contrast__returning_new_instance_re_extracts():
    """CONTRAST (not a bug): returning a NEW instance from for_run()
    triggers re-extraction. The model sees the updated instructions.

    This confirms the stale cache only happens when for_run() returns self.
    """
    captured_instructions: list[str | None] = []

    def capture_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured_instructions.append(info.instructions)
        return ModelResponse(parts=[TextPart(content='ok')])

    @dataclass
    class CorrectCap(AbstractCapability[None]):
        instr: str = 'original instructions'

        def get_instructions(self) -> str:
            return self.instr

        async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
            return CorrectCap(instr='updated instructions')  # new instance!

    agent = Agent(FunctionModel(capture_model), capabilities=[CorrectCap()])
    await agent.run('hello')

    assert captured_instructions[0] is not None
    assert 'updated instructions' in captured_instructions[0]  # correctly re-extracted


# ======================================================================
# Bug 5 — Tool retry count persists when DynamicToolset swaps tools
# ======================================================================
#
# ToolManager tracks how many times each tool has failed, by tool NAME.
# When for_run_step() advances to the next step, it carries forward the
# retry count: retries[tool_name] += 1 for each tool that failed.
#
# If a DynamicToolset (per_run_step=True) swaps the tool implementation
# between steps but keeps the same name, the new implementation inherits
# the accumulated retry count from the old implementation. A brand-new
# tool could be born with N retries already counted against it, making
# it hit max_retries faster than expected.
#
# Whether this is a "bug" or a design tradeoff depends on perspective.
# The ToolManager has no way to know the underlying tool changed.
# ======================================================================


async def test_bug5_retries_carry_over_after_tool_swap__should_reset():
    """ToolManager's retry count for a tool name persists across run steps,
    even if the underlying tool implementation was swapped by a DynamicToolset.

    Actual:   after 2 failures across 2 steps, retries['my_tool'] == 2
    Expected: if the tool was swapped, retries should reset to 0

    Tested directly on ToolManager (no agent needed).
    """
    from pydantic_ai._tool_manager import ToolManager

    ts = FunctionToolset[None]()

    @ts.tool_plain(retries=3)
    def my_tool() -> str:
        return 'ok'

    ctx0 = build_run_context(run_step=0)

    async with ts:
        run_ts = await ts.for_run(ctx0)

        async with run_ts:
            step0_ts = await run_ts.for_run_step(ctx0)
            tools = await step0_ts.get_tools(ctx0)

            tm = ToolManager(
                toolset=step0_ts,
                root_capability=CombinedCapability([]),
                ctx=ctx0,
                tools=tools,
                default_max_retries=1,
            )

            # Step 0: my_tool fails
            tm.failed_tools.add('my_tool')

            # Step 1: ToolManager carries over the failure
            tm1 = await tm.for_run_step(replace(ctx0, run_step=1))
            assert tm1.ctx is not None
            assert tm1.ctx.retries['my_tool'] == 1  # carried over

            # Step 1: my_tool fails again
            tm1.failed_tools.add('my_tool')

            # Step 2: retries accumulate further
            tm2 = await tm1.for_run_step(replace(ctx0, run_step=2))
            assert tm2.ctx is not None
            assert tm2.ctx.retries['my_tool'] == 2  # accumulated!

            # BUG: if a DynamicToolset replaced my_tool's implementation
            # between steps, this new tool starts with retries=2 inherited
            # from the old implementation. It only has 1 retry left before
            # hitting max_retries=3.
            # EXPECTED: retries should reset when the tool implementation changes


# ======================================================================
# Bug 6 — History processors can create orphaned tool returns
# ======================================================================
#
# History processors modify the message list before each model request.
# Multiple processors compose in sequence (processor 2 sees processor 1's
# output). There's no validation that the resulting messages are
# semantically consistent.
#
# In the agent message protocol, a ToolCallPart (model asks to call a tool)
# appears in a ModelResponse, and the corresponding ToolReturnPart (tool
# result) appears in the next ModelRequest. If a processor removes the
# ModelResponse containing the ToolCallPart but leaves the ModelRequest
# containing the ToolReturnPart, the model sees a tool return with no
# preceding tool call — which is semantically invalid.
# ======================================================================


async def test_bug6_processor_creates_orphaned_tool_returns__should_warn_or_validate():
    """A history processor removes ModelResponses with tool calls but
    leaves ModelRequests with tool returns. The model sees orphaned
    ToolReturnParts with no corresponding ToolCallParts.

    Actual:   model receives messages with ToolReturnPart but no ToolCallPart
    Expected: system should warn about or prevent semantically invalid
              message sequences after history processing
    """
    messages_seen_by_model: list[list[ModelMessage]] = []
    call_count = 0

    def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        messages_seen_by_model.append(list(messages))
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='c1')])
        return ModelResponse(parts=[TextPart(content='final answer')])

    def remove_tool_call_responses(messages: list[ModelMessage]) -> list[ModelMessage]:
        """Remove ModelResponse messages that contain ToolCallPart.

        This is a naive processor that strips the model's tool call requests
        but doesn't strip the corresponding tool return in the next request.
        """
        filtered: list[ModelMessage] = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                filtered.append(msg)
            else:
                has_tool_call = any(isinstance(p, ToolCallPart) for p in msg.parts)
                if not has_tool_call:
                    filtered.append(msg)
        return filtered

    def passthrough(messages: list[ModelMessage]) -> list[ModelMessage]:
        return messages

    agent = Agent(
        FunctionModel(model_func),
        history_processors=[remove_tool_call_responses, passthrough],
    )

    @agent.tool_plain
    def my_tool() -> str:
        return 'tool result'

    await agent.run('hello')

    # On the 2nd model call, check if the model sees orphaned tool returns
    if len(messages_seen_by_model) >= 2:
        second_call = messages_seen_by_model[1]
        has_tool_return = any(isinstance(part, ToolReturnPart) for msg in second_call for part in msg.parts)
        has_tool_call = any(isinstance(part, ToolCallPart) for msg in second_call for part in msg.parts)

        # BUG: ToolReturnPart present but no ToolCallPart — orphaned
        assert has_tool_return, 'tool return should be present'
        assert not has_tool_call, 'tool call was removed by processor'
        # EXPECTED: system should detect and warn about this inconsistency
