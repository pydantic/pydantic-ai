"""Enforcement meta-tests for standards the maintainers uphold by review alone.

Each test here pins a convention that reviewers repeatedly ask contributors to follow, so that a
violation shows up as a failing test (and, where relevant, an explicit allowlist diff)
instead of relying on a human to catch it in review.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunContext
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.agent.wrapper import WrapperAgent
from pydantic_ai.models.test import TestModel

from .conftest import try_import

with try_import() as temporal_imports:
    from pydantic_ai.durable_exec.temporal import TemporalAgent  # pyright: ignore[reportDeprecated]

with try_import() as dbos_imports:
    from pydantic_ai.durable_exec.dbos import DBOSAgent  # pyright: ignore[reportDeprecated]

with try_import() as prefect_imports:
    from pydantic_ai.durable_exec.prefect import PrefectAgent  # pyright: ignore[reportDeprecated]


# Frozen snapshot of public dataclasses that predate the keyword-only convention and whose
# constructor takes two or more positional parameters. It grandfathers today's offenders so the
# test below only gates NEW public dataclasses: a new one must use `_: KW_ONLY` (or
# `kw_only=True`) so that adding a field later can't break positional callers.
#
# "Public" here means the defining module path and the class name carry no leading underscore, and
# the class is not a `StreamedResponse` implementation -- see `kw_only_walker.py`, which owns both
# rules and explains why. Classes that hand-write a keyword-only `__init__` under
# `@dataclass(init=False)` are already safe and are deliberately absent.
#
# This list only ever shrinks. Converting an entry to keyword-only breaks positional callers, so
# the drain path is a major version: drop entries here in the same change that flips them,
# alongside the other `TODO(v3)` removals. Do NOT add an entry without maintainer sign-off -- a new
# entry means a new public dataclass is shipping with a fragile positional signature on purpose.
_KW_ONLY_ALLOWLIST: frozenset[str] = frozenset(
    {
        'pydantic_ai.capabilities.abstract.CapabilityOrdering',
        'pydantic_ai.capabilities.prefix_tools.PrefixTools',
        'pydantic_ai.common_tools.exa.ExaFindSimilarTool',
        'pydantic_ai.common_tools.exa.ExaSearchTool',
        'pydantic_ai.concurrency.ConcurrencyLimit',
        'pydantic_ai.embeddings.instrumented.InstrumentedEmbeddingModel',
        'pydantic_ai.function_signature.GenericTypeExpr',
        'pydantic_ai.function_signature.LiteralTypeExpr',
        'pydantic_ai.function_signature.SimpleTypeExpr',
        'pydantic_ai.function_signature.UnionTypeExpr',
        'pydantic_ai.messages.BaseToolCallPart',
        'pydantic_ai.messages.BaseToolReturnPart',
        'pydantic_ai.messages.CachePoint',
        'pydantic_ai.messages.FunctionToolCallEvent',
        'pydantic_ai.messages.NativeToolCallPart',
        'pydantic_ai.messages.NativeToolReturnPart',
        'pydantic_ai.messages.OutputToolCallEvent',
        'pydantic_ai.messages.OutputToolResultEvent',
        'pydantic_ai.messages.ToolCallPart',
        'pydantic_ai.messages.ToolReturnPart',
        'pydantic_ai.messages.UploadedFile',
        'pydantic_ai.models.concurrency.ConcurrencyLimitedModel',
        # Variadic by design -- `FallbackModel(m1, m2, m3)` is the documented call. Everything after
        # `*fallback_models` is already keyword-only, so no field can ever be inserted positionally
        # and the hazard this gate exists for cannot reach it. Listed only because the walk used to
        # be blind to `*args` and so never surfaced it.
        'pydantic_ai.models.fallback.FallbackModel',
        'pydantic_ai.models.function.DeltaToolCall',
        'pydantic_ai.models.instrumented.InstrumentedModel',
        'pydantic_ai.output.OutputContext',
        'pydantic_ai.output.OutputObjectDefinition',
        'pydantic_ai.result.FinalResult',
        'pydantic_ai.result.StreamedRunResult',
        'pydantic_ai.run.AgentRunResult',
        'pydantic_ai.tool_manager.ToolManager',
        'pydantic_ai.tool_manager.ValidatedToolCall',
        'pydantic_ai.toolsets.approval_required.ApprovalRequiredToolset',
        'pydantic_ai.toolsets.filtered.FilteredToolset',
        'pydantic_ai.toolsets.prefixed.PrefixedToolset',
        'pydantic_ai.toolsets.prepared.PreparedToolset',
        'pydantic_ai.toolsets.renamed.RenamedToolset',
        'pydantic_ai.toolsets.set_metadata.SetMetadataToolset',
    }
)


def test_new_public_dataclasses_are_keyword_only():
    """New public dataclasses must not add a second positional `__init__` parameter.

    "Pretty much all plain dataclasses need `_: KW_ONLY`" is the most-repeated unenforced review
    nit. Existing offenders are grandfathered in `_KW_ONLY_ALLOWLIST` (changing them to
    keyword-only would break positional callers); this test only fails when a NEW public dataclass
    ships with two or more positional parameters, which is where the "add a field, break callers"
    trap lives. Make the new dataclass keyword-only, or add it to the allowlist with maintainer
    sign-off.

    The walk runs out of process with the `COVERAGE_*` environment scrubbed -- see
    `kw_only_walker.py` for why -- so failures arrive as the child's stderr rather than as an
    exception here.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith('COVERAGE_')}
    process = subprocess.run(
        [sys.executable, str(Path(__file__).parent / 'kw_only_walker.py')],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert process.returncode == 0, f'dataclass walk failed:\n{process.stderr}'

    result: dict[str, list[str]] = json.loads(process.stdout)
    offenders = set(result['offenders'])
    skipped = result['skipped']

    assert result['unreadable'] == [], (
        f'could not read a constructor signature for: {result["unreadable"]}; '
        'the walk cannot classify these, so they are neither gated nor grandfathered'
    )
    # Floor: a walk that collapses -- a renamed package, an import that stops resolving -- must
    # fail loudly rather than pass vacuously by finding nothing left to check.
    assert offenders, f'the dataclass walk found nothing to check; skipped modules: {skipped}'

    unexpected_offenders = offenders - _KW_ONLY_ALLOWLIST
    assert unexpected_offenders == set(), (
        f'new public dataclass(es) with two or more positional parameters: {sorted(unexpected_offenders)}; '
        'add `_: KW_ONLY` to the dataclass, or add it to `_KW_ONLY_ALLOWLIST` with maintainer sign-off'
    )

    # Staleness ratchet, and only on a complete walk: CI shards installed without all extras can't
    # import the provider and durable-exec modules, so their offenders legitimately drop out and
    # would look stale when they aren't.
    if not skipped:  # pragma: lax no cover
        stale_entries = _KW_ONLY_ALLOWLIST - offenders
        assert stale_entries == set(), (
            f'`_KW_ONLY_ALLOWLIST` entries no longer offend and must be removed: {sorted(stale_entries)}'
        )


_AGENT_IMPLEMENTATIONS: dict[str, type] = {
    # `AbstractAgent` is the contract third parties implement and `WrapperAgent` performs the
    # forwarding, so both belong here: the `metadata` drift that motivated this test was present on
    # them too, and pyright cannot see it (a subclass *widening* an override with an extra
    # keyword-only parameter is a legal override).
    'AbstractAgent': AbstractAgent,
    'WrapperAgent': WrapperAgent,
}

# Whether each extra is installed varies by CI shard, so both directions of these guards are taken
# across the matrix but never within one job.
if temporal_imports():  # pragma: lax no cover
    _AGENT_IMPLEMENTATIONS['TemporalAgent'] = TemporalAgent  # pyright: ignore[reportDeprecated]
if dbos_imports():  # pragma: lax no cover
    _AGENT_IMPLEMENTATIONS['DBOSAgent'] = DBOSAgent  # pyright: ignore[reportDeprecated]
if prefect_imports():  # pragma: lax no cover
    _AGENT_IMPLEMENTATIONS['PrefectAgent'] = PrefectAgent  # pyright: ignore[reportDeprecated]


def _redeclared_agent_methods(implementation: type) -> list[str]:
    """Public methods the implementation redeclares from `Agent`.

    Derived rather than hardcoded: a hardcoded list carries exactly the "nobody forces this to be
    updated" drift these tests exist to eliminate, so an implementation that later redeclares
    another `Agent` method would silently escape the guard.
    """
    return sorted(
        name for name in vars(implementation) if not name.startswith('_') and callable(getattr(Agent, name, None))
    )


_AGENT_METHOD_PARAMS = [
    pytest.param(implementation, method_name, id=f'{name}-{method_name}')
    for name, implementation in _AGENT_IMPLEMENTATIONS.items()
    for method_name in _redeclared_agent_methods(implementation)
]


def _parameter_kinds(method: Callable[..., object]) -> dict[str, str]:
    return {name: parameter.kind.name for name, parameter in inspect.signature(method).parameters.items()}


@pytest.mark.parametrize(('implementation', 'method_name'), _AGENT_METHOD_PARAMS)
def test_agent_implementation_signature_parity(implementation: type, method_name: str):
    """Agent wrappers hand-mirror `Agent`'s method signatures, and nothing else forces them to stay in sync.

    `AbstractAgent`, `WrapperAgent` and the durable-execution wrappers redeclare `run`, `run_sync`,
    `run_stream`, `run_stream_events`, `iter` and `override` so they can wrap the run. Nothing
    forces a new keyword added to `Agent` to be copied into each one, so a wrapper silently drops
    support for it -- which is exactly how `override(metadata=...)` came to raise `TypeError` on
    every wrapper while `Agent` accepted it. This asserts every method accepts (at least) the same
    parameters, by name and kind, as the corresponding `Agent` method.
    """
    base_parameter_kinds = _parameter_kinds(getattr(Agent, method_name))
    actual_parameter_kinds = _parameter_kinds(getattr(implementation, method_name))

    missing = {name: kind for name, kind in base_parameter_kinds.items() if actual_parameter_kinds.get(name) != kind}
    assert missing == {}, (
        f'{implementation.__name__}.{method_name} is missing (or has a different kind for) '
        f'parameters present on `Agent.{method_name}`: {missing}'
    )


# Keyword-only parameters an implementation declares and does not pass straight on, each carrying
# the reason that is correct. `None` means the method forwards nothing at all.
#
# `test_agent_implementation_forwarding_parity` re-checks every entry against the source it exempts,
# so an entry that stops applying fails instead of silently switching the guard off for that
# parameter forever.
_UNFORWARDED_BY_DESIGN: dict[tuple[str, str], frozenset[str] | None] = {
    # Abstract declarations -- the body is `raise NotImplementedError`.
    ('AbstractAgent', 'iter'): None,
    ('AbstractAgent', 'override'): None,
    # Default implementation, `return []`; concrete agents override it.
    ('AbstractAgent', 'system_prompt_parts'): None,
    # `run_stream_events()` is unsupported under DBOS -- the body raises `UserError`.
    ('DBOSAgent', 'run_stream_events'): None,
    # `infer_name` drives name inference off the *calling* frame, which only this call can see, so
    # every method resolves the name itself and hands the inner call `infer_name=False` (or nothing)
    # precisely so it does not re-infer against the wrong frame.
    ('AbstractAgent', 'run_stream_events'): frozenset({'infer_name'}),
    ('AbstractAgent', 'run_stream_sync'): frozenset({'infer_name'}),
    ('AbstractAgent', 'run_sync'): frozenset({'infer_name'}),
    # Stored rather than delegated: `realtime()` builds an `AgentRealtime` that holds the whole
    # configuration until `session()` opens the connection, so every keyword lands on a private
    # dataclass field (`_deps=deps`) rather than on a same-named parameter. The rename is what the
    # walk sees; nothing is dropped, and `AgentRealtime` passing them on is covered by the realtime
    # session tests.
    ('AbstractAgent', 'realtime'): frozenset(
        {
            'deps',
            'model_settings',
            'instructions',
            'toolsets',
            'capabilities',
            'usage',
            'usage_limits',
            'metadata',
            'conversation_id',
            'run_id',
            'message_history',
        }
    ),
    # Same for `infer_name`, plus `event_stream_handler`, which these two consume rather than
    # delegate: they default it to `self.event_stream_handler` and then drive the event stream
    # themselves against each node's stream, so there is no inner run to hand it to.
    ('AbstractAgent', 'run'): frozenset({'infer_name', 'event_stream_handler'}),
    ('AbstractAgent', 'run_stream'): frozenset({'infer_name', 'event_stream_handler'}),
    # Transformed before forwarding: `model` is resolved to the engine's own model wrapper (or to
    # `None` inside a workflow) and that result is what `super().iter()` receives. `cancellation_token`
    # is consumed locally: it is a same-process handle that cannot cross the durable boundary, so
    # every durable-wrapper entry point rejects it up front with a `UserError` instead of forwarding.
    ('TemporalAgent', 'iter'): frozenset({'model', 'cancellation_token'}),
    # Defaulted before forwarding: `event_stream_handler or self.event_stream_handler`.
    # `cancellation_token` rejected locally (see the `TemporalAgent.iter` note).
    ('TemporalAgent', 'run'): frozenset({'event_stream_handler', 'cancellation_token'}),
    ('TemporalAgent', 'run_sync'): frozenset({'cancellation_token'}),
    ('TemporalAgent', 'run_stream'): frozenset({'cancellation_token'}),
    ('TemporalAgent', 'run_stream_events'): frozenset({'cancellation_token'}),
    ('DBOSAgent', 'run'): frozenset({'cancellation_token'}),
    ('DBOSAgent', 'run_sync'): frozenset({'cancellation_token'}),
    ('DBOSAgent', 'run_stream'): frozenset({'cancellation_token'}),
    ('PrefectAgent', 'run'): frozenset({'cancellation_token'}),
    ('PrefectAgent', 'run_sync'): frozenset({'cancellation_token'}),
    ('PrefectAgent', 'run_stream'): frozenset({'cancellation_token'}),
    ('PrefectAgent', 'run_stream_events'): frozenset({'cancellation_token'}),
    # `toolsets` is applied through the engine's override context instead of the run argument, which
    # is explicitly passed as `toolsets=None` so the runtime toolsets are not added twice.
    # `cancellation_token` rejected locally (see the `TemporalAgent.iter` note).
    ('DBOSAgent', 'iter'): frozenset({'toolsets', 'cancellation_token'}),
    ('PrefectAgent', 'iter'): frozenset({'toolsets', 'cancellation_token'}),
    # Forwarded only when set, through a `**` splat this walk deliberately does not read. The
    # conditional is residue of the removed `output_retries` deprecation shim (`24c8cdca7`) rather
    # than a compatibility mechanism; the other nine keywords forward unconditionally.
    ('WrapperAgent', 'override'): frozenset({'retries'}),
}


def _keyword_only_forwarding(implementation: type, method_name: str) -> tuple[set[str], set[str]]:
    """The keyword-only parameters the method declares, and those it passes straight on.

    "Passes straight on" means literally `inner(name=name)` in some call the method makes: `f(x=x)`
    counts and `g(other=x)` does not. A `**splat` counts for nothing, so a parameter smuggled
    through a dict reads as unforwarded and needs an explicit `_UNFORWARDED_BY_DESIGN` entry --
    treating any splat as blanket proof of forwarding would let a real omission through silently.
    """
    method = inspect.unwrap(getattr(implementation, method_name))
    source = textwrap.dedent(inspect.getsource(method))
    definition = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )

    declared = {argument.arg for argument in definition.args.kwonlyargs}
    forwarded = {
        keyword.value.id
        for call in ast.walk(definition)
        if isinstance(call, ast.Call)
        for keyword in call.keywords
        if isinstance(keyword.value, ast.Name) and keyword.arg == keyword.value.id
    }
    return declared, forwarded & declared


@pytest.mark.parametrize(('implementation', 'method_name'), _AGENT_METHOD_PARAMS)
def test_agent_implementation_forwarding_parity(implementation: type, method_name: str):
    """Accepting a keyword is not the same as passing it on.

    `test_agent_implementation_signature_parity` proves a wrapper *accepts* a parameter; a wrapper
    that accepts one and then omits it from the call it delegates to satisfies that test, stays
    fully line-covered, and silently discards the value -- the same user-visible symptom as the
    drift it was written to catch. Fixing each wrapper for `metadata` took two edits (declaration
    and forwarding), and signature parity only guards the first.

    The guarantee is deliberately narrow: every keyword-only parameter is passed straight on as a
    keyword, unless `_UNFORWARDED_BY_DESIGN` says why not. Consuming a parameter locally, renaming
    it, defaulting it or routing it through a `**` splat all count as not forwarding, so each one
    has to be spelled out with its reason rather than pass by accident.
    """
    declared, forwarded = _keyword_only_forwarding(implementation, method_name)
    by_design = _UNFORWARDED_BY_DESIGN.get((implementation.__name__, method_name), frozenset())

    if by_design is None:
        assert forwarded == set(), (
            f'`_UNFORWARDED_BY_DESIGN` records {implementation.__name__}.{method_name} as forwarding '
            f'nothing, but it now forwards {sorted(forwarded)}; drop the entry or narrow it to the '
            'parameters it still holds back'
        )
        return

    no_longer_held_back = by_design & forwarded
    assert no_longer_held_back == set(), (
        f'{implementation.__name__}.{method_name} now forwards {sorted(no_longer_held_back)}, so '
        'those `_UNFORWARDED_BY_DESIGN` entries no longer apply and must be removed'
    )

    unforwarded = declared - forwarded - by_design
    assert unforwarded == set(), (
        f'{implementation.__name__}.{method_name} declares keyword-only parameter(s) it never '
        f'passes on, so their values are silently dropped: {sorted(unforwarded)}; forward them, or '
        'add them to `_UNFORWARDED_BY_DESIGN` with the reason it is safe'
    )


@pytest.mark.anyio
async def test_wrapper_agent_override_metadata_reaches_the_run():
    """End-to-end pin for the forwarding the meta-tests above check structurally.

    `WrapperAgent` is the class in the public chain that actually performs the forwarding, and it
    needs no durable-execution infrastructure to exercise, so it carries the behavioral assertion
    for the whole family.
    """
    agent = Agent(TestModel(), metadata={'source': 'agent'})
    seen: list[dict[str, object]] = []

    @agent.instructions
    def capture_metadata(ctx: RunContext[object]) -> str:
        seen.append(dict(ctx.metadata or {}))
        return ''

    wrapper = WrapperAgent(agent)
    with wrapper.override(metadata={'source': 'override'}):
        await wrapper.run('hello', metadata={'source': 'run'})
    await wrapper.run('hello', metadata={'source': 'run'})

    assert seen == snapshot([{'source': 'override'}, {'source': 'run'}])
