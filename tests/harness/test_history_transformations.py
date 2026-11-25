"""C4, the transformation registry: no transformation of history gives a part more authority than it had.

Each case's `calls` run once natively, as in C1, with every capability lowered to `with_authority('none')`. Each
transformation in `TRANSFORMS` then turns that history into the one it hands on (a round trip, a sanitized copy, the
history a resumed sub-agent starts from), and three checks compare the two by each request part's `signature`:

1. every part the transformation doesn't drop is still there, with the same type, source, authority and frame;
2. a part of a kind it may create has no more authority than the lowest of the parts it replaces;
3. nothing else appears.

A guard keeps the registry complete: every function in Pydantic AI and Pydantic AI Harness that returns
`list[ModelMessage]` is registered on a `Transform`, or exempt in `TRANSFORM_EXEMPT` with its reason. A
transformation that breaks a check for a case is a known finding, a strict `xfail` linked to its issue in the
transform's `exempt`, so the fix fails the suite until its entry is removed.

The run-level transformations (a `ModelFallback` switch, compaction, history processors), which rerun the case rather
than transform its native history, live in `c4_transforms_runlevel.py`, which merges them into `TRANSFORMS`.
"""

from __future__ import annotations

import ast
import importlib
import re
import warnings
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pytest
from _pytest.mark import ParameterSet

from pydantic_ai._capability_parts import effective_authority
from pydantic_ai.messages import (
    Authority,
    CapabilityLoadDeltaPart,
    CapabilityPart,
    CapabilityPartSource,
    InstructionBaselinePart,
    InstructionDeltaPart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelRequestPart,
    ModelSettingsDeltaPart,
    RetryFeedbackPart,
    SpeechPart,
    SystemPromptPart,
    ToolAvailabilityDeltaPart,
    UserPromptPart,
    sanitize_messages,
)
from pydantic_ai_harness._history import STORED_HISTORY_CONTEXT
from pydantic_ai_harness.sub_agents._tasks import (
    _ADAPTER as _DELEGATION_TASK,  # pyright: ignore[reportPrivateUsage]
    DelegationTask,
)
from pydantic_ai_harness.sub_agents._toolset import _child_inputs  # pyright: ignore[reportPrivateUsage]

from ..append_only import ui_imports_successful
from ..capability_cases import Calls
from ..conftest import try_import
from ..manifests import Exempt, assert_classified
from .test_capability_routes import CASES, PROBES, PUBLIC, RouteRun, _calls, _native  # pyright: ignore[reportPrivateUsage]

with try_import():
    from pydantic_ai.ui.ag_ui import AGUIAdapter
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter

Transformation = Literal[
    'model_fallback_switch',  # run-level: fallback at request k, history compared with the native run
    'summarizing',  # SummarizingCompaction (harness)
    'sliding_window',  # SlidingWindow history processor
    'compaction_part',  # provider compaction kept as CompactionPart (OpenAI/Anthropic shapes, offline)
    'history_processor',  # a user `history_processors=[...]` that returns a filtered copy
    'type_adapter',  # ModelMessagesTypeAdapter dump/load (also in C1; kept here for the registry)
    'sanitize',  # sanitize_messages(strip_system_prompts=True)
    'vercel_ai',  # dump_messages/load_messages round trip
    'ag_ui',  # dump_messages/load_messages round trip
    'sub_agent_handoff',  # what SubAgents passes the child
]


@dataclass(frozen=True)
class Transform:
    """One transformation of history, and what it may legitimately change."""

    name: Transformation
    apply: Callable[[RouteRun, Path], Awaitable[list[ModelMessage]]]
    """Takes C1's native `RouteRun` for the case (or reruns the case, for run-level transforms) and returns the
    transformed history."""
    drops: Callable[[ModelRequestPart], bool] = lambda _: False
    """Parts the transformation may legitimately leave out (e.g. everything before a compaction boundary)."""
    creates: tuple[type[ModelRequestPart], ...] = ()
    """Part kinds it may add (a summary, a CompactionPart, a `<system>`-tag note)."""
    exempt: Mapping[str, Exempt] = field(default_factory=dict[str, Exempt])
    """Case name -> reason with issue link, as C1: the case's checks are a strict `xfail`."""
    functions: tuple[str, ...] = ()
    """The qualified names of the functions that make the transformation, which the registry guard finds."""


@dataclass
class NativeRun(RouteRun):
    """The `RouteRun` every `apply` is given: C1's native run, with the case it ran, for a transform that reruns it."""

    name: str
    """The case's name in `CASES`, or the probe's in `PROBES`."""
    calls: Calls


# --- Authority and signature ---------------------------------------------------------------------------------------

_PART_AUTHORITY: dict[type[object], Authority] = {
    SystemPromptPart: 'operator',
    # The framework's own records of the run's state, which the model reads in the application's voice.
    InstructionBaselinePart: 'operator',
    InstructionDeltaPart: 'operator',
    ToolAvailabilityDeltaPart: 'operator',
    CapabilityLoadDeltaPart: 'operator',
    ModelSettingsDeltaPart: 'operator',
    UserPromptPart: 'user',
    SpeechPart: 'user',
}
"""Until #12434 widens `effective_authority` to every request part: the authority of each kind that isn't data.

Tool output, retry feedback, a compaction summary and any other kind are `'none'`, as #12434 has them."""


def _authority(part: ModelRequestPart) -> Authority:
    """`effective_authority` for every request part, until #12434 lands it: then this stub goes."""
    if isinstance(part, CapabilityPart):
        return effective_authority(part)
    return next((authority for kind, authority in _PART_AUTHORITY.items() if isinstance(part, kind)), 'none')


_RANK: list[Authority] = ['none', 'user', 'operator']

Signature = tuple[str, CapabilityPartSource | None, Authority, tuple[str, str | None] | None]


def signature(part: ModelRequestPart) -> Signature:
    """What the model is told about who wrote `part`: its type, source, effective authority and frame."""
    if isinstance(part, CapabilityPart):
        return type(part).__name__, part.source, _authority(part), (part.rendering.voice, part.rendering.tag)
    return type(part).__name__, None, _authority(part), None


def _request_parts(messages: Sequence[ModelMessage]) -> list[ModelRequestPart]:
    return [part for message in messages if isinstance(message, ModelRequest) for part in message.parts]


# --- The transformations -------------------------------------------------------------------------------------------


async def _type_adapter(native: RouteRun, directory: Path) -> list[ModelMessage]:
    return ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(native.messages))


async def _sanitize(native: RouteRun, directory: Path) -> list[ModelMessage]:
    with warnings.catch_warnings():
        # It warns about the system prompts and capability parts it strips, as `drops` expects.
        warnings.filterwarnings('ignore', 'Parts carrying the system voice were stripped', UserWarning)
        warnings.filterwarnings('ignore', 'Capability parts were dropped', UserWarning)
        return sanitize_messages(native.messages, strip_system_prompts=True)


async def _vercel_ai(native: RouteRun, directory: Path) -> list[ModelMessage]:
    return VercelAIAdapter.load_messages(VercelAIAdapter.dump_messages(native.messages))


async def _ag_ui(native: RouteRun, directory: Path) -> list[ModelMessage]:
    return AGUIAdapter.load_messages(AGUIAdapter.dump_messages(native.messages))


async def _sub_agent_handoff(native: RouteRun, directory: Path) -> list[ModelMessage]:
    """The history a resumed child starts from: its task saved and read back as the task store does."""
    record = DelegationTask(id='0' * 32, agent_name='child', prompt='go', conversation_id='chat')
    record.messages = native.messages
    saved = _DELEGATION_TASK.validate_json(_DELEGATION_TASK.dump_json(record), context=STORED_HISTORY_CONTEXT)
    _, history, _ = _child_inputs('next', saved, None)
    assert history is not None
    return history


def _instruction_record(part: ModelRequestPart) -> bool:
    return isinstance(part, SystemPromptPart | InstructionDeltaPart | InstructionBaselinePart | CapabilityPart)


_TRANSFORMS: dict[Transformation, Transform] = {
    'type_adapter': Transform('type_adapter', _type_adapter),
    'sanitize': Transform(
        'sanitize',
        _sanitize,
        # Documented: system prompts and instruction records are the server's, and capability parts are always dropped.
        drops=_instruction_record,
        # A `'no_output'` retry feedback becomes the same feedback in the user voice.
        creates=(RetryFeedbackPart,),
        functions=('pydantic_ai.messages.sanitize_messages', 'pydantic_ai.ui._adapter.UIAdapter.sanitize_messages'),
    ),
    'vercel_ai': Transform(
        'vercel_ai',
        _vercel_ai,
        # Documented: the server owns capability parts, system prompts and instruction records, so a client never holds one.
        drops=_instruction_record,
        functions=('pydantic_ai.ui.vercel_ai._adapter.VercelAIAdapter.load_messages',),
    ),
    'ag_ui': Transform(
        'ag_ui',
        _ag_ui,
        # Documented: the server owns capability parts, system prompts and instruction records, so a client never holds one.
        drops=_instruction_record,
        functions=('pydantic_ai.ui.ag_ui._adapter.AGUIAdapter.load_messages',),
    ),
    'sub_agent_handoff': Transform(
        'sub_agent_handoff',
        _sub_agent_handoff,
        functions=(
            'pydantic_ai_harness.sub_agents._tasks.DelegationTasks.history',
            'pydantic_ai.messages.repair_messages',
        ),
    ),
}


def _run_level_transforms() -> dict[Transformation, Transform]:
    """The run-level transforms in `c4_transforms_runlevel.py`, which imports `Transform` from here, once it exists."""
    module = f'{__package__}.c4_transforms_runlevel'
    try:
        return importlib.import_module(module).RUN_LEVEL_TRANSFORMS
    except ModuleNotFoundError as error:
        # Only the module missing: an import error inside it fails the suite.
        if error.name != module:  # pragma: no cover
            raise
        return {}


TRANSFORMS: dict[Transformation, Transform] = {**_TRANSFORMS, **_run_level_transforms()}


# --- The checks ----------------------------------------------------------------------------------------------------


def _subjects() -> list[str]:
    cases = [name for name, case in sorted(CASES.items()) if name in PUBLIC and case.calls is not None]
    return [*cases, *PROBES]


def _matrix() -> Iterator[ParameterSet]:
    for name in _subjects():
        calls = PROBES.get(name) or CASES[name].calls
        assert calls is not None
        for transformation, transform in TRANSFORMS.items():
            marks: list[pytest.MarkDecorator] = []
            if (exempt := transform.exempt.get(name)) is not None:
                marks.append(pytest.mark.xfail(strict=True, raises=AssertionError, reason=exempt.reason))
            if transformation in ('vercel_ai', 'ag_ui') and not ui_imports_successful():
                marks.append(pytest.mark.skip(reason='the UI adapters are not installed'))
            if calls.moves_cache_prefix is not None:
                marks.append(pytest.mark.moves_cache_prefix(reason=calls.moves_cache_prefix))
            yield pytest.param(name, transformation, marks=marks, id=f'{name}-{transformation}')


@dataclass
class _Compared:
    native: list[ModelRequestPart]
    output: list[ModelRequestPart]
    transform: Transform

    @property
    def kept(self) -> Counter[Signature]:
        return Counter(signature(part) for part in self.native if not self.transform.drops(part))

    @property
    def created(self) -> list[ModelRequestPart]:
        return [part for part in self.output if isinstance(part, self.transform.creates)]

    @property
    def carried(self) -> Counter[Signature]:
        return Counter(signature(part) for part in self.output if not isinstance(part, self.transform.creates))


async def _compare(name: str, transformation: Transformation, directory: Path) -> _Compared:
    calls = _calls(name)
    run = await _native(calls, directory / 'native')
    assert run.ran, 'the calls ran no tool'
    native = NativeRun(run.messages, run.ran, name, calls)
    transform = TRANSFORMS[transformation]
    output = await transform.apply(native, directory / transformation)
    return _Compared(_request_parts(native.messages), _request_parts(output), transform)


@pytest.mark.parametrize(('name', 'transformation'), list(_matrix()))
async def test_kept_parts_are_unchanged(name: str, transformation: Transformation, tmp_path: Path) -> None:
    """Every part the transformation doesn't drop is still there, with the same signature."""
    compared = await _compare(name, transformation, tmp_path)
    lost = compared.kept - Counter(signature(part) for part in compared.output)
    assert not lost, f'{transformation} lost or changed {sorted(lost.elements(), key=repr)}'


@pytest.mark.parametrize(('name', 'transformation'), list(_matrix()))
async def test_no_part_gains_authority(name: str, transformation: Transformation, tmp_path: Path) -> None:
    """A created part has no more authority than the lowest of the parts it replaces.

    The parts it replaces are those whose signature no longer appears; when none is gone, it's the whole history.
    """
    compared = await _compare(name, transformation, tmp_path)
    native = Counter(signature(part) for part in compared.native)
    replaced = native - Counter(signature(part) for part in compared.output) or native
    ceiling = min((authority for _, _, authority, _ in replaced), key=_RANK.index)
    gained = [part for part in compared.created if _RANK.index(_authority(part)) > _RANK.index(ceiling)]
    assert not gained, f'{transformation} created {gained} above {ceiling!r}'


@pytest.mark.parametrize(('name', 'transformation'), list(_matrix()))
async def test_nothing_new_appears(name: str, transformation: Transformation, tmp_path: Path) -> None:
    """Every part outside the kinds the transformation may create has a native counterpart."""
    compared = await _compare(name, transformation, tmp_path)
    new = compared.carried - Counter(signature(part) for part in compared.native)
    assert not new, f'{transformation} added {sorted(new.elements(), key=repr)}'


# --- The registry guard --------------------------------------------------------------------------------------------

_ROOT = Path(__file__).parents[2]
_PACKAGES = (_ROOT / 'pydantic_ai_slim' / 'pydantic_ai', _ROOT / 'src' / 'pydantic_ai_harness' / 'pydantic_ai_harness')
_RETURNS_HISTORY = re.compile(r'list\[(\w+\.)?ModelMessage\]')


def _history_functions() -> set[str]:
    """The qualified name of every function in the packages annotated to return `list[ModelMessage]`."""
    found: set[str] = set()

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, f'{prefix}.{child.name}')
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                if child.returns is not None and _RETURNS_HISTORY.fullmatch(ast.unparse(child.returns).strip('\'"')):
                    found.add(f'{prefix}.{child.name}')
                walk(child, f'{prefix}.{child.name}.<locals>')

    for package in _PACKAGES:
        for path in sorted(package.rglob('*.py')):
            module = '.'.join((package.name, *path.relative_to(package).with_suffix('').parts)).removesuffix(
                '.__init__'
            )
            walk(ast.parse(path.read_text()), module)
    return found


_RECORD = 'returns the history a run or session recorded, as it recorded it, not a transformed copy'
_PREPARE = (
    "a step of `Model.prepare_messages` or a model's own preparation: its output goes to the provider and never back "
    "into history, and C2's sink check covers what it renders"
)
_REPAIR = 'a step of `repair_messages` or `sanitize_messages`, which the registered transforms run'
_UI = 'part of a UI adapter run, whose `load_messages` and `sanitize_messages` are registered and whose server-held history C1 checks'
_BUILDS = 'builds the parts a run appends, which C1 checks on every route, rather than transforming earlier history'
_FOREIGN = "builds a history from another protocol's or library's messages: there's no Pydantic AI history to compare"
_NOT_SENT = 'its output is compared or indexed, never sent to a model'
_DURABLE = 'a durable round trip, which D1 covers per case'
_PENDING = "PENDING: rewrites history the model later sees, and isn't a `Transform` yet"

TRANSFORM_EXEMPT: dict[str, str] = {
    **dict.fromkeys(
        [
            'pydantic_ai._cli.ask_agent',
            'pydantic_ai.agent.abstract.AgentRunEvents.all_messages',
            'pydantic_ai.agent.abstract.AgentRunEvents.new_messages',
            'pydantic_ai.agent.abstract._recorded_at_pause',
            'pydantic_ai.exceptions.RunCancelled.all_messages',
            'pydantic_ai.exceptions.RunCancelled.new_messages',
            'pydantic_ai.realtime._core.SessionCore.all_messages',
            'pydantic_ai.realtime._core.SessionCore.new_messages',
            'pydantic_ai.realtime._core.SessionCore.replayable_messages',
            'pydantic_ai.realtime._core.SessionCore._history_before',
            'pydantic_ai.realtime._core.SessionCore.conversation_so_far',
            'pydantic_ai.realtime._session.RealtimeSession.all_messages',
            'pydantic_ai.realtime._session.RealtimeSession.new_messages',
            'pydantic_ai.result.StreamedRunResult.all_messages',
            'pydantic_ai.result.StreamedRunResult.new_messages',
            'pydantic_ai.result.StreamedRunResultSync.all_messages',
            'pydantic_ai.result.StreamedRunResultSync.new_messages',
            'pydantic_ai.run.AgentRun.all_messages',
            'pydantic_ai.run.AgentRun.new_messages',
            'pydantic_ai.run.AgentRunResult.all_messages',
            'pydantic_ai.run.AgentRunResult.new_messages',
            'pydantic_ai.ui._messages_builder.MessagesBuilder._modified_since',
            'pydantic_ai_harness.conversation_search._source.HistorySource.run_history',
            'pydantic_ai_harness.conversation_search._source.SnapshotHistorySource.run_history',
        ],
        _RECORD,
    ),
    **dict.fromkeys(
        [
            'pydantic_ai._capability_parts.render_capability_parts',
            'pydantic_ai._capability_parts.render_capability_parts_for_live_channel',
            'pydantic_ai._special_tokens.break_special_tokens_in_messages',
            'pydantic_ai._tool_search.synthesize_local_tool_search_messages',
            'pydantic_ai._tool_search.repair_dangling_translated_calls',
            'pydantic_ai.direct._prepare_messages_carrying_a_retry',
            'pydantic_ai.direct._break_special_tokens',
            'pydantic_ai.durable_exec.temporal._model.TemporalModel.prepare_messages',
            'pydantic_ai.durable_exec.temporal._model.TemporalModel._reprepare_messages',
            'pydantic_ai.messages.post_compaction_window',
            'pydantic_ai.messages._post_compaction_window_for_response',
            'pydantic_ai.models.Model._restore_response_scoped_tool_call_ids',
            'pydantic_ai.models.Model.prepare_messages',
            'pydantic_ai.models.Model._repair_dangling_translated_tool_search_calls',
            'pydantic_ai.models.Model._translate_legacy_tool_reveals',
            'pydantic_ai.models.Model._announce_in_standing_prompt',
            'pydantic_ai.models._convert_speech_parts',
            'pydantic_ai.models._drop_superseded_request_scoped_prompts',
            'pydantic_ai.models._without_answered_parts',
            'pydantic_ai.models._drop_request_scoped_prompts_no_user_message_clears',
            'pydantic_ai.models._move_request_scoped_prompts_to_tail',
            'pydantic_ai.models._wrap_non_leading_system_prompts',
            'pydantic_ai.models._replies_noted_later',
            'pydantic_ai.models._tool_returns_as_json_strings',
            'pydantic_ai.models._replace_unsupported_tool_return_files',
            'pydantic_ai.models._project_instruction_state',
            'pydantic_ai.models._drop_capability_load_deltas',
            'pydantic_ai.models._project_model_settings_deltas',
            'pydantic_ai.models._without_model_settings_deltas',
            'pydantic_ai.models._render_system_prompt_contexts',
            'pydantic_ai.models._order_tool_results_by_call',
            'pydantic_ai.models._fit_tool_call_ids',
            'pydantic_ai.models._translate_retry_parts',
            'pydantic_ai.models._replace_tool_search_exchanges_with_deltas',
            'pydantic_ai.models._announce_tool_availability_delta_messages',
            'pydantic_ai.models._synthesize_tool_availability_delta_messages',
            'pydantic_ai.models.anthropic._compaction.without_unanswered_tool_calls',
            'pydantic_ai.models.decision.DecisionModel.prepare_messages',
            'pydantic_ai.models.decision._without_catalog_changes',
            'pydantic_ai.models.fallback.FallbackModel.prepare_messages',
            'pydantic_ai.models.wrapper.WrapperModel.prepare_messages',
            'pydantic_ai.realtime._session.RealtimeSession._replay_instruction_changes',
            'pydantic_ai.realtime._utils.wrap_seeded_system_prompts',
        ],
        _PREPARE,
    ),
    **dict.fromkeys(
        [
            'pydantic_ai.messages.narrow_message_parts',
            'pydantic_ai.messages._drop_compaction_parts',
            'pydantic_ai.messages._drop_orphaned_tool_results',
            'pydantic_ai.messages._repair_dangling_tool_calls',
            'pydantic_ai.messages._close_out_dangling_tool_calls',
            'pydantic_ai.messages._merge_consecutive_messages',
            'pydantic_ai.messages._clean_message_history',
        ],
        _REPAIR,
    ),
    **dict.fromkeys(
        [
            'pydantic_ai.ui._adapter.enqueued_messages_for_client',
            'pydantic_ai.ui._adapter.UIAdapter.load_messages',
            'pydantic_ai.ui._adapter.UIAdapter.messages',
            'pydantic_ai.ui._adapter.UIAdapter._server_held_client_messages',
            'pydantic_ai.ui.ag_ui._adapter.AGUIAdapter.messages',
            'pydantic_ai.ui.ag_ui._adapter.AGUIAdapter._server_held_client_messages',
            'pydantic_ai.ui.vercel_ai._adapter.VercelAIAdapter.messages',
        ],
        _UI,
    ),
    **dict.fromkeys(
        [
            'pydantic_ai._enqueue._build_enqueue_messages',
            'pydantic_ai._enqueue.stamp_capability_parts',
            'pydantic_ai._enqueue.stamped_messages',
        ],
        _BUILDS,
    ),
    **dict.fromkeys(
        ['pydantic_ai._mcp.map_from_mcp_params', 'pydantic_ai_harness.browser_use._model._map_messages'], _FOREIGN
    ),
    **dict.fromkeys(
        [
            'pydantic_ai.conversation_store.ConversationStore._sent_as_cleaned.<locals>.cleaned_from',
            'pydantic_ai_harness.conversation_search._source.without_summary_artifacts',
        ],
        _NOT_SENT,
    ),
    **dict.fromkeys(
        [
            'pydantic_ai_harness.step_persistence._helpers.continue_run',
            'pydantic_ai_harness.step_persistence._helpers.fork_run',
        ],
        _DURABLE,
    ),
    **dict.fromkeys(
        [
            'pydantic_ai._agent_graph._repair_interrupted_tail',
            'pydantic_ai.capabilities.reinject_system_prompt._without_parts',
            'pydantic_ai.capabilities.reinject_system_prompt._reinject_system_prompt',
            'pydantic_ai.run.AgentRunResult._set_output_tool_return',
            'pydantic_ai_harness.compaction._background.BackgroundCompaction.applied_to',
            'pydantic_ai_harness.compaction._background._compact_traced',
            'pydantic_ai_harness.compaction._capability_loads.keep_capability_loads',
            'pydantic_ai_harness.compaction._capability_loads.keep_tool_discoveries',
            'pydantic_ai_harness.compaction._capability_loads.keep_loaded_state',
            'pydantic_ai_harness.compaction._clamp_oversized_messages.ClampOversizedMessages.compact',
            'pydantic_ai_harness.compaction._clear_tool_results.ClearToolResults.compact',
            'pydantic_ai_harness.compaction._compact_on_request.CompactOnRequest._compact',
            'pydantic_ai_harness.compaction._deduplicate_file_reads.DeduplicateFileReads.compact',
            'pydantic_ai_harness.compaction._fallback_compaction.FallbackCompaction.take_background_compaction.<locals>.applied',
            'pydantic_ai_harness.compaction._fallback_compaction.FallbackCompaction.compact',
            'pydantic_ai_harness.compaction._fallback_compaction.FallbackCompaction._compact',
            'pydantic_ai_harness.compaction._fallback_compaction.FallbackCompaction._compact_pinned',
            'pydantic_ai_harness.compaction._manual.compact_now',
            'pydantic_ai_harness.compaction._pinning.reinject_pinned',
            'pydantic_ai_harness.compaction._prune_tool_results.PruneToolResults.after_tool_execute.<locals>.prune',
            'pydantic_ai_harness.compaction._relevance_compaction.RelevanceCompaction.compact',
            'pydantic_ai_harness.compaction._relevance_compaction.RelevanceCompaction._truncated',
            'pydantic_ai_harness.compaction._relevance_compaction.RelevanceCompaction._rebuild',
            'pydantic_ai_harness.compaction._shared.compact_with_span',
            'pydantic_ai_harness.compaction._shared.CompactionStrategy.compact',
            'pydantic_ai_harness.compaction._shared.split_legacy_memory',
            'pydantic_ai_harness.compaction._shared.prepend_first_user_message',
            'pydantic_ai_harness.compaction._shared.rebuild_with_cleared',
            'pydantic_ai_harness.compaction._shared.replace_returns',
            'pydantic_ai_harness.compaction._tiered_compaction.TieredCompaction._escalate',
            'pydantic_ai_harness.compaction._tiered_compaction.TieredCompaction.compact',
            'pydantic_ai_harness.compaction._warn_near_limits.WarnNearLimits._strip_old_warnings',
            'pydantic_ai_harness.experimental.acp._adapter._committable_history',
        ],
        _PENDING,
    ),
}
"""Each function that returns `list[ModelMessage]` without being a registered transformation, with why.

A run-level entry goes when `c4_transforms_runlevel.py` registers it; a `PENDING` one when it becomes a `Transform`."""


def test_every_history_function_is_registered() -> None:
    """A new function that returns history must be registered on a `Transform`, or exempt with its reason."""
    registered = {function: transform.name for transform in TRANSFORMS.values() for function in transform.functions}
    assert not (both := sorted(registered.keys() & TRANSFORM_EXEMPT.keys())), (
        f'registered and still exempt in `TRANSFORM_EXEMPT`, so remove the exemption: {both}'
    )
    table = {
        **{function: Exempt(f'registered as {name}') for function, name in registered.items()},
        **{function: Exempt(reason) for function, reason in TRANSFORM_EXEMPT.items()},
    }
    assert_classified(_history_functions(), table, complete_walk=True, table_name='TRANSFORM_EXEMPT')


def test_every_exemption_names_a_subject() -> None:
    """An exemption for a case that no longer has calls would hide the next finding there."""
    subjects = set(_subjects())
    assert [(t.name, case) for t in TRANSFORMS.values() for case in t.exempt if case not in subjects] == []
