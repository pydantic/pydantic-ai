"""What two of a harness capability on one agent mean.

Every capability this package ships is listed in `COMBINE_POLICY`, and
`test_every_capability_declares_a_combine_policy` fails when one is missing. Adding a capability is
therefore a decision about what two of it mean, taken once, here.

The three answers, and what picks between them:

- `Combines` -- the class declares a default `id`, so two are one configuration stated twice and
  Pydantic AI merges them field by field. This is what lets two packaged harnesses that each carry
  a `ToolOutputLimits` compose instead of colliding.
- `Anonymous` -- no default `id`, and two really do coexist.
- `Collides` -- no default `id`, and two never coexist anyway, because the toolset registers fixed
  tool names. A fact rather than a preference, and pinned by
  `test_two_of_a_colliding_capability_still_raise` so it cannot drift back into an `Anonymous`
  reason describing a configuration ("one per rooted directory") that is unreachable.

Declaring a default `id` is the whole policy: there is no `combine` to write unless the merge needs
something the field-by-field default cannot express, such as a budget that should take the *smaller*
value. None of this package's capabilities needs one.

The core half of this lives in `pydantic-ai`'s `tests/test_capability_combine.py`.

Two of the names imported below are private to pydantic-ai, which is right: the duplicate-resolution
pipeline is internal and no code in this package needs it. This file reaches in anyway rather than
reimplementing the two questions the resolver asks -- a lookalike would drift from the real answer
silently, which is the one thing the policy table exists to prevent. A rename surfaces in the
harness-compat job, which is where a private-API dependency should surface.
"""

from __future__ import annotations

import importlib
import pkgutil
import tempfile
import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, TypeGuard

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import CombinedCapability, Thinking
from pydantic_ai.capabilities.abstract import (
    AbstractCapability,
    leaf_capabilities,
)
from pydantic_ai.capabilities.abstract import (
    _combine_duplicate_capabilities as combine_duplicate_capabilities,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.capabilities.abstract import (
    _declares_default_id as declares_default_id,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel

import pydantic_ai_harness
from pydantic_ai_harness import (
    Advisor,
    Coder,
    Memory,
    Planning,
    Researcher,
    SpendLimits,
    StepPersistence,
    SubAgent,
    SubAgents,
    SummarizingCompaction,
    SystemReminders,
    ToolOutputLimits,
)
from pydantic_ai_harness.system_reminders import Reminder

pytestmark = pytest.mark.anyio

_TMP_A = Path(tempfile.mkdtemp(prefix='combine-a-'))
_TMP_B = Path(tempfile.mkdtemp(prefix='combine-b-'))
"""Two distinct roots, so a `Collides` pair differs in the way its `Anonymous` reason once claimed
mattered -- and still collides."""


@dataclass
class Anonymous:
    """No default `id`, and two of them really do coexist: two different things, both active."""

    reason: str


@dataclass
class Rejected:
    """No default `id`, and two never coexist because something else rejects them.

    Distinct from `Collides`, which is about tool names. A durability capability is refused by the
    engine's own `from_agent` lookup, which runs before an `id` would matter -- so the reason it
    cannot repeat has nothing to do with this table's subject, and saying "tool names" would be
    wrong.
    """

    reason: str


@dataclass
class Collides:
    """No default `id`, and two of them never coexist: their toolsets register the same tool names.

    A fact about the capability, not a preference. It is recorded here because the alternative --
    an `Anonymous` reason like "one per rooted directory" -- describes a configuration the fixed
    tool names make unreachable, and nothing would catch the drift.

    Whether these should instead declare a default `id`, so that two merge, is a decision for each
    capability's owner: merging `FileSystem` would union allow-lists, which widens access. This
    table records where the question applies rather than answering it silently.
    """

    reason: str
    make: Callable[[type[Any]], tuple[AbstractCapability[Any], AbstractCapability[Any]]] | None = None
    """Builds two from the discovered class, when two can be built without credentials or a network.

    Takes the class rather than closing over an import: these capabilities live in optional groups,
    and a module-level import would break the whole file on a lane that installs none of them.
    """


@dataclass
class Combines:
    """A default `id`: two of these are one configuration stated twice, and `combine` resolves them."""

    reason: str
    make: Callable[[], tuple[AbstractCapability[Any], AbstractCapability[Any]]]
    check: Callable[[Any], None]


Policy = Anonymous | Collides | Combines | Rejected


def _check_memory(merged: Any) -> None:
    assert merged.heading == 'Second'


def _check_planning(merged: Any) -> None:
    assert merged.inject is False


def _check_spend_limits(merged: Any) -> None:
    assert merged.expose_tools is False


def _check_step_persistence(merged: Any) -> None:
    assert merged.agent_name == 'second'


def _check_summarizing(merged: Any) -> None:
    assert merged.keep_messages == 7


def _check_system_reminders(merged: Any) -> None:
    # Additive sequences union, so every reminder either side declared still fires.
    assert [r.content for r in merged.reminders] == ['first', 'second']


def _check_tool_output_limits(merged: Any) -> None:
    assert merged.strip_ansi is True


def _check_advisor(merged: Any) -> None:
    assert merged.max_tokens == 4096


def _check_sub_agents(merged: Any) -> None:
    # Rosters union: an agent either side could reach stays reachable through one delegate tool.
    assert [entry.agent.name for entry in merged.agents] == ['alpha', 'beta']


COMBINE_POLICY: dict[str, Policy] = {
    # -- One per agent: a default `id`, and `combine` says what two of them mean. --
    'Memory': Combines(
        'one memory configuration per agent; its toolset registers fixed tool names',
        lambda: (Memory[Any](heading='First'), Memory[Any](heading='Second')),
        _check_memory,
    ),
    'Planning': Combines(
        'one plan per agent; `PlanningToolset` registers fixed tool names',
        lambda: (Planning[Any](), Planning[Any](inject=False)),
        _check_planning,
    ),
    'SpendLimits': Combines(
        'one spend authority per agent',
        lambda: (SpendLimits[Any](), SpendLimits[Any](expose_tools=False)),
        _check_spend_limits,
    ),
    'StepPersistence': Combines(
        'one run identity per agent',
        lambda: (StepPersistence[Any](agent_name='first'), StepPersistence[Any](agent_name='second')),
        _check_step_persistence,
    ),
    'SummarizingCompaction': Combines(
        'two would each make a model call, the second summarizing the first summary',
        lambda: (
            SummarizingCompaction[Any](max_messages=50, keep_messages=3),
            SummarizingCompaction[Any](max_messages=50, keep_messages=7),
        ),
        _check_summarizing,
    ),
    'SystemReminders': Combines(
        'reminders are additive, so both sides keep firing',
        lambda: (
            SystemReminders[Any](reminders=[Reminder('first')]),
            SystemReminders[Any](reminders=[Reminder('second')]),
        ),
        _check_system_reminders,
    ),
    'ToolOutputLimits': Combines(
        'one output-limit policy; `tool_filter`/`per_tool` are how one instance varies by tool',
        lambda: (ToolOutputLimits[Any](), ToolOutputLimits[Any](strip_ansi=True)),
        _check_tool_output_limits,
    ),
    'SubAgents': Combines(
        'one delegate tool per agent, so two rosters become one',
        lambda: (
            SubAgents[Any](agents=[SubAgent(Agent(TestModel(), name='alpha'))]),
            SubAgents[Any](agents=[SubAgent(Agent(TestModel(), name='beta'))]),
        ),
        _check_sub_agents,
    ),
    'Advisor': Combines(
        'one advisor per agent; its tool name is fixed',
        lambda: (
            Advisor[Any]('anthropic:claude-fable-5', max_tokens=2048),
            Advisor[Any]('anthropic:claude-fable-5', max_tokens=4096),
        ),
        _check_advisor,
    ),
    # -- Several of these is the normal case, so they stay anonymous. --
    'Coder': Anonymous('a packaged harness; composing two is composing their members'),
    'Researcher': Anonymous('a packaged harness; composing two is composing their members'),
    'ClampOversizedMessages': Anonymous('clamping twice is a no-op; several thresholds compose'),
    'ClearToolResults': Anonymous('several form an escalation ladder, like `TieredCompaction` tiers'),
    'DeduplicateFileReads': Anonymous('file-read identification is agent-specific; one per `file_key`'),
    'DynamicWorkflow': Anonymous('one per workflow definition'),
    'InputGuardrail': Anonymous('several guards is the design'),
    'OutputGuardrail': Anonymous('several guards is the design'),
    'PromptInjectionDefender': Anonymous('one per `tool_filter`; several scopes compose'),
    'ToolGuardrail': Anonymous('several guards is the design'),
    'ManagedPrompt': Anonymous('one per prompt name'),
    'RepoContext': Anonymous('one per workspace root'),
    'ReportContextUsage': Anonymous('a passive observer; several callbacks compose'),
    'Skills': Anonymous('a factory: one deferred capability per skill, each named after the skill'),
    'SlidingWindowCompaction': Anonymous('composes as a tier under `TieredCompaction`'),
    'StackOne': Anonymous('one per linked account, and `account_id` is what names it'),
    'TieredCompaction': Anonymous('drives other strategies; one per tier list'),
    'WarnNearLimits': Anonymous('a passive observer; several thresholds compose'),
    'WarnOnCacheBusts': Anonymous('a passive observer; several thresholds compose'),
    'AWSLambdaDurability': Rejected(
        'a durability engine is one per agent; `from_agent` rejects a second when the engine looks '
        'itself up, before any id is consulted'
    ),
    # -- No default `id`, but two never coexist anyway: their tool names collide. --
    'FileSystem': Collides(
        'its toolset registers `read_file` and friends under fixed names',
        lambda cls: (cls(str(_TMP_A)), cls(str(_TMP_B))),
    ),
    'Shell': Collides(
        'its toolset registers `run_command` and friends under fixed names',
        lambda cls: (cls(cwd=str(_TMP_A)), cls(cwd=str(_TMP_B))),
    ),
    'CapabilityCreation': Collides(
        'its toolset registers `author_capability` and friends under fixed names',
        lambda cls: (cls(str(_TMP_A)), cls(str(_TMP_B))),
    ),
    'PydanticAIDocs': Collides(
        'its toolset registers `read_pyai_docs` under a fixed name',
        lambda cls: (cls(), cls()),
    ),
    'PyaiDocs': Collides('deprecated alias of `PydanticAIDocs`, and collides the same way'),
    'Macroscope': Collides(
        'its toolset registers `run_macroscope_review` under a fixed name',
        lambda cls: (cls(), cls()),
    ),
    'LocalStack': Collides(
        'its toolset registers `aws_cli` and `localstack_health` under fixed names',
        lambda cls: (cls(), cls()),
    ),
    'CodeMode': Collides('`run_code` is reserved, so a second one is rejected by name'),
    'BrowserUse': Collides('its toolset registers its browser tools under fixed names'),
    'PlaywrightBrowser': Collides('its toolset registers `click` and friends under fixed names'),
    'ModalSandbox': Collides('its toolset registers `run_command` and friends under fixed names'),
    'ConversationSearch': Collides('its toolset registers `search_conversation_history` under a fixed name'),
    'ExaAgent': Collides('its toolset registers `web_search` and friends under fixed names'),
    'ExaSearch': Collides('its toolset registers `web_search` and friends under fixed names'),
    'YouResearch': Collides('its toolset registers `research` and friends under fixed names'),
    'YouSearch': Collides('its toolset registers `web_search` and friends under fixed names'),
}


def _is_capability_class(obj: object) -> TypeGuard[type[AbstractCapability[Any]]]:
    """Whether `obj` is a capability class, and not something that merely looks like one.

    A module's namespace holds type aliases and parameterized generics beside its classes, and on
    Python 3.10 some of those satisfy `inspect.isclass` while `issubclass` then raises on them.
    """
    if not isinstance(obj, type):
        return False
    try:
        return issubclass(obj, AbstractCapability)
    except TypeError:  # pragma: no cover
        return False


def _caused_by_missing_dependency(exc: BaseException) -> bool:
    """Whether `exc` is an uninstalled optional group rather than a broken capability.

    Follows the chain rather than matching a type, because the same condition reaches this walk in
    three shapes: `ModuleNotFoundError` from the import itself, a plain `ImportError` from a module
    that re-raises install guidance, and a `UserError` a capability raises while constructing
    something at import time. The last two are raised inside an `except ImportError` block, so the
    original is on `__cause__` or `__context__` either way.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, ImportError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def _shipped_capability_types() -> tuple[dict[str, type[AbstractCapability[Any]]], list[str]]:
    """Every capability class in `pydantic_ai_harness`, public or not.

    Walks the package rather than reading an export list, so a capability that is never re-exported
    is still covered. Modules whose optional dependency group is not installed are skipped -- the
    import error means the capability could not have been used either.
    """
    found: dict[str, type[AbstractCapability[Any]]] = {}
    skipped: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for module_info in pkgutil.walk_packages(pydantic_ai_harness.__path__, f'{pydantic_ai_harness.__name__}.'):
            try:
                module = importlib.import_module(module_info.name)
            except Exception as exc:
                if not _caused_by_missing_dependency(exc):
                    # A real failure -- a broken module, a bad constant -- has to surface.
                    # Swallowing it drops the capability from `found`, and the exhaustiveness check
                    # below then passes because the capability it should have caught is invisible
                    # rather than absent. That is not hypothetical: it is how `AWSLambdaDurability`
                    # reached this branch with no policy entry while this test was green locally.
                    raise
                skipped.append(module_info.name)
                continue
            for obj in vars(module).values():
                if _is_capability_class(obj) and obj.__module__.startswith(pydantic_ai_harness.__name__):
                    found[obj.__name__] = obj
    return found, skipped


def test_the_capability_walk_hides_a_missing_extra_and_nothing_else(monkeypatch: pytest.MonkeyPatch) -> None:
    """The walk may ignore an uninstalled optional group, and must not ignore anything else.

    Both halves have been wrong. Catching every exception let `AWSLambdaDurability` vanish from the
    discovered set, so the exhaustiveness check passed against a set that could not contain the
    capability it existed to catch. Catching only `ModuleNotFoundError` then broke every slim lane,
    because a module guarding its optional dependency re-raises install guidance as a plain
    `ImportError`. `ImportError` covers both, and stops there.
    """
    real_import = importlib.import_module

    def raise_for(target: str, exc: Exception) -> Callable[..., Any]:
        def fake(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == target:
                raise exc
            return real_import(name, *args, **kwargs)

        return fake

    # Any module the walk visits will do; the class it defines is re-exported elsewhere, so what
    # is under test is that the *import failure* is recorded as skipped rather than propagated.
    a_capability_module = 'pydantic_ai_harness.subagents._capability'

    def wrapped_in_user_error() -> Exception:
        """What a capability raises when it builds something needing a missing extra at import.

        Raised and caught rather than constructed, because `__context__` is attached by *raising*
        inside an `except` block -- which is exactly the shape the walk has to recognise.
        """
        try:
            try:
                raise ImportError('No module named markdownify')
            except ImportError:
                raise UserError('WebFetch(local=True) requires the `web-fetch` optional group')
        except UserError as user_error:
            return user_error

    # A missing optional group is skipped, in every shape it reaches the walk in.
    for exc in (
        ModuleNotFoundError('no module'),
        ImportError('Please install the `x` package'),
        wrapped_in_user_error(),
    ):
        monkeypatch.setattr(importlib, 'import_module', raise_for(a_capability_module, exc))
        _, skipped = _shipped_capability_types()
        assert a_capability_module in skipped, f'{type(exc).__name__} should be treated as a missing extra'

    # Anything else is a real failure and has to surface rather than shrink the set.
    monkeypatch.setattr(importlib, 'import_module', raise_for(a_capability_module, ValueError('bad constant')))
    with pytest.raises(ValueError, match='bad constant'):
        _shipped_capability_types()


def test_every_capability_declares_a_combine_policy() -> None:
    """A new capability must say what two of it mean before it can ship.

    Without this the answer defaults to whatever `AbstractCapability` does, which is the one
    outcome nobody chose. Add an entry to `COMBINE_POLICY`: `Anonymous` when several per agent is
    normal, `Combines` when it carries a default `id`.
    """
    found, skipped = _shipped_capability_types()
    shipped = set(found)
    declared = set(COMBINE_POLICY)
    assert not (shipped - declared), (
        f'capabilities with no `COMBINE_POLICY` entry: {sorted(shipped - declared)}. '
        'Decide what two of them mean and add an entry.'
    )
    # Only meaningful when every module imported: an optional group that is not installed makes its
    # capabilities look deleted, and the slim CI lane installs none of them.
    if not skipped:
        assert not (declared - shipped), (
            f'`COMBINE_POLICY` names capabilities that no longer exist: {sorted(declared - shipped)}.'
        )


@pytest.mark.parametrize('name', sorted(COMBINE_POLICY))
def test_capability_combine_policy_holds(name: str) -> None:
    """Each capability composes -- or refuses to -- the way its policy says."""
    policy = COMBINE_POLICY[name]
    shipped, _ = _shipped_capability_types()
    if name not in shipped:  # pragma: no cover
        pytest.skip(f'{name} needs an optional dependency group that is not installed')
    capability_type = shipped[name]

    if isinstance(policy, (Anonymous, Collides, Rejected)):
        # `declares_default_id` is the function the resolver itself uses, so this asks the same
        # question rather than a lookalike: a default `id` written in the class body, and nothing
        # else, is what makes two of a capability merge.
        assert not declares_default_id(capability_type), (
            f'{name} is declared `{type(policy).__name__}` but its class declares a default id '
            f'{getattr(capability_type, "id", None)!r}, so two of them would merge'
        )
        return

    assert declares_default_id(capability_type), (
        f'{name} is declared `Combines` but its class declares no default id, so two never meet'
    )
    first, second = policy.make()
    assert first.id is not None and first.id == second.id, (
        f'{name} is declared `Combines` but two instances do not share an id'
    )
    policy.check(type(first).combine([first, second]))


# Asyncio only: `Agent.run` reaches `asyncio.create_task` for its lifecycle hooks.
@pytest.mark.parametrize('anyio_backend', ['asyncio'])
@pytest.mark.parametrize(
    'name', sorted(n for n, p in COMBINE_POLICY.items() if isinstance(p, Collides) and p.make is not None)
)
async def test_two_of_a_colliding_capability_still_raise(name: str, anyio_backend: str) -> None:
    """A `Collides` entry states a fact about the capability, so the fact is checked.

    Each pair here is built with *different* configuration -- two roots, two working directories --
    because that is exactly the shape the reasons in this table used to claim was supported. It is
    not: the toolset's tool names are fixed, so the second registration conflicts with the first.
    """
    shipped, _ = _shipped_capability_types()
    if name not in shipped:  # pragma: no cover
        pytest.skip(f'{name} needs an optional dependency group that is not installed')
    policy = COMBINE_POLICY[name]
    assert isinstance(policy, Collides) and policy.make is not None

    first, second = policy.make(shipped[name])
    agent = Agent(TestModel(), capabilities=[first, second])
    with pytest.raises(UserError, match='conflicts with existing tool'):
        await agent.run('hello')


@pytest.mark.parametrize(
    ('field_name', 'first_kwargs', 'second_kwargs'),
    [
        pytest.param(
            'shared_capabilities', {'shared_capabilities': [Thinking(effort='high')]}, {}, id='shared_capabilities'
        ),
        pytest.param('inherit_tools', {'inherit_tools': True}, {'inherit_tools': False}, id='inherit_tools'),
        pytest.param('tool_name', {'tool_name': 'delegate_task'}, {'tool_name': 'ask_specialist'}, id='tool_name'),
        pytest.param('forward_usage', {'forward_usage': False}, {'forward_usage': True}, id='forward_usage'),
        pytest.param('tool_retries', {'tool_retries': 5}, {'tool_retries': 2}, id='tool_retries'),
        pytest.param('contain_errors', {'contain_errors': True}, {'contain_errors': False}, id='contain_errors'),
        # `None` disables disk loading entirely. The generic merge reads `None` as "not stated" and
        # would take the other value, re-enabling it in *either* order -- so an explicit disable
        # never survived. Composing only the roster is what makes that impossible.
        pytest.param('agent_folders', {'agent_folders': None}, {'agent_folders': 'agents'}, id='agent_folders'),
    ],
)
def test_sub_agents_compose_the_roster_and_nothing_else(
    field_name: str, first_kwargs: dict[str, Any], second_kwargs: dict[str, Any]
) -> None:
    """Only `agents` and `models` merge; every other field has to already agree.

    `SubAgents` has fifteen public fields and all but those two say *how* the delegates run rather
    than who they are, so merging one applies one harness's policy to the other's sub-agents. An
    allow-list is what keeps that true for fields added later: they are refused until someone
    decides, rather than merged by a default nobody chose.
    """
    first = SubAgents[Any](agents=[SubAgent(_child('alpha'), description='alpha')], **first_kwargs)
    second = SubAgents[Any](agents=[SubAgent(_child('beta'), description='beta')], **second_kwargs)

    with pytest.raises(UserError, match=f'disagree on {field_name!r}'):
        SubAgents.combine([first, second])


def test_merging_reuses_the_delegates_already_loaded_from_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A merge must not re-read `agent_folders`, because it reads them relative to the *cwd*.

    Both inputs resolved their disk delegates when they were constructed. Re-running
    `__post_init__` to rebuild the roster would load them again from wherever the process happens
    to be standing now -- so a merge after a `chdir` could hand the run a different set of
    delegates than either capability was built with, and re-invoke `tool_resolver` besides.
    """
    project = tmp_path / 'project'
    folder = project / '.agents' / 'agents'  # `<root>/.agents/<agent_folders>/`
    folder.mkdir(parents=True)
    (folder / 'helper.md').write_text('---\nname: helper\ndescription: helps\n---\n\nBe helpful.\n', encoding='utf-8')
    monkeypatch.chdir(project)

    first = SubAgents[Any](agents=[SubAgent(_child('alpha'), description='alpha')])
    second = SubAgents[Any](agents=[SubAgent(_child('beta'), description='beta')])
    assert 'helper' in first._by_name, 'the disk delegate is picked up at construction'  # pyright: ignore[reportPrivateUsage]

    # The folder is gone by the time the two are combined, which is what a `chdir` amounts to.
    monkeypatch.chdir(tmp_path)

    merged = SubAgents.combine([first, second])
    assert isinstance(merged, SubAgents)
    assert set(merged._by_name) == {'alpha', 'beta', 'helper'}, (  # pyright: ignore[reportPrivateUsage]
        'the disk delegate survives the merge because it is reused, not reloaded'
    )


def _child(name: str) -> Agent[Any, str]:
    return Agent(TestModel(), name=name)


@pytest.mark.skipif(
    find_spec('ddgs') is None or find_spec('markdownify') is None,
    reason='`Researcher` needs the `researcher` optional group.',
)
async def test_coder_and_researcher_compose() -> None:
    """The composition #7781 was filed for: two packaged harnesses on one agent.

    Both build a `ToolOutputLimits` and both delegate, so before `combine` they collided twice --
    on the capability id, and then on the `delegate_task` tool name.
    """
    tree = CombinedCapability([Coder[Any](), Researcher[Any]()])
    counts = Counter(type(leaf).__name__ for leaf in leaf_capabilities(tree))
    assert counts['ToolOutputLimits'] == 2
    assert counts['SubAgents'] == 2

    # One layer: both harnesses are on the same agent, which is what makes them merge rather
    # than one replacing the other.
    combined = combine_duplicate_capabilities(tree, [tree.capabilities])

    leaves = leaf_capabilities(combined)
    merged_counts = Counter(type(leaf).__name__ for leaf in leaves)
    assert merged_counts['ToolOutputLimits'] == 1
    assert merged_counts['SubAgents'] == 1
    # Neither harness loses a delegate: the rosters union under one `delegate_task` tool.
    sub_agents = next(leaf for leaf in leaves if isinstance(leaf, SubAgents))
    assert [entry.agent.name for entry in sub_agents.agents] == ['explorer', 'researcher']

    # And what the model is told, not just what the field holds. `_by_name` is a `compare=False`
    # cache built in `__post_init__`, so a merge that unions `agents` without rebuilding it leaves
    # the roster reading as composed while the delegate tool offers only the last harness's agents.
    instructions = sub_agents.get_instructions()
    assert isinstance(instructions, str)
    assert 'explorer' in instructions and 'researcher' in instructions, (
        'the delegate tool is built from the derived roster, so merging has to rebuild it'
    )
