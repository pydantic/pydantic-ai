"""What two of a harness capability on one agent mean.

Every capability this package ships has a case in `tests/harness/capability_cases.py` whose `combine`
says what two of it mean, and `test_every_capability_has_a_case` fails when one is missing. That module
describes the answers and what picks between them; this one checks that each holds.

The core half of this lives in `pydantic-ai`'s `tests/test_capability_combine.py`.

Two of the names imported below are private to pydantic-ai, which is right: the duplicate-resolution
pipeline is internal and no code in this package needs it. This file reaches in anyway rather than
reimplementing the two questions the resolver asks -- a lookalike would drift from the real answer
silently, which is the one thing the policy table exists to prevent. Both packages change in the
same pull request, so a rename fails this file alongside the change that made it.
"""

from __future__ import annotations

import importlib
from collections import Counter
from collections.abc import Callable
from importlib.util import find_spec
from typing import Any

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import CombinedCapability, Thinking
from pydantic_ai.capabilities.abstract import (
    _combine_duplicate_capabilities as combine_duplicate_capabilities,  # pyright: ignore[reportPrivateUsage]
    _declares_default_id as declares_default_id,  # pyright: ignore[reportPrivateUsage]
    leaf_capabilities,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness import (
    CodeMode,
    Coder,
    Researcher,
    SubAgent,
    SubAgents,
    SummarizingCompaction,
    SystemReminders,
)
from pydantic_ai_harness.sub_agents import AgentOverride
from pydantic_ai_harness.system_reminders import LLMReminder

from ..capability_cases import Anonymous, Collides, Narrows, Rejected, shipped_capability_types
from .capability_cases import CASES, COMBINE_ROOT_A, HARNESS_CAPABILITY_PACKAGES


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
    a_capability_module = 'pydantic_ai_harness.sub_agents._capability'

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
        _, skipped = shipped_capability_types(HARNESS_CAPABILITY_PACKAGES)
        assert a_capability_module in skipped, f'{type(exc).__name__} should be treated as a missing extra'

    # Anything else is a real failure and has to surface rather than shrink the set.
    monkeypatch.setattr(importlib, 'import_module', raise_for(a_capability_module, ValueError('bad constant')))
    with pytest.raises(ValueError, match='bad constant'):
        shipped_capability_types(HARNESS_CAPABILITY_PACKAGES)


@pytest.mark.parametrize('name', sorted(CASES))
def test_capability_combine_policy_holds(name: str) -> None:
    """Each capability composes -- or refuses to -- the way its policy says."""
    policy = CASES[name].combine
    shipped, _ = shipped_capability_types(HARNESS_CAPABILITY_PACKAGES)
    if name not in shipped:  # pragma: lax no cover
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
        f'{name} is declared `{type(policy).__name__}` but its class declares no default id, so two never meet'
    )
    if isinstance(policy, Narrows):
        first, second = policy.make(capability_type)
        assert first.id is not None and first.id == second.id
        assert capability_type.combine([first, first]) is first
        with pytest.raises(UserError, match='disagree on'):
            capability_type.combine([first, second])
        return
    first, second = policy.make()
    assert first.id is not None and first.id == second.id, (
        f'{name} is declared `Combines` but two instances do not share an id'
    )
    policy.check(type(first).combine([first, second]))


# Asyncio only: `Agent.run` reaches `asyncio.create_task` for its lifecycle hooks.
@pytest.mark.parametrize('anyio_backend', ['asyncio'])
@pytest.mark.parametrize(
    'name', sorted(n for n, c in CASES.items() if isinstance(c.combine, Collides) and c.combine.make is not None)
)
async def test_two_of_a_colliding_capability_still_raise(name: str, anyio_backend: str) -> None:
    """A `Collides` entry states a fact about the capability, so the fact is checked.

    Each pair here is built with *different* configuration -- two roots, two timeouts --
    because that is exactly the shape the reasons in this table used to claim was supported. It is
    not: the toolset's tool names are fixed, so the second registration conflicts with the first.
    """
    shipped, _ = shipped_capability_types(HARNESS_CAPABILITY_PACKAGES)
    if name not in shipped:  # pragma: no cover
        pytest.skip(f'{name} needs an optional dependency group that is not installed')
    policy = CASES[name].combine
    assert isinstance(policy, Collides) and policy.make is not None

    first, second = policy.make(shipped[name])
    agent = Agent(TestModel(), deps_type=type(None), capabilities=[first, second])
    with pytest.raises(UserError, match='conflicts with existing tool'):
        await agent.run('hello', workspace=LocalWorkspaceBackend(COMBINE_ROOT_A))


@pytest.mark.parametrize(
    ('field_name', 'first_kwargs', 'second_kwargs'),
    [
        pytest.param(
            'shared_capabilities', {'shared_capabilities': [Thinking(effort='high')]}, {}, id='shared_capabilities'
        ),
        pytest.param(
            'inherit_tools',
            {'inherit_tools': True},
            {'inherit_tools': False},
            id='inherit_tools',
            marks=pytest.mark.filterwarnings('ignore::pydantic_ai_harness.HarnessDeprecationWarning'),
        ),
        pytest.param('tool_name', {'tool_name': 'delegate_task'}, {'tool_name': 'ask_specialist'}, id='tool_name'),
        pytest.param('forward_usage', {'forward_usage': False}, {'forward_usage': True}, id='forward_usage'),
        pytest.param('tool_retries', {'tool_retries': 5}, {'tool_retries': 2}, id='tool_retries'),
        pytest.param('contain_errors', {'contain_errors': True}, {'contain_errors': False}, id='contain_errors'),
        # `None` disables disk loading entirely. The generic merge reads `None` as "not stated" and
        # would take the other value, re-enabling it in *either* order -- so an explicit disable
        # never survived. Composing only the roster is what makes that impossible.
        pytest.param('agent_folders', {'agent_folders': None}, {'agent_folders': 'agents'}, id='agent_folders'),
        pytest.param(
            'agent_overrides',
            {'agent_overrides': {'alpha': AgentOverride(model=TestModel(settings={'temperature': 0.0}))}},
            {'agent_overrides': {'alpha': AgentOverride(model=TestModel(settings={'temperature': 1.0}))}},
            id='agent_overrides_model_settings',
        ),
        pytest.param(
            'shared_capabilities',
            {
                'shared_capabilities': [
                    SummarizingCompaction(model=TestModel(settings={'temperature': 0.0}), max_messages=10)
                ]
            },
            {
                'shared_capabilities': [
                    SummarizingCompaction(model=TestModel(settings={'temperature': 1.0}), max_messages=10)
                ]
            },
            id='shared_capabilities_model_settings',
        ),
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


def test_code_mode_merges_after_a_run_changed_its_stats() -> None:
    """Two identical `CodeMode` still merge once one has been used: its counters are state, not configuration.

    Not a VCR test: `combine` makes no request.
    """
    first, second = CodeMode[Any](), CodeMode[Any]()
    first.speculation_stats.launched += 1

    assert CodeMode.combine([first, second]) is first


def test_system_reminders_keep_two_llm_reminders_whose_models_differ_only_in_settings() -> None:
    """Reminders asking two differently configured models are two reminders, so both keep firing.

    Not a VCR test: `combine` makes no request.
    """
    first = LLMReminder[Any](interval=10, model=TestModel(settings={'temperature': 0.0}))
    second = LLMReminder[Any](interval=10, model=TestModel(settings={'temperature': 1.0}))

    merged = SystemReminders.combine(
        [
            SystemReminders[Any](dynamic_reminders=[first]),
            SystemReminders[Any](dynamic_reminders=[second]),
        ]
    )

    assert isinstance(merged, SystemReminders)
    assert list(merged.dynamic_reminders) == [first, second]


def _child(name: str) -> Agent[Any, str]:
    return Agent(TestModel(), name=name)


def test_coder_composes_with_plain_sub_agents() -> None:
    tree = CombinedCapability([Coder[Any](), SubAgents[Any](agents=[SubAgent(_child('worker'))])])
    combined = combine_duplicate_capabilities(tree, [tree.capabilities])
    sub_agents = [leaf for leaf in leaf_capabilities(combined) if isinstance(leaf, SubAgents)]
    assert len(sub_agents) == 1
    assert sub_agents[0].include_self
    assert [entry.agent.name for entry in sub_agents[0].agents] == ['worker']


@pytest.mark.skipif(
    find_spec('ddgs') is None or find_spec('markdownify') is None,
    reason='`Researcher` needs the `researcher` optional group.',
)
async def test_coder_and_researcher_compose() -> None:
    """Coder's private limits coexist with Researcher's independent tools and limits.

    Both harnesses bring a `SubAgents`, so this also covers them merging into one roster: Researcher's
    delegate, plus the running agent that Coder delegates to.
    """
    tree = CombinedCapability([Coder[Any](), Researcher[Any]()])
    combined = combine_duplicate_capabilities(tree, [tree.capabilities])
    leaves = leaf_capabilities(combined)
    counts = Counter(type(leaf).__name__ for leaf in leaves)
    assert counts['ToolOutputLimits'] == 1
    assert counts['_BoundToolOutputs'] == 1
    assert counts['SubAgents'] == 1
    sub_agents = next(leaf for leaf in leaves if isinstance(leaf, SubAgents))
    assert [entry.agent.name for entry in sub_agents.agents] == ['researcher']
    assert sub_agents.include_self
