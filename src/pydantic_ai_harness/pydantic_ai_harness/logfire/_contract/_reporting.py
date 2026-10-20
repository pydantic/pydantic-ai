"""Saying out loud that Logfire shows one thing and the agent does another."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

OnUnmatched: TypeAlias = Literal['ignore', 'warn', 'error']
"""What to do with a published entry that reaches nothing in this deployment.

That is an instruction `id` no block carries (or that only a dynamic block carries), a tool override
whose `name` -- and `toolset`, when it sets one -- matches no tool the agent advertises, a `settings`
key the contract or the adapter cannot apply, and a whole section this adapter or this release has no
way to act on. Each is a place where Logfire shows one thing and the agent does another.

- `'warn'` (the default everywhere) emits a `UserWarning` once per process per message.
- `'error'` raises `UnmatchedConfigError` naming every
  issue at once, failing the request.
- `'ignore'` applies nothing and says nothing.

The policy is applied in one place, by
`report_issues`, rather than inside each apply
helper: an adapter plans every section and then reports, so `'error'` fails on everything the request
would have got wrong rather than on whichever section happened to be planned first.

Warning rather than raising is the default because tool availability is dynamic: one config is
applied across deployments that need not all install the same tools, and an agent can advertise
different tools from one request to the next, so an entry that reaches nothing here may be exactly
right somewhere else. `'error'` is for the deployment that would rather stop than run with part of
its published config silently unapplied.
"""

_warned: set[str] = set()
"""Messages already emitted in this process; see `warn_dropped`."""


def warn_dropped(message: str) -> None:
    """Surface a dropped managed value once per process.

    A drop means Logfire shows one thing and the agent does another, which has to be visible. But the
    config is resolved on every single request, so warning per drop would bury that signal under its
    own repetition. Deduplicating on the message rather than on the field costs nothing in clarity --
    each message names its subject and the offending value -- and still lets a *different*
    unrecognized value surface later, which a per-field guard would swallow. A concurrent first
    request can at worst duplicate the warning, which is not worth a lock.
    """
    if message in _warned:
        return
    _warned.add(message)
    try:
        warnings.warn(message)
    except Exception:
        # Emitted filter-independently, because every drop this package warns about happens while a
        # managed config is being read or applied, and the contract is that neither can take a run
        # down: `resolve` promises an unreachable Logfire leaves the agent running on its code.
        # `warnings.warn` escalates to an exception under `-W error` / `filterwarnings = ["error"]`,
        # and escalation here does not make anything stricter -- it makes it worse. A validation drop
        # raised inside `Variable.get` is caught by logfire's own resolution fallback, which turns a
        # value that parsed with one field dropped into "nothing published", silently un-managing
        # every section that did apply. And a raise from the fallback warning itself escapes
        # `_resolution`'s `except` and reaches the agent as the crash that `except` exists to prevent.
        #
        # Being strict about unmatched entries is `on_unmatched='error'`, which raises
        # `UnmatchedConfigError` naming all of them. That is the sanctioned mechanism; this is a
        # diagnostic. `logfire.variables` makes the same call for the same reason -- see
        # `_emit_resolution_warning` there.
        pass


def reset_warned_messages() -> None:
    """Forget which messages have been warned about. Intended for tests only."""
    _warned.clear()


ApplyIssueReason: TypeAlias = Literal[
    'unknown-id',
    'dynamic-id',
    'oversized-text',
    'unknown-tool',
    'unknown-parameter',
    'no-patchable-schema',
    'rename-collision',
    'unknown-setting',
    'unsupported-setting',
    'unrepresentable-timeout',
    'unknown-section',
    'unsupported-section',
    'duplicate-entry',
    'dropped-by-provider',
]
"""Why one published entry did not reach the request, as a stable string an adapter can branch on.

Instructions:

- `'unknown-id'` -- an `id` this request assembles no block under.
- `'dynamic-id'` -- an `id` only a block the framework recomputes per request carries, which cannot
  be replaced or dropped without pinning or removing that computation.
- `'oversized-text'` -- text past the contract's per-request budget, which is refused rather than
  truncated: half a prompt is not a smaller version of the prompt.

Tools:

- `'unknown-tool'` -- a `name` (narrowed by `toolset` when the entry sets one) no advertised tool has.
- `'unknown-parameter'` -- a `parameters` key the tool's schema has no top-level property for.
- `'no-patchable-schema'` -- the tool, or that one property, has no object schema to patch a
  description into at all.
- `'rename-collision'` -- a `new_name` another advertised tool, or a name the adapter reserved,
  already answers to. The rename is dropped and the tool keeps its code-side name; the same entry's
  other patches still apply.

Settings:

- `'unknown-setting'` -- a `settings` key this version of the contract has no field for, which a
  newer Logfire UI can write.
- `'unsupported-setting'` -- a canonical key this adapter declared it cannot lower; see
  `AgentSupport.settings`.
- `'unrepresentable-timeout'` -- a `timeout` that is not a request budget: negative, not finite, or
  past `MAX_TIMEOUT_SECONDS`.

Any section:

- `'unknown-section'` -- a top-level key this release has no section for. Every section is optional
  and unknown keys cost nothing, which is what lets a future section be written against an older SDK
  -- but the drop has to be audible, or the first person to publish one gets a silently degraded
  agent.
- `'unsupported-section'` -- a section this release understands and this adapter cannot apply; see
  `AgentSupport.sections`.
- `'duplicate-entry'` -- the same instruction `id`, or the same `(toolset, name)`, written twice. The
  first is applied and the rest are not.
- `'dropped-by-provider'` -- a setting the adapter did forward and the provider or its SDK dropped;
  see `ApplyIssue.dropped_by_provider`.

These are the runtime decisions: what a *valid* published value did not reach in *this* deployment,
on *this* request. They are deliberately not the parser's compatibility warnings -- an entry a newer
UI wrote that this release cannot understand -- which are about the value rather than the request,
warn once per process from validation, and never take an `OnUnmatched` policy.
"""


@dataclass(frozen=True)
class ApplyIssue:
    """One published entry that reached nothing, with the path that says which one.

    Returned by the apply helpers, which report nothing themselves: an adapter collects the issues of
    every section it planned and hands them to
    `report_issues` once. That is what makes the
    return value load-bearing rather than a duplicate of a warning already emitted, and what lets
    `'error'` fail on everything a request got wrong instead of on the first section planned.

    The fields are a path, and only the ones that apply to `reason` are set -- `tool` and `parameter`
    on a parameter patch, `instruction_id` on an instruction entry, `setting` on a settings key -- so
    an adapter never has to parse `message` to learn what an issue was about.
    """

    section: str
    """Which section of the config the issue is about.

    One of the four `Section` names, except for
    `'unknown-section'`, where it is the unrecognized top-level key itself: a key this release has no
    section for has no section name to give, and naming the key is what makes the report actionable.
    A plain string for that reason, in both cores, rather than a `Section` that half the reasons
    would have to lie about.
    """
    reason: ApplyIssueReason
    """Why it was not applied; see `ApplyIssueReason`."""
    message: str
    """What the policy reports: what was published, and what was not applied."""
    instruction_id: str | None = None
    """The instruction `id` the entry addressed, for the instruction reasons."""
    destination: str | None = None
    """The instruction destination the entry named, when it named one."""
    toolset: str | None = None
    """The toolset the entry named or the tool came from, when either has one."""
    tool: str | None = None
    """The code-side tool name the entry named, for the tool reasons."""
    parameter: str | None = None
    """The parameter the entry patched, for `'unknown-parameter'` and `'no-patchable-schema'`."""
    setting: str | None = None
    """The canonical settings key the issue is about, for the settings reasons."""

    @classmethod
    def dropped_by_provider(cls, setting: str, detail: str) -> ApplyIssue:
        """One setting the adapter forwarded and the provider, or its own SDK, did not apply.

        The one issue the core cannot find for itself: it is discovered *after* the request, by an
        adapter reading whatever its framework reports -- the Vercel AI SDK's `result.warnings`, a
        provider's own "unsupported parameter" note. Those shapes differ per SDK and the core knows
        none of them, so it takes the two things every one of them carries: which canonical setting,
        and what the SDK said about it.

        ```python skip-run="true" skip-reason="illustrative-fragment"
        control.report(*(ApplyIssue.dropped_by_provider('top_k', w.detail) for w in result.warnings))
        ```

        Args:
            setting: The canonical settings key that did not reach the model.
            detail: What the provider or SDK said, in its own words, as one clause.
        """
        return cls(
            section='settings',
            reason='dropped-by-provider',
            setting=setting,
            message=(
                f'Managed agent config sets {setting!r}, which the provider did not apply -- {detail}; '
                'that key had no effect on the request.'
            ),
        )


class UnmatchedConfigError(ValueError):
    """Raised by `on_unmatched='error'` for everything one request could not apply.

    A `ValueError` so a deployment that was catching one keeps catching this, and a class of its own
    so an adapter can tell a config the deployment asked to fail on from a model or tool failure --
    and translate it into its own framework's error type without restating a single message:

    ```python skip-run="true" skip-reason="illustrative-fragment"
    try:
        control.report(*issues)
    except UnmatchedConfigError as exc:
        raise MyFrameworkError(str(exc)) from exc
    ```

    Raised once for a whole request, with every issue's message in `str(exc)` and the issues
    themselves on `issues`, so an adapter that reports a kind it has never heard of still reports it
    faithfully.
    """

    def __init__(self, message: str, issues: Sequence[ApplyIssue] = ()) -> None:
        super().__init__(message)
        self.issues: tuple[ApplyIssue, ...] = tuple(issues)
        """Every issue this request could not apply, in the order they were planned.

        Empty when the error is raised with a message alone.
        """


def report_issues(policy: OnUnmatched, issues: Sequence[ApplyIssue]) -> None:
    """Apply one `OnUnmatched` policy to every issue a request's planning produced.

    One call for a whole request, which is the point: the apply helpers plan and report nothing, so
    `'error'` raises after every section has been planned, naming all of it, rather than on the first
    entry of the first section -- which used to mean the strictest policy reported the least.

    Never called from validation, for two reasons. Validation runs inside Logfire's resolution, which
    turns any exception into a fallback to the code-defined agent, so an `'error'` raised there would
    be swallowed and un-manage the whole config instead of stopping the request. And the same
    validation builds the code baseline, where a key this contract has no field for is the agent's
    own provider-specific setting rather than anything anyone published.

    An adapter with its own error type translates it without restating a message:

    ```python skip-run="true" skip-reason="illustrative-fragment"
    try:
        report_issues(self.on_unmatched, issues)
    except UnmatchedConfigError as exc:
        raise MyFrameworkError(str(exc)) from exc
    ```

    Raises:
        UnmatchedConfigError: naming every issue, when `policy` is `'error'`.
    """
    if not issues:
        return
    if policy == 'error':
        raise UnmatchedConfigError('\n'.join(issue.message for issue in issues), issues)
    if policy == 'warn':
        for issue in issues:
            warn_dropped(issue.message)
