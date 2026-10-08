"""Risk and governance: what agents actually do with the shell and MCP tools, as advisor notes with a control.

Risky commands are found by pattern, then each is classified (cached per call): did the developer ask for it in that
turn or the one before (`requested`) or did the agent choose it (`unprompted`), and did it hit a protected or shared
target (the default branch, a remote host, outside the workspace, secrets)? A requested action on the developer's own
work is normal and only counted. What's left is "flagged": agents acting on their own (suggest an instruction plus an
`ask` rule) or touching protected targets (suggest `deny` for clearly destructive ones, else `ask`). Never from one
developer or one session. The drafting LLM only sees aggregates of flagged calls, never raw spans, and every rule it
proposes is re-measured against the actual calls, so counts and evidence are measured, not claimed.
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Literal

from logfire.query_client import AsyncLogfireQueryClient
from pydantic import BaseModel, Field, TypeAdapter, field_validator

from pydantic_ai import Agent, ModelRetry, RunContext

from . import __version__
from .fetch import _SESSIONS, IDENTITY, NOT_TEST, OVERLAP
from .llm_cache import USAGE, run_cached
from .models import (
    Evidence,
    McpAllow,
    PolicyMatch,
    PolicyRule,
    Proposal,
    UserPrompt,
    clean_text,
    daily_trend,
    redact_secrets,
)
from .patterns import leaked_identifiers_in
from .scope import measure_scope

TOOL_CALLS_SQL = f"""
SELECT r.trace_id, r.span_id, r.start_timestamp, r.attributes->>'gen_ai.tool.name' AS tool,
       r.attributes->>'gen_ai.tool.call.arguments' AS arguments,
       {IDENTITY}
FROM records r
LEFT JOIN ({_SESSIONS}) s ON r.trace_id = s.trace_id
WHERE r.service_name = 'pydantic-clai2' AND r.span_name LIKE 'execute_tool %'
  AND r.attributes->>'gen_ai.tool.name' NOT IN ({{builtin_non_shell}})
  AND {NOT_TEST}
ORDER BY r.start_timestamp DESC
"""

SHELL_TOOLS = {'shell', 'bash', 'run_shell', 'terminal'}
BUILTIN_TOOLS = SHELL_TOOLS | {
    *('run_code', 'read_file', 'edit_file', 'write_file', 'grep', 'list_files', 'delegate_task'),
    *('ask_user_question', 'load_capability', 'read_clai_customization_guide', 'rank_relevance', 'read_pyai_docs'),
}

# Deterministic first pass: which commands are worth showing the LLM as risky at all.
RISKS: dict[str, re.Pattern[str]] = {
    'force-push': re.compile(r'\bgit\s+push\b[^;&|]*(\s--force\b|\s-f\b|\s--force-with-lease\b|\s\+\S+)'),
    'push-to-default-branch': re.compile(r'\bgit\s+push\b[^;&|]*\s(origin\s+)?(HEAD:)?(main|master)\b'),
    'hard-reset': re.compile(r'\bgit\s+reset\b[^;&|]*--hard\b'),
    'git-clean': re.compile(r'\bgit\s+clean\b[^;&|]*-[a-zA-Z]*f'),
    'discard-changes': re.compile(r'\bgit\s+(checkout\s+--\s|checkout\s+\.|restore\s+(--staged\s+)?\.)'),
    'delete-branch': re.compile(r'\bgit\s+branch\s+-D\b|\bgit\s+push\b[^;&|]*\s--delete\b'),
    'skip-hooks': re.compile(r'--no-verify\b'),
    'rm-rf': re.compile(r'\brm\s+-[a-zA-Z]*r[a-zA-Z]*f|\brm\s+-[a-zA-Z]*f[a-zA-Z]*r'),
    'secrets-files': re.compile(r'(^|[\s/"\'])\.env(\.[\w-]+)?\b|credentials|\.ssh/|id_rsa|\.aws/|\.netrc|\.pypirc'),
    'secret-env-vars': re.compile(r'\$\{?[A-Z0-9_]*(TOKEN|SECRET|PASSWORD|API_?KEY)[A-Z0-9_]*'),
    'pipe-to-shell': re.compile(r'\b(curl|wget)\b[^;&]*\|\s*(sudo\s+)?(ba|z)?sh\b'),
    'sudo': re.compile(r'(^|[;&|]\s*)sudo\b'),
    'chmod-777': re.compile(r'\bchmod\s+(-R\s+)?777\b'),
    'admin-merge': re.compile(r'\bgh\s+pr\s+merge\b[^;&|]*--admin\b'),
    'destructive-sql': re.compile(r'(?i)\b(drop\s+(table|database)|truncate\s+table)\b'),
    'kill-all': re.compile(r'\b(pkill|killall)\b|\bkill\s+-9\s+-1\b'),
    'remote-hosts': re.compile(
        r'(^|[;&|(]\s*)(ssh|scp)\s|\bkubectl\s+(apply|delete|exec|rollout|scale|edit|patch)\b'
        r'|\bterraform\s+(apply|destroy)\b|\bhelm\s+(install|upgrade|uninstall|delete)\b'
    ),
    'skip-tests': re.compile(r'\bSKIP=\S+|\bpytest\b[^;&|]*(--deselect\b|-k\s+["\']?not\b)'),
}


@dataclass
class ToolCall:
    trace_id: str
    span_id: str
    timestamp: datetime
    tool: str
    user: str
    session_id: str
    command: str | None
    team: str | None = None
    repo_slug: str | None = None


@dataclass
class Group:
    """Calls sharing a risk category or MCP-ish tool, with who made them."""

    key: str
    calls: list[ToolCall] = field(default_factory=list[ToolCall])

    @property
    def users(self) -> set[str]:
        return {c.user for c in self.calls}


async def fetch_tool_calls(
    read_token: str, *, base_url: str, since: datetime, store_path: Path | None = None
) -> list[ToolCall]:
    """Tool calls in the window. With `store_path`, only calls newer than the stored watermark are queried."""
    stored: dict[str, dict[str, Any]] = {}
    watermark: datetime | None = None
    if store_path and store_path.exists():
        data = json.loads(store_path.read_text())
        stored = {r['span_id']: r for r in data['rows']}
        watermark = datetime.fromisoformat(data['watermark']) if data.get('watermark') else None
    query_since = max(since, watermark - OVERLAP) if watermark else since
    async with AsyncLogfireQueryClient(read_token, base_url=base_url, timeout=180) as client:
        # File and code tools carry no policy signal, and the query API caps rows at 10k: leave them out.
        builtin = ', '.join(f"'{t}'" for t in sorted(BUILTIN_TOOLS - SHELL_TOOLS))
        sql = TOOL_CALLS_SQL.format(builtin_non_shell=builtin)
        rows = (await client.query_json_rows(sql, min_timestamp=query_since, limit=10_000))['rows']
        if len(rows) == 10_000:
            print('warning: hit the 10k row cap; only the most recent tool calls were mined')
    for r in rows:
        r['start_timestamp'] = _parse_time(r['start_timestamp']).isoformat()
        stored[r['span_id']] = r
    rows = [r for r in stored.values() if _parse_time(r['start_timestamp']) >= since]
    if store_path:
        latest = max((_parse_time(r['start_timestamp']) for r in rows), default=watermark)
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_text(json.dumps({'watermark': latest.isoformat() if latest else None, 'rows': rows}))
    emails = {r['host']: r['user_email'] for r in rows if r.get('user_email') and r.get('host')}
    calls: list[ToolCall] = []
    for r in sorted(rows, key=lambda r: r['start_timestamp']):
        user = r.get('user_email') or emails.get(r.get('host')) or f'host:{r.get("host")}'
        calls.append(
            ToolCall(
                trace_id=r['trace_id'],
                span_id=r['span_id'],
                timestamp=_parse_time(r['start_timestamp']),
                tool=r['tool'] or '',
                user=user,
                session_id=r['session_id'] or r['trace_id'],
                command=_command(r['tool'], r.get('arguments')),
                team=r.get('team'),
                repo_slug=r.get('repo_slug'),
            )
        )
    return calls


def _parse_time(value: str | datetime) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(value.replace('Z', '+00:00'))


def _command(tool: str | None, arguments: Any) -> str | None:
    if tool not in SHELL_TOOLS or not arguments:
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            try:  # Logfire stores the Python repr of the arguments dict
                arguments = ast.literal_eval(arguments)
            except (ValueError, SyntaxError):
                return arguments
    if isinstance(arguments, dict):
        command = arguments.get('command') or arguments.get('cmd')
        return command if isinstance(command, str) else None
    return None


_SEGMENT_SPLIT = re.compile(r'\s*(?:&&|\|\||;|\n|(?<!\|)\|(?!\|))\s*')


def matching_segment(command: str, glob: str) -> str | None:
    """The part of a compound command a rule glob matches, or None.

    Globs are matched per segment (`a && b; c`), so `git push ...; git worktree remove --force` does not count as a
    force push. A glob that itself contains `|` (e.g. `*curl*|*sh*`) is matched against the whole command.
    """
    if '|' in re.sub(r'\[[^\]]*\]', '', glob):  # a literal pipe, not one inside a `[...]` class
        return command if fnmatchcase(command, glob) else None
    return next((seg for seg in _SEGMENT_SPLIT.split(command) if seg and fnmatchcase(seg, glob)), None)


def risk_groups(calls: list[ToolCall]) -> list[Group]:
    groups: dict[str, Group] = {}
    for call in calls:
        if call.command is None:
            continue
        for key, pattern in RISKS.items():
            if pattern.search(call.command):
                groups.setdefault(key, Group(key)).calls.append(call)
    return sorted(groups.values(), key=lambda g: (len(g.users), len(g.calls)), reverse=True)


def mcp_groups(calls: list[ToolCall]) -> list[Group]:
    """Non-built-in tools, grouped by name: the LLM infers which server they come from."""
    groups: dict[str, Group] = {}
    for call in calls:
        if call.tool and call.tool not in BUILTIN_TOOLS:
            groups.setdefault(call.tool, Group(call.tool)).calls.append(call)
    return sorted(groups.values(), key=lambda g: g.key)


CLASSIFY_INSTRUCTIONS = """\
You classify risky shell commands coding agents ran, for a company's oversight of its agents. Each comes with what the
developer typed in that turn (`prompt`), the turn before (`previous_prompt`) and the turn after (`next_prompt`).

- `requested`: true only when `prompt` or `previous_prompt` explicitly asked for this action (e.g. "force push it",
  "reset to main", "delete the branch", "ssh into the VM and ..."). A general task ("fix the tests", "ship it") is NOT
  a request for a force push or a hard reset the agent chose along the way.
- `target`:
  - `protected`: a shared or protected resource: the default or a release branch (main, master, release/*), someone
    else's branch, a production or remote host, files outside the project and temp dirs, secrets or credentials.
  - `own`: the developer's own feature branch, worktree or project files.
  - `harmless`: nothing at risk: a temp dir or file the agent itself made, a cache, a process it started, a read-only
    probe that only matched the risk pattern by accident.
- `steered`: true when `next_prompt` pushes back on this action ("why did you force push?", "don't do that").
- `note`: at most 8 words on what the command does to what (no names, paths or secrets).
"""


class CallClass(BaseModel):
    span_id: str
    requested: bool
    target: Literal['protected', 'own', 'harmless']
    steered: bool = False
    note: str = ''

    _clean = field_validator('note')(clean_text)

    @property
    def flagged(self) -> bool:
        """Worth governing: the agent chose it on its own, or it hit a protected target (requested or not).

        A requested action on the developer's own branch or workspace is normal: counted in the oversight stats only.
        """
        return self.target == 'protected' or (not self.requested and self.target != 'harmless')


class _CallClasses(BaseModel):
    calls: list[CallClass]


_classes_adapter = TypeAdapter(dict[str, CallClass])


async def classify_calls(
    calls: list[ToolCall],
    prompts: list[UserPrompt],
    *,
    model: str,
    cache_path: Path | None,
    batch: int = 40,
    concurrency: int = 8,
) -> dict[str, CallClass]:
    """Requested or unprompted, and on what, per risky call. Cached per span id: only new risky calls cost anything."""
    cache: dict[str, CallClass] = (
        _classes_adapter.validate_json(cache_path.read_bytes()) if cache_path and cache_path.exists() else {}
    )
    todo = [c for c in calls if c.span_id not in cache and c.command]
    if todo:
        by_session: dict[str, list[UserPrompt]] = defaultdict(list)
        for p in sorted(prompts, key=lambda p: p.timestamp):
            by_session[p.session_id or p.trace_id].append(p)

        def turn(call: ToolCall) -> dict[str, Any]:
            session = by_session.get(call.session_id, [])
            i = next((n for n, p in reversed(list(enumerate(session))) if p.timestamp <= call.timestamp), None)

            def text(n: int | None, limit: int) -> str | None:
                return redact_secrets(session[n].text[:limit]) if n is not None and 0 <= n < len(session) else None

            return {
                'span_id': call.span_id,
                'command': redact_secrets(' '.join((call.command or '').split()))[:400],
                'prompt': text(i, 400),
                'previous_prompt': text(i - 1, 200) if i is not None else None,
                'next_prompt': text(i + 1 if i is not None else 0, 200),
            }

        agent = Agent(model, output_type=_CallClasses, instructions=CLASSIFY_INSTRUCTIONS, name='fleet_miner_classify')
        semaphore = asyncio.Semaphore(concurrency)

        async def one(chunk: list[ToolCall]) -> None:
            async with semaphore:
                result = await agent.run('Risky calls:\n' + json.dumps([turn(c) for c in chunk], indent=2))
            USAGE.add(result)
            wanted = {c.span_id for c in chunk}
            cache.update({c.span_id: c for c in result.output.calls if c.span_id in wanted})

        todo.sort(key=lambda c: (c.session_id, c.timestamp))
        await asyncio.gather(*(one(todo[i : i + batch]) for i in range(0, len(todo), batch)))
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(_classes_adapter.dump_json(cache, indent=2))
    return {c.span_id: cache[c.span_id] for c in calls if c.span_id in cache}


@dataclass
class Oversight:
    """Per risk category, what agents did: the stats every finding is measured against."""

    calls: int = 0
    requested_own: int = 0
    harmless: int = 0
    unprompted: int = 0
    protected: int = 0
    steered: int = 0
    flagged_users: set[str] = field(default_factory=set[str])
    flagged_sessions: set[str] = field(default_factory=set[str])

    def __str__(self) -> str:
        return (
            f'{self.calls} calls: {self.unprompted} unprompted, {self.protected} on protected targets, '
            f'{self.requested_own} requested on own work, {self.harmless} harmless, {self.steered} steered; '
            f'flagged by {len(self.flagged_users)} developers in {len(self.flagged_sessions)} sessions'
        )


def oversight(groups: list[Group], classes: dict[str, CallClass]) -> dict[str, Oversight]:
    stats: dict[str, Oversight] = {}
    for group in groups:
        s = stats[group.key] = Oversight()
        for call in group.calls:
            if (c := classes.get(call.span_id)) is None:
                continue
            s.calls += 1
            s.harmless += c.target == 'harmless'
            s.protected += c.target == 'protected'
            s.unprompted += not c.requested and c.target != 'harmless'
            s.requested_own += c.requested and c.target == 'own'
            s.steered += c.steered
            if c.flagged:
                s.flagged_users.add(call.user)
                s.flagged_sessions.add(call.session_id)
    return stats


def _summary(group: Group, classes: dict[str, CallClass], *, examples: int = 8) -> dict[str, Any]:
    """What the LLM sees: counts and deduplicated, redacted flagged command shapes, never who ran them."""
    flagged = [c for c in group.calls if (k := classes.get(c.span_id)) and k.flagged]
    shapes: dict[str, set[str]] = defaultdict(set)
    origins: dict[str, set[str]] = defaultdict(set)
    for call in flagged:
        if call.command:
            shape = redact_secrets(' '.join(call.command.split()))[:240]
            shapes[shape].add(call.user)
            k = classes[call.span_id]
            origins[shape].add(f'{"requested" if k.requested else "unprompted"}, {k.target}')
    top = sorted(shapes.items(), key=lambda kv: len(kv[1]), reverse=True)[:examples]
    return {
        'key': group.key,
        'flagged_calls': len(flagged),
        'unprompted_calls': sum(not classes[c.span_id].requested for c in flagged),
        'protected_target_calls': sum(classes[c.span_id].target == 'protected' for c in flagged),
        'steered_calls': sum(classes[c.span_id].steered for c in flagged),
        'distinct_users': len({c.user for c in flagged}),
        'examples': [
            {'command': cmd, 'distinct_users': len(users), 'origin': sorted(origins[cmd])} for cmd, users in top
        ],
    }


POLICY_INSTRUCTIONS = """\
You advise a company's platform team on what its coding agents do with the shell and MCP tools, aggregated across
developers, and propose controls to roll out in observe mode first (record what would be blocked, block nothing).

You get risk categories with only the FLAGGED calls: ones the agent chose without being asked (`unprompted`), or that
hit a protected target (the default or a release branch, a remote or production host, outside the workspace, secrets),
requested or not. Calls a developer asked for on their own branch or workspace were already left out: they are normal.
You also get the non-built-in (MCP) tool names in use with their user counts.

Propose:
- `rules`: at most one per category, for risky behaviour worth governing (cover flag variants in one glob, e.g.
  `*git push*-f*` for `-f` and `--force`). Never turn a requested action into advice to do it.
  - Mostly unprompted (the agent does it on its own): `action` `ask`, and an `instruction` for the agents, phrased as
    "Don't X unless the user asks" (or the convention to follow instead).
  - A clearly destructive action on a protected target (force-pushing or hard-resetting the default branch, deleting
    shared branches, `curl | sh`, reading secret files): `deny`. Otherwise `ask`. The organization decides.
  `command` is a glob over the WHOLE command string (fnmatch: `*` matches anything), e.g. `*git push*--force*`. Make it
  precise: it must match the flagged examples and not routine commands (`git push origin my-branch` must not match a
  force-push rule). Generalize paths beyond the spelling in the examples: `~/.ssh/key`, `$HOME/.ssh/key` and
  `/Users/x/.ssh/key` are the same risk, so match `*.ssh/*`, not `*~/.ssh/*`.
  Skip categories where the flagged examples are all harmless after all.
- `mcp_servers`: servers that several developers adopted, worth adding to the company allowlist. Name each server and
  give globs over the tool names it provides.

Text is short. `finding`: one plain sentence on what agents are doing, as an advisor would say it to the platform team
("Agents force-push feature branches on their own while rebasing."), no counts (they are added from measurements).
`title`: the control, at most 8 words. `description`: one sentence, why. Never include names, usernames, emails,
tokens or keys.
"""


class _RuleDraft(BaseModel):
    category: str = Field(description='The risk category key this rule addresses.')
    name: str = Field(description='kebab-case')
    title: str = Field(
        description='What the rule does, as a short imperative a person would say, at most 8 words, no globs or '
        'flags (e.g. "Ask before force-deleting a branch", "Never pipe a download into a shell")'
    )
    finding: str = Field(description='What agents are doing, one sentence, no counts.')
    description: str = Field(description='One sentence: why it matters, in plain words.')
    action: Literal['deny', 'ask']
    command: str = Field(description='fnmatch glob over the whole shell command')
    instruction: str | None = Field(
        default=None, description='For behaviour agents choose unprompted: "Don\'t X unless the user asks."'
    )
    rationale: str

    _clean = field_validator('name', 'title', 'finding', 'description', 'rationale', 'command', 'instruction')(
        clean_text
    )


class _McpDraft(BaseModel):
    server: str
    tool_globs: list[str]
    description: str
    rationale: str

    _clean = field_validator('server', 'description', 'rationale')(clean_text)


class _PolicyDrafts(BaseModel):
    rules: list[_RuleDraft]
    mcp_servers: list[_McpDraft]


@dataclass
class _Deps:
    identifiers: set[str]
    commands: list[str]
    flagged: dict[str, list[str]]
    """Per risk category, the flagged commands a rule for it should cover."""


@dataclass(frozen=True)
class PolicyGates:
    """Never from one developer or one session; a risky action needs fewer occurrences than a style preference."""

    min_users: int = 2
    min_sessions: int = 2
    max_pending: int = 3


@dataclass
class PolicyResult:
    proposals: list[Proposal]
    """Passing findings, best first; those beyond `max_pending` are `stale` and `emerging`."""
    stale_reasons: dict[str, str]
    oversight: dict[str, Oversight]


def _days(calls: list[ToolCall]) -> int:
    return len({c.timestamp.date() for c in calls})


async def mine_policy(
    calls: list[ToolCall],
    *,
    prompts: list[UserPrompt],
    model: str,
    gates: PolicyGates,
    identifiers: set[str],
    window: tuple[datetime, datetime],
    classes_path: Path | None = None,
    max_evidence: int = 5,
) -> PolicyResult:
    risks = risk_groups(calls)
    tools = mcp_groups(calls)
    risky = [c for g in risks for c in g.calls]
    classes = await classify_calls(
        list({c.span_id: c for c in risky}.values()), prompts, model=model, cache_path=classes_path
    )
    stats = oversight(risks, classes)
    # Only categories that could pass go to the drafter: flagged by 2+ developers in 2+ sessions.
    eligible = [
        g
        for g in risks
        if len(stats[g.key].flagged_users) >= gates.min_users
        and len(stats[g.key].flagged_sessions) >= gates.min_sessions
    ]
    # Per category, for earlier suggestions whose category no longer qualifies at all.
    stale_reasons = {
        f'policy-{key}': (
            f"didn't pass: {s.requested_own} of {s.calls} calls were requested on the developer's own work and "
            f'{s.harmless} were harmless; flagged calls came from {len(s.flagged_users)} developer(s) in '
            f'{len(s.flagged_sessions)} session(s)'
        )
        for key, s in stats.items()
        if all(g.key != key for g in eligible)
    }
    if not eligible and not tools:
        return PolicyResult([], stale_reasons, stats)
    commands = [c.command for c in calls if c.command]
    agent = Agent(
        model,
        deps_type=_Deps,
        output_type=_PolicyDrafts,
        instructions=POLICY_INSTRUCTIONS,
        name='fleet_miner_policy',
        retries=3,
    )

    @agent.output_validator
    def check(ctx: RunContext[_Deps], drafts: _PolicyDrafts) -> _PolicyDrafts:
        problems: list[str] = []
        for rule in drafts.rules:
            if not any(matching_segment(c, rule.command) for c in ctx.deps.commands):
                problems.append(f'rule `{rule.name}`: glob `{rule.command}` matches none of the commands')
            elif flagged := ctx.deps.flagged.get(rule.category):
                covered = sum(1 for c in flagged if matching_segment(c, rule.command))
                if 2 * covered < len(flagged):
                    problems.append(
                        f'rule `{rule.name}`: glob `{rule.command}` matches only {covered} of the {len(flagged)} flagged '
                        f'`{rule.category}` commands; widen it to cover their variants'
                    )
        text = json.dumps(drafts.model_dump())
        if leaked := leaked_identifiers_in(text, ctx.deps.identifiers):
            problems.append(f'contains personal identifiers ({", ".join(sorted(leaked))}); remove them')
        if problems and ctx.retry < 2:
            raise ModelRetry('; '.join(problems))
        return drafts

    output = await run_cached(
        agent,
        'Risk categories (flagged calls only):\n'
        + json.dumps([_summary(g, classes) for g in eligible], indent=2)
        + '\n\nNon-built-in tools in use:\n'
        + json.dumps([{'tool': g.key, 'calls': len(g.calls), 'distinct_users': len(g.users)} for g in tools], indent=2),
        output_type=_PolicyDrafts,
        deps=_Deps(
            identifiers=identifiers,
            commands=commands,
            flagged={
                g.key: [c.command for c in g.calls if c.command and (k := classes.get(c.span_id)) and k.flagged]
                for g in eligible
            },
        ),
    )
    generated_by = f'fleet-miner {__version__} / {model}'
    passing: list[tuple[tuple[int, ...], Proposal]] = []
    flagged = [c for c in risky if (k := classes.get(c.span_id)) and k.flagged]
    for draft in output.rules:
        # Measure what the glob actually matches among the flagged calls; the LLM's claims don't count.
        matched = Group(
            draft.name,
            list({c.span_id: c for c in flagged if c.command and matching_segment(c.command, draft.command)}.values()),
        )
        users, sessions = len(matched.users), len({c.session_id for c in matched.calls})
        # Keyed by what the rule is about, not the LLM's wording, so reruns update it instead of adding a twin.
        proposal_id = f'policy-{draft.category}-{draft.action}'
        if any(p.id == proposal_id for _, p in passing):
            continue  # a second rule for the same category and action: the first one that passed stands
        if users < gates.min_users or sessions < gates.min_sessions:
            stale_reasons.setdefault(
                proposal_id,
                f"didn't pass: the rule matches flagged calls from {users} developer(s) in {sessions} session(s)",
            )
            continue
        ks = [classes[c.span_id] for c in matched.calls]
        unprompted = sum(not k.requested for k in ks)
        protected = sum(k.target == 'protected' for k in ks)
        steered = sum(k.steered for k in ks)
        s = stats.get(draft.category)
        normal = (
            f" {s.requested_own} more {'was' if s.requested_own == 1 else 'were'} requested on the developer's own"
            ' work: normal, not counted.'
            if s and s.requested_own
            else ''
        )
        control = f'Suggested control: {draft.action} (watching only at first)' + (
            f', plus the instruction "{draft.instruction}".' if draft.instruction and unprompted else '.'
        )
        note = (
            f'{draft.finding} {len(matched.calls)} flagged calls by {users} developers in {sessions} sessions over '
            f'{_days(matched.calls)} day(s): {unprompted} unprompted by the agent, {protected} on a protected target'
            f'{f", {steered} the developer then pushed back on" if steered else ""}.{normal} {control}'
        )
        rule = PolicyRule(
            name=draft.name,
            description=draft.title,
            action=draft.action,
            match=PolicyMatch(tool='shell', command=draft.command),
            proposal_id=proposal_id,
        )
        proposal = Proposal(
            id=proposal_id,
            kind='policy',
            name=draft.name,
            # The UI leads with the human sentence; the glob lives in `rule.match.command`.
            description=draft.title,
            text=note,
            suggested_tier='required',
            rationale=f'{draft.description} {draft.rationale}',
            pattern=f'{draft.category}: {draft.command}',
            distinct_users=users,
            sessions=sessions,
            matching_calls=len(matched.calls),
            corrections=steered,
            suggested_instruction=draft.instruction if unprompted else None,
            evidence=_evidence(matched, max_evidence, classes),
            rule=rule,
            trend=daily_trend(((c.timestamp, c.user) for c in matched.calls), *window),
            **_scope_fields(matched.calls, calls),
            generated_by=generated_by,
        )
        # Agents acting on their own first, then pushback, then spread.
        passing.append(((min(unprompted, 1), steered, users, sessions, len(matched.calls)), proposal))
    by_tool = {g.key: g for g in tools}
    for draft in output.mcp_servers:
        matched = Group(
            draft.server, [c for g in tools for c in g.calls if any(fnmatchcase(g.key, p) for p in draft.tool_globs)]
        )
        users, sessions = len(matched.users), len({c.session_id for c in matched.calls})
        proposal_id = f'policy-mcp-{re.sub(r"[^a-z0-9]+", "-", draft.server.lower()).strip("-")}'
        if users < gates.min_users or sessions < gates.min_sessions or not by_tool:
            stale_reasons[proposal_id] = f"didn't pass: used by {users} developer(s) in {sessions} session(s)"
            continue
        proposal = Proposal(
            id=proposal_id,
            kind='policy',
            name=f'allow-mcp-{draft.server}',
            description=draft.description,
            text=f'Developers adopted the `{draft.server}` MCP server: {len(matched.calls)} tool calls by {users} '
            f'developers in {sessions} sessions. Suggested control: add it to the MCP allowlist (watching only at '
            'first).',
            suggested_tier='default_on',
            rationale=f'{draft.rationale} Measured: {len(matched.calls)} tool calls by {users} developers.',
            pattern=f'mcp server {draft.server}: {", ".join(draft.tool_globs)}',
            distinct_users=users,
            sessions=sessions,
            matching_calls=len(matched.calls),
            evidence=_evidence(matched, max_evidence, classes),
            mcp=McpAllow(allow=[draft.server]),
            trend=daily_trend(((c.timestamp, c.user) for c in matched.calls), *window),
            **_scope_fields(matched.calls, calls),
            generated_by=generated_by,
        )
        passing.append(((0, 0, users, sessions, len(matched.calls)), proposal))
    ranked = [p for _, p in sorted(passing, key=lambda kv: kv[0], reverse=True)]
    # An earlier suggestion for a category that now passes with the other action was replaced, not dropped.
    for p in ranked:
        category, action = p.id.rsplit('-', 1)
        other = f'{category}-{"ask" if action == "deny" else "deny"}'
        stale_reasons[other] = f'replaced by `{p.id}`: {p.text.split(". ")[0]}.'
    for p in ranked:
        stale_reasons.pop(p.id, None)
    for p in ranked[gates.max_pending :]:
        p.status, p.emerging = 'stale', True
        p.status_reason = 'Emerging: passes every gate, but ranks below the pending policy suggestions.'
    return PolicyResult(ranked, stale_reasons, stats)


def _scope_fields(matched: list[ToolCall], calls: list[ToolCall]) -> dict[str, Any]:
    measured = measure_scope(
        [(c.team, c.repo_slug) for c in matched],
        window_teams={c.team for c in calls if c.team},
        window_repos={c.repo_slug for c in calls if c.repo_slug},
    )
    if measured is None:
        tagged = sum(1 for c in matched if c.team or c.repo_slug)
        return {
            'scope': 'organization',
            'scope_reason': f'Default (only {tagged} of {len(matched)} matching calls carry repo or team data).',
        }
    scope, reason, applies_to = measured
    return {'scope': scope, 'scope_reason': reason, 'applies_to': applies_to}


def _evidence(group: Group, limit: int, classes: dict[str, CallClass]) -> list[Evidence]:
    """Pointers to the matched calls, one per developer first to show the spread, each saying who chose it."""
    picked: list[ToolCall] = []
    for call in group.calls:
        if call.user not in {p.user for p in picked}:
            picked.append(call)
    picked += [c for c in group.calls if c not in picked]

    def labels(call: ToolCall) -> dict[str, Any]:
        if (k := classes.get(call.span_id)) is None:
            return {}
        return {
            'origin': 'requested' if k.requested else 'unprompted',
            'target': 'protected' if k.target == 'protected' else 'own',
        }

    return [
        Evidence(trace_id=c.trace_id, span_id=c.span_id, timestamp=c.timestamp, **labels(c)) for c in picked[:limit]
    ]


def mask_identifiers(text: str, identifiers: set[str], replacement: str = 'a teammate') -> str:
    """Last-line defense (drafting already retries on identifiers): rephrase rather than leave a `<person>` token."""
    for i in sorted(identifiers, key=len, reverse=True):
        text = re.sub(rf'(?<![\w-]){re.escape(i)}(?![\w-])', replacement, text, flags=re.IGNORECASE)
    return text
