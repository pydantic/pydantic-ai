"""Policy mining: risky shell commands and MCP servers across the fleet, drafted as observe-mode policy proposals.

The LLM only sees aggregates (deduplicated, redacted command patterns with per-pattern user counts), never raw spans.
Every rule it proposes is then re-checked against the actual commands, so counts and evidence are measured, not
claimed.
"""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Literal

from logfire.query_client import AsyncLogfireQueryClient
from pydantic import BaseModel, Field, field_validator

from pydantic_ai import Agent, ModelRetry, RunContext

from . import __version__
from .fetch import _SESSIONS, IDENTITY, NOT_TEST, OVERLAP
from .llm_cache import run_cached
from .models import Evidence, McpAllow, daily_trend, PolicyMatch, PolicyRule, Proposal, clean_text, redact_secrets
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


def _summary(group: Group, *, examples: int = 8) -> dict[str, Any]:
    """What the LLM sees: counts and deduplicated, redacted command shapes, never who ran them."""
    shapes: dict[str, set[str]] = defaultdict(set)
    for call in group.calls:
        if call.command:
            shapes[redact_secrets(' '.join(call.command.split()))[:240]].add(call.user)
    top = sorted(shapes.items(), key=lambda kv: len(kv[1]), reverse=True)[:examples]
    return {
        'key': group.key,
        'calls': len(group.calls),
        'distinct_users': len(group.users),
        'examples': [{'command': cmd, 'distinct_users': len(users)} for cmd, users in top],
    }


POLICY_INSTRUCTIONS = """\
You review how a company's coding agents use the shell and MCP tools, aggregated across developers, and propose
policy rules for the platform team to roll out in observe mode first (record what would be blocked, block nothing).

You get risk categories, each with call counts, distinct-user counts and deduplicated example commands, plus the
non-built-in (MCP) tool names in use with their user counts.

Propose:
- `rules`: one per genuinely risky behavior worth governing. `action` is `deny` for things an agent should never do
  unattended (force-pushing, `git reset --hard` on shared work, `curl | sh`, reading secret files), `ask` for things
  that are fine with a human's OK (pushing to main, deleting branches, `rm -rf` of a directory). `command` is a glob
  over the WHOLE command string (fnmatch: `*` matches anything), e.g. `*git push*--force*`. Make it precise: it must
  match the risky examples and not routine commands (`git push origin my-branch` must not match a force-push rule).
  Generalize paths beyond the spelling in the examples: `~/.ssh/key`, `$HOME/.ssh/key` and `/Users/x/.ssh/key` are
  the same risk, so match `*.ssh/*`, not `*~/.ssh/*`.
  Skip categories where the examples are all harmless (e.g. `rm -rf` of `node_modules` or a temp dir).
- `mcp_servers`: servers that several developers use, worth adding to the company allowlist. Name each server and
  give globs over the tool names it provides.

Text is short: `description` one sentence, `rationale` one or two sentences citing the counts you were given.
Never include names, usernames, emails, tokens or keys.
"""


class _RuleDraft(BaseModel):
    category: str = Field(description='The risk category key this rule addresses.')
    name: str = Field(description='kebab-case')
    title: str = Field(
        description='What the rule does, as a short imperative a person would say, at most 8 words, no globs or '
        'flags (e.g. "Ask before force-deleting a branch", "Never pipe a download into a shell")'
    )
    description: str = Field(description='One sentence: why, in plain words.')
    action: Literal['deny', 'ask']
    command: str = Field(description='fnmatch glob over the whole shell command')
    rationale: str

    _clean = field_validator('name', 'title', 'description', 'rationale', 'command')(clean_text)


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


async def mine_policy(
    calls: list[ToolCall],
    *,
    model: str,
    min_users: int,
    identifiers: set[str],
    window: tuple[datetime, datetime],
    max_evidence: int = 5,
) -> list[Proposal]:
    risks = risk_groups(calls)
    tools = mcp_groups(calls)
    if not risks and not tools:
        return []
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
        text = json.dumps(drafts.model_dump())
        if leaked := leaked_identifiers_in(text, ctx.deps.identifiers):
            problems.append(f'contains personal identifiers ({", ".join(sorted(leaked))}); remove them')
        if problems and ctx.retry < 2:
            raise ModelRetry('; '.join(problems))
        return drafts

    output = await run_cached(
        agent,
        'Risk categories:\n'
        + json.dumps([_summary(g) for g in risks], indent=2)
        + '\n\nNon-built-in tools in use:\n'
        + json.dumps([{'tool': g.key, 'calls': len(g.calls), 'distinct_users': len(g.users)} for g in tools], indent=2),
        output_type=_PolicyDrafts,
        deps=_Deps(identifiers=identifiers, commands=commands),
    )
    generated_by = f'fleet-miner {__version__} / {model}'
    proposals: list[Proposal] = []
    groups = {g.key: g for g in risks}
    for draft in output.rules:
        # Measure what the glob actually matches; the LLM's claims don't count.
        segments = {c.span_id: seg for c in calls if c.command and (seg := matching_segment(c.command, draft.command))}
        matched = Group(draft.name, [c for c in calls if c.span_id in segments])
        users = len(matched.users)
        # Never suggest from one person: a rule needs `min_users` distinct developers like every other suggestion.
        if users < min_users:
            continue
        # Keyed by what the rule is about, not the LLM's wording, so reruns update it instead of adding a twin.
        proposal_id = f'policy-{draft.category}-{draft.action}'
        rule = PolicyRule(
            name=draft.name,
            description=draft.title,
            action=draft.action,
            match=PolicyMatch(tool='shell', command=draft.command),
            proposal_id=proposal_id,
        )
        proposals.append(
            Proposal(
                id=proposal_id,
                kind='policy',
                name=draft.name,
                # The UI leads with the human sentence; the glob lives in `rule.match.command`.
                description=draft.title,
                text=draft.description,
                suggested_tier='required',
                rationale=f'{draft.rationale} Measured: {len(matched.calls)} matching calls by {users} '
                f'developer(s) in {len({c.session_id for c in matched.calls})} sessions.',
                pattern=f'{groups[draft.category].key if draft.category in groups else draft.category}: {draft.command}',
                distinct_users=users,
                sessions=len({c.session_id for c in matched.calls}),
                matching_calls=len(matched.calls),
                evidence=_evidence(matched, max_evidence),
                rule=rule,
                trend=daily_trend(((c.timestamp, c.user) for c in matched.calls), *window),
                **_scope_fields(matched.calls, calls),
                generated_by=generated_by,
            )
        )
    by_tool = {g.key: g for g in tools}
    for draft in output.mcp_servers:
        matched = Group(
            draft.server, [c for g in tools for c in g.calls if any(fnmatchcase(g.key, p) for p in draft.tool_globs)]
        )
        users = len(matched.users)
        if users < min_users or not by_tool:
            continue
        proposals.append(
            Proposal(
                id=f'policy-mcp-{re.sub(r"[^a-z0-9]+", "-", draft.server.lower()).strip("-")}',
                kind='policy',
                name=f'allow-mcp-{draft.server}',
                description=draft.description,
                text=f'Add `{draft.server}` to the MCP allowlist (observe mode).',
                suggested_tier='default_on',
                rationale=f'{draft.rationale} Measured: {len(matched.calls)} tool calls by {users} developers.',
                pattern=f'mcp server {draft.server}: {", ".join(draft.tool_globs)}',
                distinct_users=users,
                sessions=len({c.session_id for c in matched.calls}),
                matching_calls=len(matched.calls),
                evidence=_evidence(matched, max_evidence),
                mcp=McpAllow(allow=[draft.server]),
                trend=daily_trend(((c.timestamp, c.user) for c in matched.calls), *window),
                **_scope_fields(matched.calls, calls),
                generated_by=generated_by,
            )
        )
    return proposals


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


def _evidence(group: Group, limit: int) -> list[Evidence]:
    """Pointers to the matched calls, one per developer first to show the spread."""
    picked: list[ToolCall] = []
    for call in group.calls:
        if call.user not in {p.user for p in picked}:
            picked.append(call)
    picked += [c for c in group.calls if c not in picked]
    return [Evidence(trace_id=c.trace_id, span_id=c.span_id, timestamp=c.timestamp) for c in picked[:limit]]


def mask_identifiers(text: str, identifiers: set[str], replacement: str = 'a teammate') -> str:
    """Last-line defense (drafting already retries on identifiers): rephrase rather than leave a `<person>` token."""
    for i in sorted(identifiers, key=len, reverse=True):
        text = re.sub(rf'(?<![\w-]){re.escape(i)}(?![\w-])', replacement, text, flags=re.IGNORECASE)
    return text
