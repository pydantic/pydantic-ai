"""Send an agent that edited code back to verify it before the run ends."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from fnmatch import fnmatch
from typing import Any

from pydantic_ai import CallToolsNode, DeferredToolRequests, PendingModelResponse, RunContext
from pydantic_ai._capability_parts import check_operator_text
from pydantic_ai.capabilities import AbstractCapability, AgentNode, CapabilityOrdering, NodeResult
from pydantic_ai.capabilities.abstract import ValidatedToolArgs
from pydantic_ai.messages import (
    InstructionCapabilityPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    OperatorSource,
    OperatorText,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import AgentDepsT, ToolDefinition
from pydantic_ai_harness._shell_command import shell_command, shell_command_arg
from pydantic_ai_harness.file_system._tool_kinds import EDIT_FILE_KIND, WRITE_FILE_KIND
from pydantic_ai_harness.require_verification._events import CompletionUnverifiedEvent
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits
from pydantic_graph import End

DEFAULT_REQUIRE_VERIFICATION_INSTRUCTIONS = (
    'You edited code, but no verification has passed since your last edit. Run the relevant tests, lint, '
    'type checks, or build now, each as its own foreground command whose output is not piped or followed by '
    'another command, so its exit status shows whether it passed. If verification cannot run, explain the '
    'concrete blocker instead of claiming the work is verified.'
)
"""What `RequireVerification` tells the agent each time it sends the run back, unless `instructions` is set."""

DEFAULT_VERIFICATION_COMMANDS: tuple[str, ...] = (
    # `ruff format`, `black` and `isort` count only with `--check` or `--diff` later in the step. `_verifies` matches
    # only steps without `;`, `&`, `|` or a newline, so the first place one of them appears decides, and the atomic
    # group commits to it, so the lookahead runs once per match attempt. Folded into one alternative with the other
    # tools, the lookahead would re-read the rest of the step for each way `(?:\S*/)?` and `(?:\s+-\S+)*` can split
    # it at a `/` or a flag, which takes quadratic time on a long step such as `black/black/.../black x`.
    r'\A(?<!\S)(?:'
    r'(?:(?:uv|poetry|pipenv|hatch|pdm)\s+run(?:\s+-\S+)*\s+)?(?:(?<!\S)(?:\S*/)?python3?\s+-m\s+)?(?<!\S)(?:\S*/)?'
    r'(?:pytest|py\.test|unittest|tox|nox|mypy|pyright|basedpyright|ty\s+check|ruff\s+check)\b'
    r'|(?>(?:(?:uv|poetry|pipenv|hatch|pdm)\s+run(?:\s+-\S+)*?\s+)??(?:(?<!\S)(?:\S*?/)??python3?\s+-m\s+)??'
    r'(?<!\S)(?:\S*?/)??(?:ruff\s+format|black|isort)\b)'
    r'(?=[^;&|\n]*\s--(?:check|diff)))',
    r'(?:npm|pnpm|yarn|bun)'
    r'(?:\s+(?>workspace\s+\S+|-\S+(?:\s+(?!(?:run|test|lint|typecheck|check|build|workspace)\b)[^-\s]\S*)?))*'
    r'\s+(?:run\s+)?(?:test|lint|typecheck|check|build)\b',
    r'(?<!\S)(?:npx\s+)?(?<!\S)(?:\S*/)?(?:tsc|eslint|vitest|jest)\b',
    r'(?:cargo(?:\s+\+\S+)?|go|dotnet)\s+(?:test|build|check|clippy|vet)\b',
    r'(?:make|just)\s+(?:\S+\s+)*?(?:test|lint|typecheck|check|build)\w*\b',
    r'(?:mvn|gradle|\./mvnw|\./gradlew)\s+(?:\S+\s+)*?(?:test|verify|check|build)\b',
)
"""Regular expressions for the commands that count as verification, matched at the start of a command's steps."""

DEFAULT_EDIT_COMMANDS: tuple[str, ...] = (
    r'(?:^|(?<=[;&|(\n]))(?<![ \t])[ \t]*+(?:(?:uv|poetry|pipenv|hatch|pdm)\s+run\s+)?(?:ruff\s+format|black|isort|cargo\s+fmt|gofmt\s+-w)\b'
    r'(?![^;&|\n]*\s--(?:check|diff))',
    r'--(?:fix|write)\b',
    r'(?<![\w-])(?:sed|perl)(?:\s+-\w+)*\s+-\w*i',
    r'\bgit\s+(?:apply|am|checkout|cherry-pick|merge|mv|pull|rebase|reset|restore|revert|rm|stash|switch)\b',
    r'(?<![\w-])(?:patch|mv|cp|rm|tee)\s',
    r'(?<![\w-])(?:ba|z|da)?sh\s+-\w*c\b',
    r'(?<![\w-])(?:python3?\s+(?:-c|-)(?=\s|$)|(?:node|ruby|perl)\s+-e\b)',
    r'(?<![\d&>-])>>?(?!&|\s*/dev/null)',
)
"""Regular expressions for the commands that count as edits, searched for anywhere outside quoted strings."""

_ASSIGNMENTS = re.compile(r'^(?:env\s+)?(?:\w+=\S*\s+)*')
_MASKING = re.compile(r'[|&]')
_STATEMENT_SEPARATOR = re.compile(r'[;\n]')
_PROBE = re.compile(r'(?<!\S)(?:--version|--help|-h|--collect-only|--co)(?!\S)')
_SHORT_CIRCUIT = re.compile(r'(?<![\w/.-])(?:exit|exec|return)\b')
_FD_DUPLICATION = re.compile(r'(?<!\d)\d*[<>]&(?:\d+|-)')
_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
_FAILED = (
    '[exit code:',
    '[Command timed out',
    '[File-size limit exceeded',
    '[Command failed with max_file_bytes',
    '[exited with code',
    '[terminated by signal',
)
"""The last line of a failed command's result from `Shell`'s tools and ACP's `run_command`."""
_FILE_EDIT_KINDS = (WRITE_FILE_KIND.id, EDIT_FILE_KIND.id)
_SUPERVISOR = re.compile(r'PID: \d+ \(supervisor\b')
"""The handles header of the `shell` tool's result, before its output, log, and status lines."""


@dataclass(kw_only=True)
class VerificationRequiredPart(
    InstructionCapabilityPart, namespace='require_verification', name='verification_required'
):
    """The instruction `RequireVerification` adds to the conversation when it sends the run back to verify.

    Its `content` is the capability's `instructions`, in the application's voice. Find redirects in history by this
    type.
    """

    attempt: int
    """Which of the run's redirects it is, from 1. Never sent to the model."""


@dataclass(kw_only=True)
class RequireVerification(AbstractCapability[AgentDepsT]):
    """Send the agent back to verify its code edits before the run ends.

    A successful `FileSystem` `write_file` or `edit_file` call, or a shell command matching
    `edit_commands`, is an edit and makes earlier verification results stale. A shell command matching
    `verification_commands` records whether it passed. When the run would end after an edit and the
    latest run of every verification command since then did not pass, a
    [`VerificationRequiredPart`][pydantic_ai_harness.require_verification.VerificationRequiredPart] carrying
    `instructions` is added and the model answers again, up to `max_attempts` times per run. A run that
    still ends unverified emits a `CompletionUnverifiedEvent`.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import FileSystem, RequireVerification, Shell

    agent = Agent('anthropic:claude-opus-5-5', capabilities=[FileSystem(), Shell(), RequireVerification()])
    ```
    """

    max_attempts: int = 2
    """How many times per run the agent is sent back to verify."""

    instructions: str = DEFAULT_REQUIRE_VERIFICATION_INSTRUCTIONS
    """What the agent is told each time it is sent back."""

    verification_commands: Sequence[str] = DEFAULT_VERIFICATION_COMMANDS
    """Regular expressions for the commands that count as verification."""

    edit_commands: Sequence[str] = DEFAULT_EDIT_COMMANDS
    """Regular expressions for the commands that count as edits."""

    exempt_paths: Sequence[str] = ('*.md', '*.mdx', '*.rst')
    """Glob patterns for the files whose edits need no verification."""

    _verification: tuple[re.Pattern[str], ...] = field(init=False, repr=False, compare=False)
    _edit: tuple[re.Pattern[str], ...] = field(init=False, repr=False, compare=False)
    _edits: int = field(default=0, init=False, repr=False, compare=False)
    _started: dict[str, int] = field(default_factory=dict[str, int], init=False, repr=False, compare=False)
    _results: dict[str, bool] = field(default_factory=dict[str, bool], init=False, repr=False, compare=False)
    _attempts: int = field(default=0, init=False, repr=False, compare=False)
    _reported: bool = field(default=False, init=False, repr=False, compare=False)
    _calls: int = field(default=0, init=False, repr=False, compare=False)
    _calls_at_redirect: int = field(default=0, init=False, repr=False, compare=False)
    _command_args: dict[str, str] = field(default_factory=dict[str, str], init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.max_attempts < 0:
            raise ValueError(f'`max_attempts` must be at least 0, got {self.max_attempts}.')
        try:
            self._verification = tuple(re.compile(pattern) for pattern in self.verification_commands)
            self._edit = tuple(re.compile(pattern) for pattern in self.edit_commands)
        except re.error as error:
            raise ValueError(f'RequireVerification has an invalid command pattern: {error}') from error
        check_operator_text(self.instructions, what='`RequireVerification.instructions`')

    def get_ordering(self) -> CapabilityOrdering:
        """Run inside `ToolOutputLimits`, so results are read before it truncates, spills, or summarizes them."""
        return CapabilityOrdering(wrapped_by=(ToolOutputLimits,))

    async def prepare_tools(self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        """Note which tools take a shell command, to read edits from the history of a resumed run."""
        for tool_def in tool_defs:
            if (name := shell_command_arg(tool_def)) is not None:
                self._command_args[tool_def.name] = name
        return tool_defs

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> RequireVerification[AgentDepsT]:
        """Return a fresh instance, so runs don't share their edits, results, or attempts."""
        return replace(self)

    async def before_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
    ) -> ValidatedToolArgs:
        """Note how many edits preceded a verification command, as calls in one response run in parallel."""
        # A command started in the background is not a check, even if it finished before the tool returned.
        if (
            (command := shell_command(tool_def, args)) is not None
            and args.get('mode') != 'background'
            and self._verifies(command)
        ):
            self._started[call.tool_call_id] = self._edits
        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        """Record a successful edit or verification command."""
        self._calls += 1
        edits_before = self._started.pop(call.tool_call_id, None)
        if tool_def.tool_kind in _FILE_EDIT_KINDS:
            if not self._exempt(args.get('path')):
                self._mark_edit()
        elif (command := shell_command(tool_def, args)) is not None:
            self._record_command(command, edits_before, result)
        return result

    def _exempt(self, path: object) -> bool:
        return isinstance(path, str) and any(fnmatch(path, pattern) for pattern in self.exempt_paths)

    def _edited_before_resume(self, messages: Sequence[ModelMessage], run_id: str | None) -> bool:
        """Whether an earlier run in this turn, which paused for its caller, made an edit."""
        returned = {
            part.tool_call_id
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.outcome == 'success'
        }
        turn_start = max(
            (
                index
                for index, message in enumerate(messages)
                if isinstance(message, ModelRequest) and any(isinstance(part, UserPromptPart) for part in message.parts)
            ),
            default=0,
        )
        return any(
            isinstance(part, ToolCallPart) and part.tool_call_id in returned and self._is_edit(part)
            for message in messages[turn_start:]
            if isinstance(message, ModelResponse) and message.run_id != run_id
            for part in message.parts
        )

    def _is_edit(self, call: ToolCallPart) -> bool:
        """Whether a call from history was an edit: a `FileSystem` write, or a shell command matching `edit_commands`."""
        args = call.args_as_dict()
        if call.tool_kind in _FILE_EDIT_KINDS:
            return not self._exempt(args.get('path'))
        name = self._command_args.get(call.tool_name)
        command = None if name is None else args.get(name)
        return isinstance(command, str) and self._is_edit_command(command)

    def _is_edit_command(self, command: str) -> bool:
        unquoted = _QUOTED.sub('', command)
        return any(pattern.search(unquoted) for pattern in self._edit)

    def _record_command(self, command: str, edits_before: int | None, result: object) -> None:
        if self._is_edit_command(command):
            self._mark_edit()
            return
        # A verification that an edit in a parallel call may have raced, or no verification at all, records nothing.
        if edits_before != self._edits:
            return
        if (passed := _passed(result)) is None:
            self._results.pop(command, None)
        else:
            self._results[command] = passed

    async def after_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        result: NodeResult[AgentDepsT],
    ) -> NodeResult[AgentDepsT]:
        """Send an unverified run back to the model, or report that it ended unverified."""
        # As in `RefineOutput`: a run paused for its caller, `Agent.run_stream()`'s own end node, and an
        # output that came from no model request cannot take another turn.
        if not (
            isinstance(node, CallToolsNode)
            and node.model_response.model_name is not None
            and isinstance(result, End)
            and not isinstance(result.data.output, DeferredToolRequests | PendingModelResponse)
            and (self._edits or self._edited_before_resume(ctx.messages, ctx.run_id))
            and not (self._results and all(self._results.values()))
        ):
            return result
        # A model that answers a redirect without calling a tool has explained a blocker: asking again repeats it.
        if self._attempts < self.max_attempts and (not self._attempts or self._calls > self._calls_at_redirect):
            self._attempts += 1
            self._calls_at_redirect = self._calls
            ctx.tracer.start_span(
                'verification required', attributes={'require_verification.attempt': self._attempts}
            ).end()
            redirect = VerificationRequiredPart(
                content=OperatorText(self.instructions), source=OperatorSource(), attempt=self._attempts
            )
            ctx.enqueue(redirect, priority='when_idle')
        elif not self._reported:
            # Something else, like `RefineOutput`, can still continue the run: report its first unverified end once.
            self._reported = True
            ctx.tracer.start_span(
                'completion unverified', attributes={'require_verification.attempts': self._attempts}
            ).end()
            failed = [command for command, passed in self._results.items() if not passed]
            await ctx.emit(CompletionUnverifiedEvent(attempts=self._attempts, failed_commands=failed))
        return result

    def _mark_edit(self) -> None:
        # Any edit makes every earlier result stale. Narrowing that to the edits a check depends on would need
        # its full dependency closure (fixtures, config, imports), and a wrong guess there passes unverified work.
        self._edits += 1
        self._results.clear()

    def _verifies(self, command: str) -> bool:
        """Whether `command` runs a verification whose exit status no other part of it can hide.

        The exit status is the last `;`- or newline-separated statement's, so the check must be in it,
        joined only by `&&` to the rest of that statement.
        """
        if '$(' in command or '`' in command:
            return False
        # Separators, pipes, and words inside quotes are literal text, not shell syntax.
        text = _FD_DUPLICATION.sub('', _QUOTED.sub("''", command.replace('\\\n', ' ')))
        statements = [statement for statement in _STATEMENT_SEPARATOR.split(text) if statement.strip()]
        if not statements or _SHORT_CIRCUIT.search(text):
            return False
        steps = [_ASSIGNMENTS.sub('', step.strip(), count=1) for step in statements[-1].split('&&')]
        if any(_MASKING.search(step) for step in steps):
            return False
        return any(not _PROBE.search(step) and pattern.match(step) for step in steps for pattern in self._verification)


def _passed(result: object) -> bool | None:
    """Whether a shell tool's result reports a command that passed; `None` while it is still running.

    The `shell` tool's result ends with its handles and, once the command finished, its JSON status;
    `start_command` only reports that it started. Any other result is read by its last line, in the formats
    of `run_command` and ACP's `run_command`: a failure trailer, or a pass.
    """
    if not isinstance(result, str):
        return None
    # The last four lines, each one whole: the command's own output comes before the `shell` tool's handles.
    lines = result.rstrip().rsplit('\n', 4)[-4:]
    if len(lines) == 4 and _SUPERVISOR.match(lines[0]):
        try:
            status = json.loads(lines[3])
        except ValueError:
            return None
        match status:
            case {'exit_code': int(exit_code)}:
                return exit_code == 0
            case _:
                return None
    if len(lines) >= 3 and _SUPERVISOR.match(lines[-3]) or result.startswith('Started background command:'):
        return None
    return not lines[-1].startswith(_FAILED)
