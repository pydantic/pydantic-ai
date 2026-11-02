"""Capability that supplies an Amazon Bedrock AgentCore Code Interpreter session to an agent run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness._workspace_provider import check_env, check_integer, check_working_dir
from pydantic_ai_harness.agentcore_sandbox._backend import (
    DEFAULT_CODE_INTERPRETER,
    DEFAULT_SESSION_TIMEOUT,
    MAX_SESSION_TIMEOUT,
    PROVIDER,
    AgentCoreSandboxBackend,
    stop_session,
)


@dataclass(kw_only=True)
class AgentCoreSandbox(AbstractCapability[AgentDepsT]):
    """Supply an isolated [Amazon Bedrock AgentCore Code Interpreter](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-tool.html) session as the run's workspace.

    A run with no reference starts a fresh session on first use; pass a `WorkspaceRef` to attach to
    one that is still running. Ending a run does not stop the session: it runs until `destroy` stops
    it or its `session_timeout` runs out, and it cannot be resumed after that. AWS credentials and
    the region come from boto3's usual configuration.

    This capability supplies execution only. Compose it with tools or capabilities that use the
    workspace, such as `Coder`, `Shell`, or `FileSystem`. Shell commands run under `sh -c`.
    """

    region_name: str | None = None
    """AWS region of the code interpreter; boto3's configured region when `None`.

    A ref does not record its region, so a run that attaches to a session must use the region it was started in.
    """

    code_interpreter: str = DEFAULT_CODE_INTERPRETER
    """The AWS-managed `aws.codeinterpreter.v1`, or the id of a code interpreter you created.

    The AWS-managed one runs in sandbox network mode, which reaches Amazon S3 but not the internet;
    create your own in public or VPC network mode for more.
    An unknown one raises `WorkspaceUnavailableError` on first use.
    """

    session_timeout: int = DEFAULT_SESSION_TIMEOUT
    """Total lifetime of a newly started session in seconds, up to 28800 (8 hours).

    It is not an idle timeout: the session ends when it runs out, whatever it is doing, and its files go with it.
    """

    working_dir: str | None = None
    """Absolute directory commands start in and relative paths resolve against; `None` uses the session's own.

    Created in a session this capability starts; an attached session must already have it.
    """

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Environment variables every command gets, also in an attached session; nothing is read from the host.

    The agent's commands can read them.
    """

    def __post_init__(self) -> None:
        if self.defer_loading:
            raise UserError(
                '`AgentCoreSandbox` does not support `defer_loading=True`: '
                'the workspace is selected before deferred capabilities load.'
            )
        check_integer('session_timeout', self.session_timeout)
        if self.session_timeout > MAX_SESSION_TIMEOUT:
            raise UserError(f'session_timeout must be at most {MAX_SESSION_TIMEOUT}, got {self.session_timeout!r}.')
        check_working_dir(self.working_dir)
        try:
            check_env(self.env)
        except (TypeError, ValueError) as error:
            raise UserError(f'`env`: {error}.') from None

    def backend(self, ref: WorkspaceRef) -> AgentCoreSandboxBackend:
        """Attach lazily to a running session."""
        return AgentCoreSandboxBackend(
            ref=ref, region_name=self.region_name, working_dir=self.working_dir, env=self.env
        )

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Stop a session by its ref. One that has already ended returns quietly."""
        await stop_session(ref, region_name=self.region_name)

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build the backend for this run. No I/O here: it starts or attaches on first use."""
        del ctx
        if ref is not None and ref.provider != PROVIDER:
            return None
        return AgentCoreSandboxBackend(
            ref=ref,
            region_name=self.region_name,
            code_interpreter=self.code_interpreter,
            session_timeout=self.session_timeout,
            working_dir=self.working_dir,
            env=self.env,
        )
