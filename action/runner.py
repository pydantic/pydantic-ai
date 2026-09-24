"""Run a Pydantic AI agent, for the composite action defined in `action.yml`.

The action passes every input through the environment rather than as arguments, so a prompt that
starts with a dash or spans several lines needs no quoting rules of its own.
"""

from __future__ import annotations

import importlib
import os
import secrets
import sys
from pathlib import Path
from typing import TypeGuard

from pydantic_ai import Agent

_SPEC_SUFFIXES = {'.yml', '.yaml', '.json'}


def _is_agent(value: object) -> TypeGuard[Agent[object, object]]:
    return isinstance(value, Agent)


def _resolve_agent(target: str) -> Agent[object, object]:
    """Load the agent `target` names: an agent spec file, or a `module:variable` pair."""
    if Path(target).suffix.lower() in _SPEC_SUFFIXES:
        agent: Agent[object, object] = Agent.from_file(target)
        return agent

    module_name, separator, variable_name = target.partition(':')
    if not separator or not module_name or not variable_name:
        raise ValueError(
            'expected a `module:variable` target, or an agent spec file ending in `.yml`, `.yaml`, or `.json`'
        )

    module = importlib.import_module(module_name)
    value: object = getattr(module, variable_name)
    if not _is_agent(value):
        value_type = f'{type(value).__module__}.{type(value).__qualname__}'
        raise TypeError(f'{target!r} resolved to a {value_type}, expected an `Agent`')
    return value


def _read_prompt() -> str:
    """Return the prompt, from the `prompt` input or the file `prompt-file` names."""
    prompt = os.environ.get('PAI_PROMPT', '')
    prompt_file = os.environ.get('PAI_PROMPT_FILE', '')
    if bool(prompt) == bool(prompt_file):
        raise ValueError('set exactly one of `prompt` and `prompt-file`')

    if prompt_file:
        try:
            prompt = Path(prompt_file).read_text(encoding='utf-8')
        except OSError as error:
            raise ValueError(f'could not read prompt file {prompt_file!r}: {error}') from error
        if not prompt:
            raise ValueError(f'prompt file {prompt_file!r} is empty')

    return prompt


def _append_github_output(path: str, output: str) -> None:
    """Publish `output` as the step's `result` output.

    The delimiter is random and checked against the output because a model writes whatever it
    likes: a run whose output happened to contain the delimiter on a line of its own could
    otherwise close the value early and leave the rest of it parsed as further outputs.
    """
    output_lines = set(output.splitlines())
    delimiter = f'EOF_{secrets.token_hex(16)}'
    while delimiter in output_lines:
        delimiter = f'EOF_{secrets.token_hex(16)}'

    with Path(path).open('a', encoding='utf-8') as output_file:
        output_file.write(f'result<<{delimiter}\n{output}\n{delimiter}\n')


def _append_step_summary(path: str, output: str) -> None:
    with Path(path).open('a', encoding='utf-8') as summary_file:
        summary_file.write(output)
        if not output.endswith('\n'):
            summary_file.write('\n')


def _print_output(output: str) -> None:
    """Print the agent's output with workflow-command processing turned off around it.

    The runner reads workflow commands such as `::add-mask::` out of a step's own stdout, so a
    model that writes one is
    steering the workflow rather than answering: masking text, faking annotations, or stopping log
    processing for everything after it. Bracketing the output with a random stop token makes the
    runner read all of it as text. Outside Actions nothing reads those lines, and printing the
    markers would only be noise.
    """
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        print(output)
        return

    token = f'PAI_{secrets.token_hex(16)}'
    while token in output:
        token = f'PAI_{secrets.token_hex(16)}'

    print(f'::stop-commands::{token}')
    try:
        print(output)
    finally:
        print(f'::{token}::')


def _configure_observability() -> None:
    """Send the run to Logfire when the workflow's environment says where to.

    A run nobody watched leaves nothing behind but the job log, so the credential the workflow
    already had to provide is also what turns tracing on. This happens before the agent is
    imported, so an agent that configures Logfire itself still has the last word.
    """
    if not (os.environ.get('LOGFIRE_TOKEN') or os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT')):
        return

    try:
        import logfire
    except ImportError:
        print(
            'warning: the environment asks for tracing, but `logfire` is not installed; '
            'add it with `pip-install: logfire`',
            file=sys.stderr,
        )
        return

    logfire.configure(send_to_logfire='if-token-present', console=False)
    logfire.instrument_pydantic_ai()


def main() -> int:
    """Resolve the action's inputs, run the agent, and publish what it said."""
    try:
        prompt = _read_prompt()
    except ValueError as error:
        print(f'error: {error}', file=sys.stderr)
        return 2

    _configure_observability()

    target = os.environ.get('PAI_AGENT', '').strip()
    if not target:
        print('error: `agent` is required', file=sys.stderr)
        return 2

    try:
        agent = _resolve_agent(target)
    except Exception as error:
        print(f'error: could not resolve agent {target!r}: {error}', file=sys.stderr)
        return 2

    # An empty `model` leaves whichever model the agent itself carries: a Python agent is usually
    # built with one, and an agent spec can name one. The input overrides both, and when neither
    # supplies a model, Pydantic AI says so itself when the run starts.
    model = os.environ.get('PAI_MODEL', '').strip() or None

    try:
        output = str(agent.run_sync(prompt, model=model).output)
    except Exception as error:
        print(f'error: agent run failed: {error}', file=sys.stderr)
        return 1

    if github_output := os.environ.get('GITHUB_OUTPUT'):
        try:
            _append_github_output(github_output, output)
        except OSError as error:
            print(f'error: could not write `GITHUB_OUTPUT`: {error}', file=sys.stderr)
            return 1

    if step_summary := os.environ.get('GITHUB_STEP_SUMMARY'):
        try:
            _append_step_summary(step_summary, output)
        except OSError as error:
            print(f'error: could not write `GITHUB_STEP_SUMMARY`: {error}', file=sys.stderr)
            return 1

    _print_output(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
