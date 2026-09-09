"""Regression tests for the JavaScript inside `gh-aw/pydantic.md`.

The definition is the shipped artifact, so the scripts under test are read out of it
rather than copied here: gh-aw runs those exact bytes, and a copy would let the two
drift. They are JavaScript, so `node` runs them, and a shell script standing in
for the interpreter records the argv and environment the launcher hands it. The
Python program the launcher passes to `-c` is then run with the real interpreter,
using the recorded bytes. The `log-parser` is driven the same way, over a log
holding the line shapes a run emits.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ConfigDict, Field

if shutil.which('node') is None:  # pragma: no cover
    pytest.skip('the gh-aw harness script is JavaScript and needs node', allow_module_level=True)

DEFINITION = Path(__file__).parents[2] / 'gh-aw' / 'pydantic.md'

_CLI_PACKAGES = ('argcomplete', 'prompt_toolkit', 'pyperclip', 'rich')

requires_cli = pytest.mark.skipif(
    any(importlib.util.find_spec(package) is None for package in _CLI_PACKAGES),
    reason='running the CLI needs the pydantic-ai `cli` extra',
)
requires_safe_path = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason='the launcher is invoked with `-P`, which arrived in Python 3.11',
)

# Stands in for gh-aw's own helper module, which the harness script requires next to
# itself. The two functions the script calls mirror the upstream shapes: the resolved
# endpoint carries the models-listing origin, and the derived base URL is that origin
# plus the path prefix with a trailing `/models` removed.
REFLECT_STUB = """
const payload = JSON.parse(process.env.GH_AW_TEST_REFLECT || '{"endpoints": []}');

module.exports = {
  fetchAWFReflect: async () => ({ ok: true, reflectData: payload }),
  resolveProviderEndpointFromReflect: ({ provider }) => {
    const configured = payload.endpoints.filter(entry => entry.configured === true);
    const matched = configured.find(entry => entry.provider === provider) || configured[0];
    return { provider, endpointProvider: matched.provider, baseUrl: new URL(matched.models_url).origin };
  },
  deriveBaseUrlFromModelsURL: modelsUrl => {
    const parsed = new URL(modelsUrl);
    return `${parsed.origin}${parsed.pathname.replace(/\\/models\\/?$/i, "")}`;
  },
};
"""

RECORDER = """import json
import os
import sys
from pathlib import Path

record = {'argv': sys.argv[1:], 'env': dict(os.environ), 'cwd': os.getcwd()}
Path(os.environ['GH_AW_TEST_RECORD']).write_text(json.dumps(record))
"""

AGENT_MODULE = """import os
from pathlib import Path

from pydantic_ai import Agent

with Path(os.environ['GH_AW_TEST_IMPORTS']).open('a') as handle:
    handle.write('NAME\\n')

agent = Agent(name='NAME', instructions='Answer briefly.')
"""

LOG_PARSER_DRIVER = """
const fs = require("fs");
const { parseLog } = require("./parser.cjs");

process.stdout.write(JSON.stringify(parseLog(fs.readFileSync(process.argv[2], "utf8"))));
"""

# One line of each shape the parser branches on, in the order a run emits them: an
# infra line it drops, text, a usage record, a reply that is bare JSON, a tool call,
# and text after it.
AGENT_LOG = """[pydantic-ai] starting the agent
{"msg": "Reading the issue."}
{"input_tokens": 10, "output_tokens": 5}
{"answer": 42}
{"type": "tool_call", "tool": "read_file", "msg": "read 12 lines"}
{"msg": "Done."}
"""

PROMPT = 'summarize the issue'

# One configured endpoint per api-proxy backend, in the shape `/reflect` reports.
# `models_url` carries the `/v1` prefix that separates the two base URLs: the
# OpenAI-compatible client appends `/chat/completions` to it, the Anthropic client
# appends `/v1/messages` to the origin.
REFLECT_PAYLOAD = json.dumps(
    {
        'endpoints': [
            {'provider': 'anthropic', 'configured': True, 'models_url': 'http://host.docker.internal:10001/v1/models'},
            {'provider': 'openai', 'configured': True, 'models_url': 'http://host.docker.internal:10000/v1/models'},
            {'provider': 'github', 'configured': True, 'models_url': 'http://host.docker.internal:10002/v1/models'},
        ]
    }
)


class _Behaviors(BaseModel):
    model_config = ConfigDict(extra='ignore')

    harness_script: str = Field(alias='harness-script')
    log_parser: str = Field(alias='log-parser')


class _Engine(BaseModel):
    model_config = ConfigDict(extra='ignore')

    behaviors: _Behaviors


class _Frontmatter(BaseModel):
    model_config = ConfigDict(extra='ignore')

    engine: _Engine


class _Invocation(BaseModel):
    """What the launcher handed the interpreter."""

    argv: list[str]
    env: dict[str, str]
    cwd: str

    @property
    def target(self) -> str:
        return self.argv[3]

    @property
    def cli_args(self) -> list[str]:
        return self.argv[4:]

    @property
    def program(self) -> str:
        return self.argv[2]

    @property
    def python_path(self) -> list[Path]:
        return [Path(entry) for entry in self.env['PYTHONPATH'].split(':')]


class _Block(BaseModel):
    """One content block of a reconstructed log entry."""

    model_config = ConfigDict(extra='ignore')

    type: str
    text: str = ''
    name: str = ''
    id: str = ''
    tool_use_id: str = ''
    content: str = ''


class _Message(BaseModel):
    content: list[_Block] = Field(default_factory=list[_Block])


class _Entry(BaseModel):
    model_config = ConfigDict(extra='ignore')

    type: str
    message: _Message = Field(default_factory=_Message)
    num_turns: int = 0
    usage: dict[str, int] = Field(default_factory=dict[str, int])


class _ParsedLog(BaseModel):
    """What the log parser reconstructed from a run's output."""

    model_config = ConfigDict(extra='ignore')

    markdown: str
    log_entries: list[_Entry] = Field(alias='logEntries')

    @property
    def texts(self) -> list[str]:
        return [block.text for entry in self.log_entries for block in entry.message.content if block.type == 'text']


def behaviors() -> _Behaviors:
    """The `engine.behaviors` block, read from the definition gh-aw consumes."""
    lines = DEFINITION.read_text(encoding='utf-8').splitlines()
    frontmatter: object = yaml.safe_load('\n'.join(lines[1 : lines.index('---', 1)]))
    return _Frontmatter.model_validate(frontmatter).engine.behaviors


def launch(tmp_path: Path, env: dict[str, str]) -> _Invocation:
    """Run the harness script against an interpreter that records instead of running."""
    actions = tmp_path / 'actions'
    actions.mkdir(parents=True, exist_ok=True)
    (actions / 'harness.cjs').write_text(behaviors().harness_script, encoding='utf-8')
    (actions / 'awf_reflect.cjs').write_text(REFLECT_STUB, encoding='utf-8')

    # `pythonLocation` is what `actions/setup-python` exports, and the script joins
    # `bin/python3` onto it.
    python_location = tmp_path / 'python'
    binaries = python_location / 'bin'
    binaries.mkdir(parents=True, exist_ok=True)
    recorder = python_location / 'recorder.py'
    recorder.write_text(RECORDER, encoding='utf-8')
    interpreter = binaries / 'python3'
    interpreter.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(recorder))} "$@"\n')
    interpreter.chmod(0o755)

    workspace = tmp_path / 'workspace'
    workspace.mkdir(parents=True, exist_ok=True)
    # gh-aw's config adapter writes the MCP config into this tree on the host runner,
    # and the agent step mounts it into the sandbox read-only.
    runner_temp = tmp_path / 'runner-temp'
    runner_temp.mkdir(parents=True, exist_ok=True)
    # `os.tmpdir()`, where the harness script puts the generated module. Pointing it
    # into the test's own directory keeps that write out of the real /tmp.
    sandbox_tmp = tmp_path / 'sandbox-tmp'
    sandbox_tmp.mkdir(parents=True, exist_ok=True)
    prompt = tmp_path / 'prompt.md'
    prompt.write_text(PROMPT, encoding='utf-8')
    record = tmp_path / 'record.json'

    completed = subprocess.run(
        ['node', str(actions / 'harness.cjs'), 'pai'],
        env={
            'PATH': os.environ['PATH'],
            'HOME': str(tmp_path / 'home'),
            'TMPDIR': str(sandbox_tmp),
            'RUNNER_TEMP': str(runner_temp),
            'GITHUB_WORKSPACE': str(workspace),
            'GH_AW_PROMPT': str(prompt),
            'GH_AW_TEST_RECORD': str(record),
            'pythonLocation': str(python_location),
            **env,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return _Invocation.model_validate_json(record.read_text(encoding='utf-8'))


def parse_log(tmp_path: Path, log: str) -> _ParsedLog:
    """Run the shipped `log-parser` over `log`, exported the way gh-aw exports it."""
    # gh-aw wraps the block in a module that exports the `parseLog` the block defines,
    # so the driver requires it under that name.
    parser = tmp_path / 'parser.cjs'
    parser.write_text(f'{behaviors().log_parser}\nmodule.exports = {{ parseLog }};\n', encoding='utf-8')
    driver = tmp_path / 'driver.cjs'
    driver.write_text(LOG_PARSER_DRIVER, encoding='utf-8')
    agent_log = tmp_path / 'agent.log'
    agent_log.write_text(log, encoding='utf-8')

    completed = subprocess.run(
        ['node', str(driver), str(agent_log)],
        env={'PATH': os.environ['PATH']},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return _ParsedLog.model_validate_json(completed.stdout)


def gateway_config(tmp_path: Path) -> Path:
    """The config adapter's output path, as the harness script resolves it."""
    return tmp_path / 'runner-temp' / 'gh-aw' / 'mcp-config' / 'mcp-servers.json'


def proxy_env(provider: str, model: str) -> dict[str, str]:
    """The environment a compiled workflow reaches the api-proxy with."""
    return {
        'GH_AW_LLM_PROVIDER': provider,
        'PAI_MODEL': model,
        'AWF_REFLECT_ENABLED': '1',
        'GH_AW_TEST_REFLECT': REFLECT_PAYLOAD,
    }


def test_the_default_target_is_the_generated_module(tmp_path: Path) -> None:
    invocation = launch(
        tmp_path,
        {**proxy_env('github', 'copilot/claude-sonnet-4-5'), 'COPILOT_GITHUB_TOKEN': 'a-token'},
    )

    assert invocation.argv[:2] == ['-P', '-c']
    assert invocation.target == 'gh_aw_agent:agent'
    assert invocation.cli_args == ['-a', 'gh_aw_agent:agent', '-m', 'openai-chat:claude-sonnet-4.5', PROMPT]
    # The module is written to a private directory under `os.tmpdir()`, never into the
    # checkout: a package committed under a directory the engine puts on PYTHONPATH
    # would shadow an installed one for the whole run.
    (module_dir,) = invocation.python_path
    assert module_dir.parent == tmp_path / 'sandbox-tmp'
    assert (module_dir / 'gh_aw_agent.py').read_text().startswith('from pydantic_ai')
    assert not (tmp_path / 'workspace' / '.pydantic-ai').exists()
    # gh-aw sets this for the copilot backend; the proxy holds the real credential,
    # so the agent has no use for it.
    assert 'COPILOT_GITHUB_TOKEN' not in invocation.env


def test_the_checkout_is_off_the_import_path_without_pai_agent(tmp_path: Path) -> None:
    invocation = launch(tmp_path, proxy_env('openai', 'openai/gpt-5'))

    workspace = tmp_path / 'workspace'
    assert not any(entry == workspace or workspace in entry.parents for entry in invocation.python_path)
    # The CLI still runs in the checkout, which is what the agent reads and writes.
    assert invocation.cwd == str(workspace)


def test_pai_agent_replaces_the_target_and_puts_the_checkout_on_the_path(tmp_path: Path) -> None:
    invocation = launch(tmp_path, {**proxy_env('openai', 'openai/gpt-5'), 'PAI_AGENT': 'my_agent:agent'})

    workspace = tmp_path / 'workspace'
    assert invocation.target == 'my_agent:agent'
    assert invocation.cli_args[:2] == ['-a', 'my_agent:agent']
    # No module is generated, so the checkout is the whole path.
    assert invocation.python_path == [workspace]
    assert list((tmp_path / 'sandbox-tmp').iterdir()) == []


def test_a_spec_file_target_reaches_the_cli_unchanged(tmp_path: Path) -> None:
    invocation = launch(tmp_path, {**proxy_env('openai', 'openai/gpt-5'), 'PAI_AGENT': 'agent.yml'})

    assert invocation.target == 'agent.yml'
    assert invocation.cli_args[:2] == ['-a', 'agent.yml']


def test_mcp_config_is_passed_only_when_the_gateway_wrote_one(tmp_path: Path) -> None:
    without = launch(tmp_path / 'without', proxy_env('openai', 'openai/gpt-5'))

    assert '--mcp-config' not in without.cli_args

    config = gateway_config(tmp_path / 'with')
    config.parent.mkdir(parents=True)
    config.write_text('{"mcpServers": {}}', encoding='utf-8')
    with_config = launch(tmp_path / 'with', proxy_env('openai', 'openai/gpt-5'))

    assert with_config.cli_args[2:4] == ['--mcp-config', str(config)]


def test_a_committed_mcp_config_is_not_passed_to_the_cli(tmp_path: Path) -> None:
    """Only the host-written config reaches `--mcp-config`.

    `load_mcp_toolsets` starts a stdio server the config names, so a file the
    repository can commit must not be a candidate for the flag.
    """
    committed = tmp_path / 'workspace' / '.pydantic-ai' / 'mcp.json'
    committed.parent.mkdir(parents=True)
    committed.write_text('{"mcpServers": {"local": {"command": "python3", "args": ["x.py"]}}}', encoding='utf-8')

    invocation = launch(tmp_path, proxy_env('openai', 'openai/gpt-5'))

    assert '--mcp-config' not in invocation.cli_args
    assert str(committed) not in invocation.cli_args


def test_the_anthropic_backend_is_addressed_with_the_messages_api(tmp_path: Path) -> None:
    invocation = launch(tmp_path, proxy_env('anthropic', 'anthropic/claude-sonnet-4-5'))

    assert invocation.cli_args[-2] == 'anthropic:claude-sonnet-4-5'
    # The Anthropic client appends `/v1/messages`, so it gets the endpoint's origin
    # rather than the `/v1` base the OpenAI-compatible client needs.
    assert invocation.env['ANTHROPIC_BASE_URL'] == 'http://host.docker.internal:10001'
    assert invocation.env['ANTHROPIC_API_KEY'] == 'awf-anthropic-proxy'
    assert 'OPENAI_BASE_URL' not in invocation.env


@pytest.mark.parametrize(
    ('provider', 'model', 'port'),
    [('github', 'copilot/gpt-5', 10002), ('openai', 'openai/gpt-5', 10000)],
)
def test_openai_shaped_backends_stay_on_chat_completions(tmp_path: Path, provider: str, model: str, port: int) -> None:
    invocation = launch(tmp_path, proxy_env(provider, model))

    assert invocation.cli_args[-2] == 'openai-chat:gpt-5'
    assert invocation.env['OPENAI_BASE_URL'] == f'http://host.docker.internal:{port}/v1'
    assert 'ANTHROPIC_BASE_URL' not in invocation.env


@pytest.mark.parametrize('model', ['anthropic/claude-sonnet-4-5', 'copilot/gpt-5', 'openai/gpt-5'])
def test_pai_base_url_keeps_every_provider_on_chat_completions(tmp_path: Path, model: str) -> None:
    invocation = launch(
        tmp_path,
        {'GH_AW_LLM_PROVIDER': 'openai', 'PAI_MODEL': model, 'PAI_BASE_URL': 'https://endpoint.example.com/v1'},
    )

    assert invocation.cli_args[-2].startswith('openai-chat:')
    assert invocation.env['OPENAI_BASE_URL'] == 'https://endpoint.example.com/v1'
    assert 'ANTHROPIC_BASE_URL' not in invocation.env


@requires_safe_path
class TestLauncherProgram:
    """The `-c` program, run by the real interpreter with the bytes the launcher sends."""

    @staticmethod
    def program(tmp_path: Path) -> str:
        return launch(tmp_path / 'launch', proxy_env('openai', 'openai/gpt-5')).program

    @staticmethod
    def run(tmp_path: Path, target: str, *cli_args: str) -> subprocess.CompletedProcess[str]:
        """Run the launcher over a module directory that a checkout file shadows."""
        program = TestLauncherProgram.program(tmp_path)
        module_dir = tmp_path / 'module'
        module_dir.mkdir(parents=True, exist_ok=True)
        workspace = tmp_path / 'workspace'
        workspace.mkdir(parents=True, exist_ok=True)
        imports = tmp_path / 'imports.txt'

        (module_dir / 'gh_aw_agent.py').write_text(AGENT_MODULE.replace('NAME', 'module-directory'), encoding='utf-8')
        # `load_agent` prepends the working directory to `sys.path`, so this is the
        # file the CLI would reach on its own.
        (workspace / 'gh_aw_agent.py').write_text(AGENT_MODULE.replace('NAME', 'checkout'), encoding='utf-8')

        return subprocess.run(
            [sys.executable, '-P', '-c', program, target, *cli_args],
            cwd=workspace,
            env={
                'PATH': os.environ['PATH'],
                'HOME': str(tmp_path / 'home'),
                'PYTHONPATH': str(module_dir),
                'GH_AW_TEST_IMPORTS': str(imports),
                'PYTHONIOENCODING': 'utf-8',
            },
            capture_output=True,
            text=True,
            check=False,
        )

    @requires_cli
    def test_the_agent_module_is_imported_once_and_not_from_the_checkout(self, tmp_path: Path) -> None:
        completed = self.run(tmp_path, 'gh_aw_agent:agent', '-a', 'gh_aw_agent:agent', '-m', 'test', 'hello')

        assert completed.returncode == 0, completed.stderr
        # One line, from the module directory: the CLI reused the module the launcher
        # imported rather than importing anything a second time or reaching the
        # checkout copy.
        assert (tmp_path / 'imports.txt').read_text(encoding='utf-8') == 'module-directory\n'

    def test_a_target_that_is_not_an_agent_names_what_it_found(self, tmp_path: Path) -> None:
        (tmp_path / 'module').mkdir(parents=True, exist_ok=True)
        (tmp_path / 'module' / 'not_an_agent.py').write_text('agent = 1\n', encoding='utf-8')

        completed = self.run(tmp_path, 'not_an_agent:agent', '-a', 'not_an_agent:agent', '-m', 'test', 'hello')

        assert completed.returncode != 0
        assert 'TypeError: not_an_agent:agent is int, not pydantic_ai.Agent' in completed.stderr

    def test_an_agent_that_raises_on_import_fails_with_its_traceback(self, tmp_path: Path) -> None:
        (tmp_path / 'module').mkdir(parents=True, exist_ok=True)
        (tmp_path / 'module' / 'broken_agent.py').write_text(
            "raise RuntimeError('the agent could not be built')\n", encoding='utf-8'
        )

        completed = self.run(tmp_path, 'broken_agent:agent', '-a', 'broken_agent:agent', '-m', 'test', 'hello')

        assert completed.returncode != 0
        assert 'Traceback (most recent call last)' in completed.stderr
        assert 'RuntimeError: the agent could not be built' in completed.stderr
        # The message `pai` prints instead of a traceback when its own load fails.
        assert 'Could not load agent' not in completed.stderr + completed.stdout


class TestLogParser:
    """The `log-parser` gh-aw runs over the agent step's output."""

    def test_a_reply_that_is_bare_json_is_kept(self, tmp_path: Path) -> None:
        parsed = parse_log(tmp_path, AGENT_LOG)

        # The line carries no field the parser reads text from, so it survives only as
        # the line itself.
        assert parsed.texts == ['Reading the issue.\n{"answer": 42}', 'Done.']

    def test_a_usage_record_is_counted_and_not_repeated_as_text(self, tmp_path: Path) -> None:
        parsed = parse_log(tmp_path, AGENT_LOG)

        assert parsed.log_entries[-1].usage == {'input_tokens': 10, 'output_tokens': 5}
        assert 'input_tokens' not in '\n'.join(parsed.texts)
        assert '**Tokens:** 15' in parsed.markdown

    def test_infra_lines_are_dropped_and_a_tool_call_becomes_a_pair(self, tmp_path: Path) -> None:
        parsed = parse_log(tmp_path, AGENT_LOG)

        assert '[pydantic-ai]' not in '\n'.join(parsed.texts)
        assert [entry.type for entry in parsed.log_entries] == [
            'system',
            'assistant',
            'assistant',
            'user',
            'assistant',
            'result',
        ]
        (call,) = parsed.log_entries[2].message.content
        (result,) = parsed.log_entries[3].message.content
        assert (call.type, call.name) == ('tool_use', 'read_file')
        assert (result.type, result.tool_use_id, result.content) == ('tool_result', call.id, 'read 12 lines')
        assert parsed.log_entries[-1].num_turns == 2
