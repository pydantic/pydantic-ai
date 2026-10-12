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

import asyncio
import importlib.util
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType

import pytest
import yaml
from pydantic import BaseModel, ConfigDict, Field

from pydantic_ai import Agent, ModelMessage, RunContext, ToolReturnPart
from pydantic_ai.messages import PartEndEvent, TextPart
from pydantic_ai.models.function import AgentInfo, DeltaThinkingPart, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

if shutil.which('node') is None:  # pragma: no cover
    pytest.skip('the gh-aw harness script is JavaScript and needs node', allow_module_level=True)

DEFINITION = Path(__file__).parents[3] / 'src' / 'pydantic_ai_harness' / 'gh-aw' / 'pydantic.md'
CLAI2_SOURCE = Path(__file__).parents[3] / 'src' / 'pydantic_clai2'

requires_clai2 = pytest.mark.skipif(
    importlib.util.find_spec('pydantic_clai2') is None,
    reason='running the launcher requires pydantic-clai2',
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

record = {'argv': sys.argv[1:], 'env': dict(os.environ), 'cwd': os.getcwd(), 'stdin': sys.stdin.read()}
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

ADD_MASK_REDACTION = r"""
function collectAddMaskedValues(logContent) {
  const values = new Set();
  for (const line of logContent.split("\n")) {
    const match = line.match(/::add-mask::(.*)$/);
    if (match && match[1].trim()) values.add(match[1].trim());
  }
  return [...values].sort((a, b) => b.length - a.length);
}
function redactArtifactMaskedValues(content, maskedValues) {
  const redact = value => {
    if (typeof value === "string") {
      return maskedValues.reduce((text, masked) => text.split(masked).join("***"), value);
    }
    if (Array.isArray(value)) return value.map(redact);
    if (value && typeof value === "object") {
      return Object.fromEntries(Object.entries(value).map(([key, nested]) => [redact(key), redact(nested)]));
    }
    return value;
  };
  return JSON.stringify(redact(JSON.parse(content)));
}
module.exports = { collectAddMaskedValues, redactArtifactMaskedValues };
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

    version: str
    behaviors: _Behaviors


class _PreAgentStep(BaseModel):
    model_config = ConfigDict(extra='ignore')

    run: str


class _Frontmatter(BaseModel):
    model_config = ConfigDict(extra='ignore')

    engine: _Engine
    pre_agent_steps: list[_PreAgentStep] = Field(alias='pre-agent-steps')


class _Invocation(BaseModel):
    """What the launcher handed the interpreter."""

    argv: list[str]
    env: dict[str, str]
    cwd: str
    stdout: str
    stdin: str = ''

    @property
    def target(self) -> str:
        return self.argv[3]

    @property
    def cli_args(self) -> list[str]:
        return self.argv[5:] if self.prompt_file.exists() else self.argv[4:]

    @property
    def program(self) -> str:
        return self.argv[2]

    @property
    def python_path(self) -> list[Path]:
        return [Path(entry) for entry in self.env['PYTHONPATH'].split(':')]

    @property
    def frame_key(self) -> str:
        match = re.search(r'\x1eGH-AW-SESSION-KEY:([0-9a-f]{32})\x1e', self.stdout)
        assert match is not None, self.stdout
        return match.group(1)

    @property
    def prompt_file(self) -> Path:
        return Path(self.argv[4])


class _CanonicalEvent(BaseModel):
    model_config = ConfigDict(extra='allow')

    type: str
    data: dict[str, object]
    id: str | None = None
    parent_id: str | None = Field(default=None, alias='parentId')
    timestamp: str | None = None
    session_id: str | None = None


class _ParsedLog(BaseModel):
    """What the log parser reconstructed from a run's output."""

    model_config = ConfigDict(extra='ignore')

    markdown: str
    log_entries: list[_CanonicalEvent] = Field(alias='logEntries')


def definition() -> _Frontmatter:
    """Read the frontmatter gh-aw consumes."""
    lines = DEFINITION.read_text(encoding='utf-8').splitlines()
    frontmatter: object = yaml.safe_load('\n'.join(lines[1 : lines.index('---', 1)]))
    return _Frontmatter.model_validate(frontmatter)


def behaviors() -> _Behaviors:
    """The `engine.behaviors` block, read from the definition gh-aw consumes."""
    return definition().engine.behaviors


def test_install_pins_clai2_and_installs_spec_extra_for_yaml_agents() -> None:
    requirements = [
        argument
        for step in definition().pre_agent_steps
        for line in step.run.splitlines()
        if line.strip().startswith('python3 -P -m pip install ')
        for argument in shlex.split(line)
        if argument.startswith('pydantic-ai-slim[')
    ]
    (requirement,) = requirements
    extras = requirement.split('[', 1)[1].split(']', 1)[0].split(',')
    assert 'spec' in extras
    harness_requirements = [
        argument
        for step in definition().pre_agent_steps
        for line in step.run.splitlines()
        if line.strip().startswith('python3 -P -m pip install ')
        for argument in shlex.split(line)
        if argument.startswith(('pydantic-ai-harness==', 'pydantic-clai2=='))
    ]
    assert harness_requirements == [
        'pydantic-ai-harness==${GH_AW_ENGINE_VERSION}',
        'pydantic-clai2==${GH_AW_ENGINE_VERSION}',
    ]
    assert definition().engine.version == '0.55.0'
    assert requirement == 'pydantic-ai-slim[anthropic,openai,mcp,spec]>=2.54.0'


def launch(
    tmp_path: Path,
    env: dict[str, str],
    *,
    extra_python_path: Path | None = None,
    prompt_text: str = PROMPT,
) -> _Invocation:
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
    prompt.write_text(prompt_text, encoding='utf-8')
    record = tmp_path / 'record.json'

    pythonpath_env: dict[str, str] = {'PYTHONPATH': str(extra_python_path)} if extra_python_path is not None else {}
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
            **pythonpath_env,
            **env,
        },
        capture_output=True,
        text=True,
        input='',
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    recorded: object = json.loads(record.read_text(encoding='utf-8'))
    assert isinstance(recorded, dict)
    recorded['stdout'] = completed.stdout
    return _Invocation.model_validate(recorded)


def parse_log(tmp_path: Path, log: str) -> _ParsedLog:
    """Run the shipped `log-parser` over `log`, exported the way gh-aw exports it."""
    # gh-aw wraps the block in a module that exports the `parseLog` the block defines,
    # so the driver requires it under that name.
    parser = tmp_path / 'parser.cjs'
    parser.write_text(f'{behaviors().log_parser}\nmodule.exports = {{ parseLog }};\n', encoding='utf-8')
    (tmp_path / 'add_mask_redaction.cjs').write_text(ADD_MASK_REDACTION, encoding='utf-8')
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


def run_invocation(invocation: _Invocation, *extra_cli_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *invocation.argv, *extra_cli_args],
        cwd=invocation.cwd,
        env=invocation.env,
        input=invocation.stdin,
        capture_output=True,
        text=True,
        check=False,
    )


FRAME_KEY = '0123456789abcdef0123456789abcdef'
FRAME_HEADER = f'\x1eGH-AW-SESSION-KEY:{FRAME_KEY}\x1e'


def frame_event(event_type: str, data: dict[str, object], sequence: int) -> str:
    event = {
        'type': event_type,
        'data': data,
        'id': f'pydantic-ai-run-{sequence}',
        'parentId': None,
        'timestamp': '2026-10-06T00:00:00Z',
        'session_id': 'conversation-id',
    }
    return f'\x1eGH-AW-SESSION/{FRAME_KEY} {json.dumps(event, separators=(",", ":"))}'


def framed_log(*records: str) -> str:
    return '\n'.join((FRAME_HEADER, *records))


def import_generated_agent(invocation: _Invocation, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the exact generated wrapper with the environment the launcher prepared."""
    monkeypatch.chdir(invocation.cwd)
    monkeypatch.setattr(sys, 'path', [str(entry) for entry in invocation.python_path] + sys.path)
    monkeypatch.setenv('GH_AW_SESSION_FRAME_KEY', invocation.frame_key)
    if configured_agent := invocation.env.get('PAI_AGENT'):
        monkeypatch.setenv('PAI_AGENT', configured_agent)
    else:
        monkeypatch.delenv('PAI_AGENT', raising=False)
    module_path = invocation.python_path[0] / 'gh_aw_agent.py'
    spec = importlib.util.spec_from_file_location(f'gh_aw_agent_{module_path.parent.name}', module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('target', ['custom_agent:agent', 'custom_agent.agent'])
def test_wrapper_imports_a_custom_agent_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    import_log = tmp_path / 'imports.txt'
    (workspace / 'custom_agent.py').write_text(
        'from pathlib import Path\nfrom pydantic_ai import Agent\n'
        f"with Path({str(import_log)!r}).open('a') as log: log.write('imported\\n')\n"
        "agent = Agent(name='custom')\n",
        encoding='utf-8',
    )
    invocation = launch(tmp_path, {**proxy_env('openai', 'openai/gpt-5'), 'PAI_AGENT': target})

    # Each parameter value uses a different checkout; do not reuse the module
    # imported by the preceding value.
    monkeypatch.delitem(sys.modules, 'custom_agent', raising=False)
    module = import_generated_agent(invocation, monkeypatch)

    agent = getattr(module, 'agent')
    assert isinstance(agent, Agent)
    assert agent.name == 'custom'
    assert import_log.read_text(encoding='utf-8') == 'imported\n'


@pytest.mark.parametrize(
    ('suffix', 'contents', 'name'),
    [
        ('.json', '{"name":"json-agent","instructions":"Answer briefly."}', 'json-agent'),
        ('.yaml', 'name: yaml-agent\ninstructions: Answer briefly.\n', 'yaml-agent'),
    ],
)
def test_wrapper_imports_json_and_yaml_agent_specs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str, contents: str, name: str
) -> None:
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    spec_path = workspace / f'agent{suffix}'
    spec_path.write_text(contents, encoding='utf-8')
    invocation = launch(tmp_path, {**proxy_env('openai', 'openai/gpt-5'), 'PAI_AGENT': str(spec_path)})

    module = import_generated_agent(invocation, monkeypatch)

    agent = getattr(module, 'agent')
    assert isinstance(agent, Agent)
    assert agent.name == name


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
    assert invocation.cli_args == ['-a', 'gh_aw_agent:agent', '-m', 'openai-chat:claude-sonnet-4.5']
    assert invocation.prompt_file.read_text(encoding='utf-8') == PROMPT
    frame_key = invocation.frame_key
    assert re.fullmatch(r'[0-9a-f]{32}', frame_key)
    assert invocation.stdout == f'\x1eGH-AW-SESSION-KEY:{frame_key}\x1e\n'
    # The module is written to a private directory under `os.tmpdir()`, never into the
    # checkout: a package committed under a directory the engine puts on PYTHONPATH
    # would shadow an installed one for the whole run.
    module_dir = invocation.python_path[0]
    assert module_dir.parent == tmp_path / 'sandbox-tmp'
    assert 'agent = Agent(' in (module_dir / 'gh_aw_agent.py').read_text()
    assert not (tmp_path / 'workspace' / '.pydantic-ai').exists()
    # gh-aw sets this for the copilot backend; the proxy holds the real credential,
    # so the agent has no use for it.
    assert 'COPILOT_GITHUB_TOKEN' not in invocation.env


def test_a_large_prompt_is_not_passed_as_an_oversized_process_argument(tmp_path: Path) -> None:
    prompt = 'x' * (1024 * 1024)
    invocation = launch(tmp_path, proxy_env('openai', 'openai/gpt-5'), prompt_text=prompt)

    assert max(map(len, invocation.argv)) < 128 * 1024
    assert invocation.prompt_file.read_text(encoding='utf-8') == prompt


def test_the_frame_key_is_sent_out_of_band_and_removed_from_the_child_environment(tmp_path: Path) -> None:
    invocation = launch(tmp_path, proxy_env('openai', 'openai/gpt-5'))

    assert invocation.env.get('GH_AW_SESSION_FRAME_KEY') is None
    assert invocation.stdin == f'{invocation.frame_key}\n'


@pytest.mark.subprocess(reason='runs the generated gh-aw launcher program as a real script')
@requires_clai2
def test_an_imported_agent_does_not_pass_the_frame_key_to_its_children(tmp_path: Path) -> None:
    invocation = launch(
        tmp_path,
        {**proxy_env('openai', 'openai/gpt-5'), 'PAI_AGENT': 'custom_agent:agent'},
        extra_python_path=CLAI2_SOURCE,
    )
    child_env_path = Path(invocation.cwd) / 'child-environment.json'
    custom_agent = Path(invocation.cwd) / 'custom_agent.py'
    custom_agent.write_text(
        """import json
import os
import subprocess
import sys
from pathlib import Path

from pydantic_ai import Agent

child = subprocess.run(
    [sys.executable, '-c', "import json, os; from pathlib import Path; p = Path(f'/proc/{os.getppid()}/environ'); print(json.dumps({'key': os.environ.get('GH_AW_SESSION_FRAME_KEY'), 'parent_has_key': b'GH_AW_SESSION_FRAME_KEY=' in p.read_bytes() if p.exists() else None}))"],
    capture_output=True,
    text=True,
    check=True,
)
Path(__file__).with_name('child-environment.json').write_text(child.stdout, encoding='utf-8')
agent = Agent(name='custom', instructions='Answer briefly.')
""",
        encoding='utf-8',
    )

    completed = run_invocation(invocation, '--gh-aw-invalid')

    assert completed.returncode == 2
    assert 'unrecognized arguments: --gh-aw-invalid' in completed.stderr
    child_environment = json.loads(child_env_path.read_text(encoding='utf-8'))
    assert child_environment['key'] is None
    parent_has_key = child_environment['parent_has_key']
    assert parent_has_key is None or parent_has_key is False


def test_the_checkout_is_off_the_import_path_without_pai_agent(tmp_path: Path) -> None:
    invocation = launch(tmp_path, proxy_env('openai', 'openai/gpt-5'))

    workspace = tmp_path / 'workspace'
    assert not any(entry == workspace or workspace in entry.parents for entry in invocation.python_path)
    # The CLI still runs in the checkout, which is what the agent reads and writes.
    assert invocation.cwd == str(workspace)


def test_pai_agent_is_wrapped_and_adds_the_checkout_to_the_import_path(tmp_path: Path) -> None:
    invocation = launch(tmp_path, {**proxy_env('openai', 'openai/gpt-5'), 'PAI_AGENT': 'my_agent:agent'})

    workspace = tmp_path / 'workspace'
    assert invocation.target == 'gh_aw_agent:agent'
    assert invocation.cli_args[:2] == ['-a', 'gh_aw_agent:agent']
    (module_dir,) = invocation.python_path
    assert module_dir.parent == tmp_path / 'sandbox-tmp'
    assert workspace not in invocation.python_path
    assert (module_dir / 'gh_aw_agent.py').is_file()


def test_a_spec_file_is_wrapped_before_it_reaches_the_cli(tmp_path: Path) -> None:
    invocation = launch(tmp_path, {**proxy_env('openai', 'openai/gpt-5'), 'PAI_AGENT': 'agent.yml'})

    assert invocation.target == 'gh_aw_agent:agent'
    assert invocation.cli_args[:2] == ['-a', 'gh_aw_agent:agent']


def test_mcp_config_is_passed_only_when_the_gateway_wrote_one(tmp_path: Path) -> None:
    without = launch(
        tmp_path / 'without',
        {**proxy_env('openai', 'openai/gpt-5'), 'GH_AW_MCP_CONFIG': '/host/override/mcp.json'},
    )

    assert 'GH_AW_MCP_CONFIG' not in without.env

    config = gateway_config(tmp_path / 'with')
    config.parent.mkdir(parents=True)
    config.write_text('{"mcpServers": {}}', encoding='utf-8')
    with_config = launch(tmp_path / 'with', proxy_env('openai', 'openai/gpt-5'))

    assert with_config.env['GH_AW_MCP_CONFIG'] == str(config)
    assert '--mcp-config' not in with_config.cli_args


def test_a_committed_mcp_config_does_not_reach_the_wrapper(tmp_path: Path) -> None:
    """Only the host-written config reaches the adapter.

    `load_mcp_toolsets` starts a stdio server the config names, so a file the
    repository can commit must not be a candidate for the flag.
    """
    committed = tmp_path / 'workspace' / '.pydantic-ai' / 'mcp.json'
    committed.parent.mkdir(parents=True)
    committed.write_text('{"mcpServers": {"local": {"command": "python3", "args": ["x.py"]}}}', encoding='utf-8')

    invocation = launch(tmp_path, proxy_env('openai', 'openai/gpt-5'))

    assert 'GH_AW_MCP_CONFIG' not in invocation.env
    assert str(committed) not in invocation.env['PYTHONPATH']


def test_the_anthropic_backend_is_addressed_with_the_messages_api(tmp_path: Path) -> None:
    invocation = launch(tmp_path, proxy_env('anthropic', 'anthropic/claude-sonnet-4-5'))

    assert invocation.cli_args[3] == 'anthropic:claude-sonnet-4-5'
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

    assert invocation.cli_args[3] == 'openai-chat:gpt-5'
    assert invocation.env['OPENAI_BASE_URL'] == f'http://host.docker.internal:{port}/v1'
    assert 'ANTHROPIC_BASE_URL' not in invocation.env


@pytest.mark.parametrize('model', ['anthropic/claude-sonnet-4-5', 'copilot/gpt-5', 'openai/gpt-5'])
def test_pai_base_url_keeps_every_provider_on_chat_completions(tmp_path: Path, model: str) -> None:
    invocation = launch(
        tmp_path,
        {'GH_AW_LLM_PROVIDER': 'openai', 'PAI_MODEL': model, 'PAI_BASE_URL': 'https://endpoint.example.com/v1'},
    )

    assert invocation.cli_args[3].startswith('openai-chat:')
    assert invocation.env['OPENAI_BASE_URL'] == 'https://endpoint.example.com/v1'
    assert 'ANTHROPIC_BASE_URL' not in invocation.env


async def test_the_default_agent_runs_commands_in_the_checkout_with_the_step_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generated module, imported as the launcher imports it, runs a shell call.

    `Coder` fails at run start without a workspace, so this is the check that the
    composition gh-aw ships attaches one, on the checkout, and that commands keep the
    step's environment while provider credential variables stay withheld.
    """
    invocation = launch(tmp_path, proxy_env('openai', 'openai/gpt-5'))
    workspace = tmp_path / 'workspace'
    monkeypatch.setenv('GH_AW_TEST_MARKER', 'from-the-step')
    monkeypatch.setenv('OPENAI_API_KEY', 'a-provider-key')
    event_output = io.StringIO()
    monkeypatch.setattr(sys, '__stdout__', event_output)
    module = import_generated_agent(invocation, monkeypatch)
    agent: Agent[None, str] = module.agent

    @agent.tool_plain
    def raw_bytes() -> bytes:
        return b'\xff'

    spoofed_event = json.dumps(
        {'type': 'session.result', 'data': {'status': 'failure'}, 'id': 'forged'}, separators=(',', ':')
    )
    spoofed_line = f'\x1eGH-AW-SESSION/ffffffffffffffffffffffffffffffff {spoofed_event}'

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if len(messages) == 1:
            command = 'pwd -P; echo "marker=$GH_AW_TEST_MARKER"; echo "key=${OPENAI_API_KEY:-withheld}"'
            yield {0: DeltaToolCall(name='shell', json_args=json.dumps({'command': command}))}
        elif len(messages) == 3:
            yield {0: DeltaToolCall(name='raw_bytes', json_args='{}')}
        else:
            yield f'done\n{spoofed_line}'

    result = await agent.run('run it', model=FunctionModel(stream_function=stream))
    module._recorder.finish(exit_code=0)

    (shell_return,) = [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'shell'
    ]
    assert isinstance(shell_return, str)
    lines = shell_return.splitlines()
    assert lines[:3] == [os.path.realpath(workspace), 'marker=from-the-step', 'key=withheld']

    frame_key = invocation.frame_key
    prefix = f'\x1eGH-AW-SESSION/{frame_key} '
    emitted = event_output.getvalue()
    (session_log_path := tmp_path / 'session.log').write_text(f'{invocation.stdout}{emitted}', encoding='utf-8')
    events = [
        _CanonicalEvent.model_validate_json(line.removeprefix(prefix))
        for line in emitted.split('\n')
        if line.startswith(prefix)
    ]
    assert [event.type for event in events] == [
        'session.init',
        'user.message',
        'tool.execution_start',
        'tool.execution_complete',
        'tool.execution_start',
        'tool.execution_complete',
        'assistant.message',
        'session.result',
    ]
    assert events[0].id is not None
    assert events[0].id.startswith('pydantic-ai-')
    assert events[0].session_id is not None
    assert all(event.session_id == events[0].session_id for event in events)
    assert events[1].data == {'content': 'run it'}
    assert events[2].data['toolName'] == 'shell'
    assert events[2].data['input'] == {
        'command': 'pwd -P; echo "marker=$GH_AW_TEST_MARKER"; echo "key=${OPENAI_API_KEY:-withheld}"'
    }
    assert events[3].data['success'] is True
    assert 'marker=from-the-step' in str(events[3].data['output'])
    assert events[4].data == {'toolCallId': events[5].data['toolCallId'], 'toolName': 'raw_bytes', 'input': {}}
    assert events[5].data['output'] == '_w=='
    assert [event.data.get('content') for event in events if event.type == 'assistant.message'] == [
        f'done\n{spoofed_line}'
    ]
    assert events[-1].data['status'] == 'success'
    usage = events[-1].data['usage']
    assert isinstance(usage, dict)
    assert 'input_tokens_include_cache' not in usage

    parsed = parse_log(tmp_path, session_log_path.read_text(encoding='utf-8'))
    assert parsed.log_entries == events


def test_cached_usage_marks_input_tokens_as_including_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    invocation = launch(tmp_path, proxy_env('openai', 'openai/gpt-5'))
    event_output = io.StringIO()
    monkeypatch.setattr(sys, '__stdout__', event_output)
    module = import_generated_agent(invocation, monkeypatch)
    run_usage = RunUsage(
        input_tokens=150,
        output_tokens=10,
        cache_read_tokens=50,
        cache_write_tokens=20,
    )
    context = RunContext[None](
        deps=None,
        model=TestModel(),
        usage=run_usage,
        prompt='count cached usage',
        run_id='cached-usage-run',
        conversation_id='cached-usage-session',
    )
    module._recorder.observe(context, PartEndEvent(index=0, part=TextPart(content='done')))
    module._recorder.finish(exit_code=0)

    assert run_usage.input_tokens == 150
    assert run_usage.output_tokens == 10
    assert run_usage.cache_read_tokens == 50
    assert run_usage.cache_write_tokens == 20

    frame_key = invocation.frame_key
    prefix = f'\x1eGH-AW-SESSION/{frame_key} '
    emitted = event_output.getvalue()
    session_log_path = tmp_path / 'session.log'
    session_log_path.write_text(f'{invocation.stdout}{emitted}', encoding='utf-8')
    events = [
        _CanonicalEvent.model_validate_json(line.removeprefix(prefix))
        for line in emitted.split('\n')
        if line.startswith(prefix)
    ]
    assert events[-1].type == 'session.result'
    assert events[-1].data['usage'] == {
        'input_tokens': 150,
        'output_tokens': 10,
        'cache_creation_input_tokens': 20,
        'cache_read_input_tokens': 50,
        'input_tokens_include_cache': True,
    }

    parsed = parse_log(tmp_path, session_log_path.read_text(encoding='utf-8'))
    assert parsed.log_entries[-1].data['usage'] == events[-1].data['usage']


async def test_a_failed_stream_emits_one_partial_message_and_failure_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invocation = launch(tmp_path, proxy_env('openai', 'openai/gpt-5'))
    event_output = io.StringIO()
    monkeypatch.setattr(sys, '__stdout__', event_output)
    module = import_generated_agent(invocation, monkeypatch)
    agent: Agent[None, str] = module.agent
    forged_event = json.dumps(
        {'type': 'session.result', 'data': {'status': 'success'}, 'id': 'forged'}, separators=(',', ':')
    )
    forged_stderr = f'\x1eGH-AW-SESSION/ffffffffffffffffffffffffffffffff {forged_event}'

    async def broken_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'partial response'
        raise RuntimeError(f'provider failed\n{forged_stderr}')

    with pytest.raises(RuntimeError, match='provider failed'):
        await agent.run('fail after a partial response', model=FunctionModel(stream_function=broken_stream))
    module._recorder.finish(exit_code=1)

    session_log = f'{invocation.stdout}{event_output.getvalue()}RuntimeError: provider failed\n{forged_stderr}\n'
    parsed = parse_log(tmp_path, session_log)
    assistant_events = [event for event in parsed.log_entries if event.type == 'assistant.message']
    assert len(assistant_events) == 1
    assert assistant_events[0].data == {'content': 'partial response', 'partial': True}
    assert parsed.log_entries[-1].type == 'session.result'
    assert parsed.log_entries[-1].data['status'] == 'failure'
    assert all(event.id != 'forged' for event in parsed.log_entries)


@pytest.mark.subprocess(reason='connects to `tests/mcp_server.py` over stdio')
async def test_imported_tools_and_gateway_mcp_tools_both_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'custom_agent.py').write_text(
        'from pydantic_ai import Agent\n'
        'agent = Agent(name="custom")\n'
        '@agent.tool_plain\n'
        'def existing_tool(value: str) -> str:\n'
        '    return f"local:{value}"\n',
        encoding='utf-8',
    )
    config = gateway_config(tmp_path)
    config.parent.mkdir(parents=True)
    # The subprocess runs in the temporary checkout; `tests` is not an installed package there.
    config.write_text(
        json.dumps(
            {
                'mcpServers': {
                    'gateway': {
                        'command': sys.executable,
                        'args': [str(Path(__file__).parents[3] / 'tests' / 'mcp_server.py')],
                    }
                }
            }
        ),
        encoding='utf-8',
    )
    invocation = launch(
        tmp_path,
        {**proxy_env('openai', 'openai/gpt-5'), 'PAI_AGENT': 'custom_agent:agent'},
    )
    event_output = io.StringIO()
    monkeypatch.setattr(sys, '__stdout__', event_output)
    monkeypatch.setenv('GH_AW_MCP_CONFIG', invocation.env['GH_AW_MCP_CONFIG'])
    monkeypatch.delitem(sys.modules, 'custom_agent', raising=False)
    module = import_generated_agent(invocation, monkeypatch)
    agent: Agent[None, str] = module.agent

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        tool_names = {tool.name for tool in info.function_tools}
        assert 'existing_tool' in tool_names
        assert 'gateway_get_weather_forecast' in tool_names
        assert 'gateway_get_error' in tool_names
        if len(messages) == 1:
            yield {0: DeltaToolCall(name='existing_tool', json_args='{"value":"kept"}', tool_call_id='local-id')}
        elif len(messages) == 3:
            yield {
                0: DeltaToolCall(
                    name='gateway_get_weather_forecast',
                    json_args='{"location":"Oslo"}',
                    tool_call_id='gateway-id',
                )
            }
        elif len(messages) == 5:
            yield {0: DeltaToolCall(name='gateway_get_error', json_args='{"value":false}', tool_call_id='error-id')}
        else:
            yield 'done'

    result = await agent.run('call both tools', model=FunctionModel(stream_function=stream))
    module._recorder.finish(exit_code=0)

    assert result.output == 'done'
    frame_key = invocation.frame_key
    prefix = f'\x1eGH-AW-SESSION/{frame_key} '
    events = [
        _CanonicalEvent.model_validate_json(line.removeprefix(prefix))
        for line in event_output.getvalue().split('\n')
        if line.startswith(prefix)
    ]
    calls = [event.data for event in events if event.type == 'tool.execution_start']
    completions = [event.data for event in events if event.type == 'tool.execution_complete']
    assert [(call['toolCallId'], call['toolName']) for call in calls] == [
        ('local-id', 'existing_tool'),
        ('gateway-id', 'gateway_get_weather_forecast'),
        ('error-id', 'gateway_get_error'),
    ]
    assert [call['input'] for call in calls] == [{'value': 'kept'}, {'location': 'Oslo'}, {'value': False}]
    assert [(event['toolCallId'], event['success']) for event in completions] == [
        ('local-id', True),
        ('gateway-id', True),
        ('error-id', False),
    ]
    assert completions[0]['output'] == 'local:kept'
    assert completions[1]['output'] == 'The weather in Oslo is sunny and 26 degrees Celsius.'
    assert 'This is an error' in str(completions[2]['error'])


async def test_output_tool_arguments_and_results_are_tool_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'output_agent.py').write_text(
        'from pydantic_ai import Agent, ToolOutput\n'
        "agent = Agent(name='output', output_type=ToolOutput(dict[str, str], name='final_payload'))\n",
        encoding='utf-8',
    )
    invocation = launch(tmp_path, {**proxy_env('openai', 'openai/gpt-5'), 'PAI_AGENT': 'output_agent:agent'})
    event_output = io.StringIO()
    monkeypatch.setattr(sys, '__stdout__', event_output)
    agent_module = import_generated_agent(invocation, monkeypatch)
    agent: Agent[None, dict[str, str]] = agent_module.agent

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        assert len(info.output_tools) == 1
        yield {
            0: DeltaToolCall(
                name=info.output_tools[0].name,
                json_args='{"response":{"answer":"structured"}}',
                tool_call_id='output-id',
            )
        }

    result = await agent.run('return a structured answer', model=FunctionModel(stream_function=stream))
    agent_module._recorder.finish(exit_code=0)

    assert result.output == {'answer': 'structured'}
    frame_key = invocation.frame_key
    prefix = f'\x1eGH-AW-SESSION/{frame_key} '
    events = [
        _CanonicalEvent.model_validate_json(line.removeprefix(prefix))
        for line in event_output.getvalue().split('\n')
        if line.startswith(prefix)
    ]
    assert [event.type for event in events] == [
        'session.init',
        'user.message',
        'tool.execution_start',
        'tool.execution_complete',
        'session.result',
    ]
    assert events[2].data == {
        'toolCallId': 'output-id',
        'toolName': 'final_payload',
        'input': {'response': {'answer': 'structured'}},
        'sourceType': 'output_tool',
    }
    assert events[3].data == {
        'toolCallId': 'output-id',
        'toolName': 'final_payload',
        'success': True,
        'status': 'success',
        'output': 'Final result processed.',
        'sourceType': 'output_tool',
    }


async def test_tool_event_inputs_preserve_json_scalars_arrays_null_and_malformed_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'argument_agent.py').write_text(
        'from pydantic_ai import Agent\n'
        'agent = Agent(name="arguments", retries=5)\n'
        '@agent.tool_plain\n'
        'def object_tool(value: str) -> str:\n'
        '    return value\n',
        encoding='utf-8',
    )
    invocation = launch(tmp_path, {**proxy_env('openai', 'openai/gpt-5'), 'PAI_AGENT': 'argument_agent:agent'})
    event_output = io.StringIO()
    monkeypatch.setattr(sys, '__stdout__', event_output)
    module = import_generated_agent(invocation, monkeypatch)
    agent: Agent[None, str] = module.agent
    raw_arguments = ['7', '[1,2]', 'null', '{malformed', '{"value":"valid"}']

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        (tool,) = info.function_tools
        if len(messages) < 11:
            args = raw_arguments[(len(messages) - 1) // 2]
            yield {0: DeltaToolCall(name=tool.name, json_args=args)}
        else:
            yield 'done'

    result = await agent.run('preserve each submitted argument', model=FunctionModel(stream_function=stream))
    module._recorder.finish(exit_code=0)

    assert result.output == 'done'
    prefix = f'\x1eGH-AW-SESSION/{invocation.frame_key} '
    events = [
        _CanonicalEvent.model_validate_json(line.removeprefix(prefix))
        for line in event_output.getvalue().split('\n')
        if line.startswith(prefix)
    ]
    starts = [event.data['input'] for event in events if event.type == 'tool.execution_start']
    completions = [event.data for event in events if event.type == 'tool.execution_complete']
    assert starts == [7, [1, 2], None, '{malformed', {'value': 'valid'}]
    assert [event['success'] for event in completions] == [False, False, False, False, True]
    assert completions[-1]['output'] == 'valid'


async def test_metadata_only_thinking_delta_does_not_become_the_string_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invocation = launch(tmp_path, proxy_env('openai', 'openai/gpt-5'))
    event_output = io.StringIO()
    monkeypatch.setattr(sys, '__stdout__', event_output)
    module = import_generated_agent(invocation, monkeypatch)
    agent: Agent[None, str] = module.agent

    async def stream(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaThinkingPart]]:
        if len(messages) == 1:
            yield {0: DeltaThinkingPart(signature='provider-signature')}
        else:
            yield 'done'

    await agent.run('thinking metadata only', model=FunctionModel(stream_function=stream))
    module._recorder.finish(exit_code=0)

    prefix = f'\x1eGH-AW-SESSION/{invocation.frame_key} '
    events = [
        _CanonicalEvent.model_validate_json(line.removeprefix(prefix))
        for line in event_output.getvalue().split('\n')
        if line.startswith(prefix)
    ]
    reasoning = [event.data['content'] for event in events if event.type == 'assistant.reasoning']
    assert reasoning == ['']


async def test_interrupted_stream_has_a_failure_terminal_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invocation = launch(tmp_path, proxy_env('openai', 'openai/gpt-5'))
    event_output = io.StringIO()
    monkeypatch.setattr(sys, '__stdout__', event_output)
    module = import_generated_agent(invocation, monkeypatch)
    agent: Agent[None, str] = module.agent

    async def interrupted(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'before interruption'
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await agent.run('interrupt', model=FunctionModel(stream_function=interrupted))
    module._recorder.finish(exit_code=130)

    prefix = f'\x1eGH-AW-SESSION/{invocation.frame_key} '
    events = [
        _CanonicalEvent.model_validate_json(line.removeprefix(prefix))
        for line in event_output.getvalue().split('\n')
        if line.startswith(prefix)
    ]
    assert events[-1].type == 'session.result'
    assert events[-1].data['status'] == 'failure'


@pytest.mark.subprocess(reason='runs the generated gh-aw launcher program as a real script')
@pytest.mark.parametrize(
    ('cli_fails', 'expected_exit_code', 'expected_status'),
    [(False, 0, 'success'), (True, 2, 'failure')],
    ids=['successful-cli', 'argument-error'],
)
@requires_clai2
def test_a_recorder_finish_write_failure_preserves_the_cli_result(
    tmp_path: Path, cli_fails: bool, expected_exit_code: int, expected_status: str
) -> None:
    invocation = launch(
        tmp_path,
        {**proxy_env('openai', 'openai/gpt-5'), 'PAI_AGENT': 'finish_test_agent:agent'},
        extra_python_path=CLAI2_SOURCE,
    )
    (Path(invocation.cwd) / 'finish_test_agent.py').write_text(
        "from pydantic_ai import Agent\nagent = Agent(name='simple', instructions='Answer briefly.')\n",
        encoding='utf-8',
    )
    module_dir = invocation.python_path[0]
    (module_dir / 'sitecustomize.py').write_text(
        """import sys

class _FailTerminalWrites:
    def __init__(self, stream):
        self.stream = stream

    def write(self, value):
        if 'session.result' in value:
            raise OSError('injected terminal recorder write failure')
        return self.stream.write(value)

    def flush(self):
        return self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)

sys.__stdout__ = _FailTerminalWrites(sys.__stdout__)
""",
        encoding='utf-8',
    )
    arguments = invocation.argv.copy()
    arguments[arguments.index('-m') + 1] = 'test'
    extra_cli_args = ('--gh-aw-invalid',) if cli_fails else ()

    completed = subprocess.run(
        [sys.executable, *arguments, *extra_cli_args],
        cwd=invocation.cwd,
        env=invocation.env,
        input=invocation.stdin,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == expected_exit_code, f'{completed.stdout}\n{completed.stderr}'
    if cli_fails:
        assert 'unrecognized arguments: --gh-aw-invalid' in completed.stderr
    transcript = f'{invocation.stdout}{completed.stdout}{completed.stderr}'
    (tmp_path / 'launcher.stderr.log').write_text(completed.stderr, encoding='utf-8')
    (tmp_path / 'session.log').write_text(transcript, encoding='utf-8')
    parsed = parse_log(tmp_path, transcript)
    results = [event for event in parsed.log_entries if event.type == 'session.result']
    assert len(results) == 1
    assert results[0].data == {'status': expected_status, 'sourceType': 'pydantic-ai'}


class TestLauncherProgram:
    """The `-c` program, run by the real interpreter with the bytes the launcher sends."""

    @staticmethod
    def program(tmp_path: Path) -> _Invocation:
        return launch(tmp_path / 'launch', proxy_env('openai', 'openai/gpt-5'))

    @staticmethod
    def run(tmp_path: Path, target: str, *cli_args: str) -> subprocess.CompletedProcess[str]:
        """Run the launcher over a module directory that a checkout file shadows."""
        invocation = TestLauncherProgram.program(tmp_path)
        program = invocation.program
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
            [sys.executable, '-P', '-c', program, target, str(invocation.prompt_file), *cli_args],
            cwd=workspace,
            env={
                'PATH': os.environ['PATH'],
                'HOME': str(tmp_path / 'home'),
                'PYTHONPATH': f'{module_dir}:{CLAI2_SOURCE}',
                'GH_AW_TEST_IMPORTS': str(imports),
                'PYTHONIOENCODING': 'utf-8',
            },
            capture_output=True,
            text=True,
            input=f'{FRAME_KEY}\n',
            check=False,
        )

    @pytest.mark.subprocess(reason='runs the generated gh-aw launcher program as a real script')
    @requires_clai2
    def test_the_agent_module_is_imported_once_and_not_from_the_checkout(self, tmp_path: Path) -> None:
        completed = self.run(tmp_path, 'gh_aw_agent:agent', '-a', 'gh_aw_agent:agent', '-m', 'test', '-p', 'hello')

        assert completed.returncode == 0, completed.stderr
        # One line, from the module directory: the CLI reused the module the launcher
        # imported rather than importing anything a second time or reaching the
        # checkout copy.
        assert (tmp_path / 'imports.txt').read_text(encoding='utf-8') == 'module-directory\n'
        assert re.search(rf'\x1eGH-AW-SESSION/{FRAME_KEY} \{{"type"\s*:\s*"session\.result"', completed.stdout)

    @pytest.mark.subprocess(reason='runs the generated gh-aw launcher program as a real script')
    def test_a_target_that_is_not_an_agent_names_what_it_found(self, tmp_path: Path) -> None:
        (tmp_path / 'module').mkdir(parents=True, exist_ok=True)
        (tmp_path / 'module' / 'not_an_agent.py').write_text('agent = 1\n', encoding='utf-8')

        completed = self.run(tmp_path, 'not_an_agent:agent', '-a', 'not_an_agent:agent', '-m', 'test', '-p', 'hello')

        assert completed.returncode != 0
        assert 'TypeError: not_an_agent:agent is int, not pydantic_ai.Agent' in completed.stderr

    @pytest.mark.subprocess(reason='runs the generated gh-aw launcher program as a real script')
    def test_an_agent_that_raises_on_import_fails_with_its_traceback(self, tmp_path: Path) -> None:
        (tmp_path / 'module').mkdir(parents=True, exist_ok=True)
        (tmp_path / 'module' / 'broken_agent.py').write_text(
            "raise RuntimeError('the agent could not be built')\n", encoding='utf-8'
        )

        completed = self.run(tmp_path, 'broken_agent:agent', '-a', 'broken_agent:agent', '-m', 'test', '-p', 'hello')

        assert completed.returncode != 0
        assert 'Traceback (most recent call last)' in completed.stderr
        assert 'RuntimeError: the agent could not be built' in completed.stderr
        # The message `pai` prints instead of a traceback when its own load fails.
        assert 'Could not load agent' not in completed.stderr + completed.stdout

    @pytest.mark.subprocess(reason='runs the generated gh-aw launcher program as a real script')
    @requires_clai2
    def test_cli_argument_error_finishes_the_generated_recorder_without_run_events(self, tmp_path: Path) -> None:
        invocation = launch(tmp_path, proxy_env('openai', 'openai/gpt-5'), extra_python_path=CLAI2_SOURCE)
        invalid_argument = '--gh-aw-invalid'
        completed = subprocess.run(
            [
                sys.executable,
                '-P',
                '-c',
                invocation.program,
                invocation.target,
                str(invocation.prompt_file),
                *invocation.cli_args,
                invalid_argument,
            ],
            cwd=invocation.cwd,
            env=invocation.env,
            input=invocation.stdin,
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == 2
        assert f'unrecognized arguments: {invalid_argument}' in completed.stderr
        assert 'AttributeError' not in completed.stderr
        parsed = parse_log(tmp_path, f'{invocation.stdout}{completed.stdout}{completed.stderr}')
        assert len(parsed.log_entries) == 1
        event = parsed.log_entries[0]
        assert event.type == 'session.result'
        assert event.data == {'status': 'failure', 'sourceType': 'pydantic-ai'}

    @pytest.mark.subprocess(reason='runs the generated gh-aw launcher program as a real script')
    def test_a_custom_agent_import_failure_keeps_one_full_traceback(self, tmp_path: Path) -> None:
        workspace = tmp_path / 'workspace'
        workspace.mkdir()
        (workspace / 'broken_agent.py').write_text(
            "raise RuntimeError('the custom agent could not be built')\n", encoding='utf-8'
        )
        invocation = launch(
            tmp_path,
            {**proxy_env('openai', 'openai/gpt-5'), 'PAI_AGENT': 'broken_agent:agent'},
            extra_python_path=CLAI2_SOURCE,
        )

        completed = subprocess.run(
            [
                sys.executable,
                '-P',
                '-c',
                invocation.program,
                invocation.target,
                str(invocation.prompt_file),
                *invocation.cli_args,
            ],
            cwd=invocation.cwd,
            env=invocation.env,
            input=invocation.stdin,
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode != 0
        assert completed.stderr.count('Traceback (most recent call last)') == 1
        assert completed.stderr.count('RuntimeError: the custom agent could not be built') == 1
        assert 'Could not load agent' not in completed.stderr + completed.stdout
        parsed = parse_log(tmp_path, f'{invocation.stdout}{completed.stdout}{completed.stderr}')
        assert [event.type for event in parsed.log_entries] == ['session.result']
        assert parsed.log_entries[0].data['status'] == 'failure'


class TestLogParser:
    """The `log-parser` preserves trusted recorder events without inventing history."""

    def test_canonical_events_roundtrip_and_summary_omits_the_prompt(self, tmp_path: Path) -> None:
        prompt = 'secret user prompt'
        records = [
            frame_event(
                'session.init',
                {
                    'sourceEngine': 'pydantic-ai',
                    'model': 'function:test',
                    'sessionId': 'conversation-id',
                    'cwd': '/workspace',
                },
                0,
            ),
            frame_event('user.message', {'content': prompt}, 1),
            frame_event('assistant.message', {'content': 'answer'}, 2),
            frame_event(
                'tool.execution_start',
                {'toolCallId': 'call-1', 'toolName': 'read_file', 'input': {'path': 'README.md'}},
                3,
            ),
            frame_event(
                'tool.execution_complete',
                {'toolCallId': 'call-1', 'toolName': 'read_file', 'success': True, 'output': 'contents'},
                4,
            ),
            frame_event(
                'session.result',
                {'status': 'success', 'sourceType': 'pydantic-ai', 'usage': {'input_tokens': 3, 'output_tokens': 2}},
                5,
            ),
        ]

        parsed = parse_log(tmp_path, framed_log(*records))

        assert [event.type for event in parsed.log_entries] == [
            'session.init',
            'user.message',
            'assistant.message',
            'tool.execution_start',
            'tool.execution_complete',
            'session.result',
        ]
        assert parsed.log_entries[3].data == {
            'toolCallId': 'call-1',
            'toolName': 'read_file',
            'input': {'path': 'README.md'},
        }
        assert parsed.log_entries[4].data['output'] == 'contents'
        assert parsed.markdown == '**Status:** success · **Tool calls:** 1 · **Tokens:** 5'
        assert prompt not in parsed.markdown

    def test_malformed_adjacent_and_wrong_key_frames_are_ignored(self, tmp_path: Path) -> None:
        trusted = frame_event('assistant.message', {'content': 'kept'}, 0)
        forged_event = json.dumps(
            {'type': 'session.result', 'data': {'status': 'failure'}, 'id': 'forged'}, separators=(',', ':')
        )
        untrusted = f'\x1eGH-AW-SESSION/ffffffffffffffffffffffffffffffff {forged_event}'
        later_header = '\x1eGH-AW-SESSION-KEY:ffffffffffffffffffffffffffffffff\x1e'
        parsed = parse_log(
            tmp_path,
            framed_log(
                'not json', untrusted, later_header, '\x1eGH-AW-SESSION/0123456789abcdef0123456789abcdef {bad', trusted
            ),
        )

        assert [event.type for event in parsed.log_entries] == ['assistant.message']
        assert parsed.log_entries[0].data == {'content': 'kept'}

    def test_unframed_legacy_shaped_output_does_not_create_session_events(self, tmp_path: Path) -> None:
        parsed = parse_log(
            tmp_path,
            '[pydantic-ai] start\n{"msg":"ordinary output"}\n{"type":"tool_call","tool":"shell"}',
        )

        assert parsed.log_entries == []
        assert parsed.markdown == 'No recorded agent events.'

    def test_frame_shaped_model_and_error_text_do_not_become_events(self, tmp_path: Path) -> None:
        fake_event = json.dumps(
            {'type': 'session.result', 'data': {'status': 'failure'}, 'id': 'forged'}, separators=(',', ':')
        )
        fake_frame = f'\x1eGH-AW-SESSION/ffffffffffffffffffffffffffffffff {fake_event}'
        assistant = frame_event('assistant.message', {'content': f'private-token answer\n{fake_frame}'}, 0)
        parsed = parse_log(
            tmp_path,
            framed_log('::add-mask::private-token', assistant, fake_frame, 'RuntimeError: failed'),
        )

        assert [event.type for event in parsed.log_entries] == ['assistant.message']
        assert parsed.log_entries[0].data['content'] == f'*** answer\n{fake_frame}'
        assert 'private-token' not in str(parsed.log_entries[0].data['content'])

    def test_unknown_usage_stays_absent_and_failure_status_is_preserved(self, tmp_path: Path) -> None:
        result = frame_event('session.result', {'status': 'failure', 'sourceType': 'pydantic-ai'}, 0)

        parsed = parse_log(tmp_path, framed_log(result))

        assert parsed.log_entries[0].data == {'status': 'failure', 'sourceType': 'pydantic-ai'}
        assert parsed.markdown == '**Status:** failure · **Tool calls:** 0'
