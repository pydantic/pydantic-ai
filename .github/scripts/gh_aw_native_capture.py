"""Run the model-free recorder acceptance on a Linux Docker host.

This is a standalone CI boundary test: process, /proc, Docker, and host-native AWF
capture behavior cannot be exercised by an in-process pytest substitute.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

GHAW_VERSION = '1.0.1'
AWF_VERSION = '0.29.0'
GHAW_SOURCE = '0627fdbe4beb5f3b71aa569f8d97150b87f2111a'
GHAW_HASHES = {
    'x86_64': ('linux-amd64', '746d0b41d5fb87e2c010b8ac23a464436bb745ff89fbd19a3456431942c37652'),
    'aarch64': ('linux-arm64', '84a3fb897ec6f6ec698bd160cf8ef8700fd147278e239ad626bd0ec32c3baed2'),
}
AWF_HASHES = {
    'x86_64': ('awf-linux-x64', 'e2ac3c4f95626bafb1c9277283a105538647f41434e7acf2c7bc51d030ecdb41'),
    'aarch64': ('awf-linux-arm64', '54bab35325196b3bc76e645a6b1ca85625f72faa49673fc895ebae90ee064964'),
}
# These are the authenticated v0.29.0 image pins emitted by gh-aw v1.0.1.
IMAGE_TAG = (
    '0.29.0,squid=sha256:6e0cbf7f980beb78b5461563a8164f6439ca959bc93733d5a6b52972e523ed50,'
    'agent=sha256:b14bd399376180b5649a9324be3b50fbfc82bfe8f0c80045840416b5a4e65b96'
)
FIXTURE_SECRET = 'acceptance-secret-unique-123'


def download(url: str, destination: Path, expected: str) -> None:
    """Fetch one pinned public artifact and verify its content digest."""
    if destination.exists():
        data = destination.read_bytes()
    else:
        # Respect the runner's configured proxy and network policy.
        with urllib.request.urlopen(url, timeout=45) as response:
            data = response.read(128 * 1024 * 1024)
        destination.parent.mkdir(parents=True, exist_ok=True)
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise RuntimeError(f'SHA-256 mismatch for {destination.name}: {actual}')
    destination.write_bytes(data)


def execute(
    command: list[str],
    *,
    env: dict[str, str],
    log: Path,
    cwd: Path | None = None,
    timeout: int = 180,
    expected: int | None = 0,
) -> subprocess.CompletedProcess[str]:
    """Run a bounded command, retain its output, and terminate its process group on failure."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as output:
        process = subprocess.Popen(
            command, env=env, cwd=cwd, stdout=output, stderr=subprocess.STDOUT, text=True, start_new_session=True
        )
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            raise
    if expected is not None and code != expected:
        raise RuntimeError(f'{Path(command[0]).name} exited {code}; see {log}')
    return subprocess.CompletedProcess(command, code, log.read_text(encoding='utf-8'))


def block(definition: str, name: str) -> str:
    """Extract an engine YAML literal block without requiring a YAML runtime dependency."""
    lines = definition.splitlines()
    start = lines.index(f'    {name}: |') + 1
    end = start
    while end < len(lines) and (not lines[end] or lines[end].startswith('      ')):
        end += 1
    return '\n'.join(line[6:] for line in lines[start:end]).rstrip('\n') + '\n'


def heredoc(lock: str, filename: str, expected_script: str = '') -> str:
    """Read one exact engine script packaged by the released compiler."""
    match = re.search(r"^          cat <<'([^']+)' > .*?/" + re.escape(filename) + r'"\n', lock, re.M)
    if match is None:
        raise RuntimeError(f'Released compiler did not package {filename}')
    end = lock.index(f'          {match[1]}\n', match.end())
    script = '\n'.join(line[10:] for line in lock[match.end() : end].splitlines()) + '\n'
    if expected_script.rstrip('\n') not in script:
        raise RuntimeError(f'Released compiler changed or omitted {filename}')
    return script


def clean_docker(project: str, env: dict[str, str], evidence: Path) -> None:
    """Remove only containers and networks carrying this run's Compose project label."""
    for kind, listing, remove in [
        ('containers', ['docker', 'ps', '-aq'], ['docker', 'rm', '-f']),
        ('networks', ['docker', 'network', 'ls', '-q'], ['docker', 'network', 'rm']),
    ]:
        result = execute(
            [*listing, '--filter', f'label=com.docker.compose.project={project}'],
            env=env,
            log=evidence / f'cleanup-{kind}-list.log',
            timeout=30,
        )
        identifiers = result.stdout.split()
        if identifiers:
            execute([*remove, *identifiers], env=env, log=evidence / f'cleanup-{kind}.log', timeout=45)


def clean_shared(shared: Path, marker: str, *, remove_root: bool) -> None:
    """Clear the shared session tree only while this task's exact ownership marker remains."""
    if not shared.exists():
        return
    owner = shared / '.native-capture-owner'
    if not owner.exists() or owner.read_text() != marker:
        raise RuntimeError('Lost ownership of shared session tree; refusing cleanup')
    for child in shared.iterdir():
        if child.name != owner.name:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    if remove_root:
        owner.unlink()
        shared.rmdir()


def run_case(
    work: Path,
    evidence: Path,
    env: dict[str, str],
    node: Path,
    case: str,
    fail: str,
    interrupt: str,
    expected_exit: int,
) -> dict[str, object]:
    """Run one native process-boundary scenario through the released conclusion path."""
    workspace = work / 'workspace'
    fixture = workspace / 'fixture'
    helpers = work / 'runner/gh-aw/actions'
    packages = workspace / 'packages'
    case_evidence = evidence / case
    case_evidence.mkdir(mode=0o700)
    project = f'ghaw-native-{os.getpid()}-{case}'
    (workspace / 'prompt.md').write_text('Execute the controlled acceptance probe.')
    explicit = {
        'GITHUB_WORKSPACE': str(workspace),
        'RUNNER_TEMP': str(work / 'runner'),
        'GH_AW_PROMPT': str(workspace / 'prompt.md'),
        'GH_AW_PROTECTED_SESSION_CAPTURE': '1',
        'PAI_AGENT': 'acceptance_agent:agent',
        'PAI_MODEL': 'openai/gpt-5',
        'PAI_BASE_URL': 'http://127.0.0.1:9',
        'FIXTURE_FAILURE': fail,
        'FIXTURE_INTERRUPT': interrupt,
        'PYTHONPATH': f'{packages}:{fixture}',
        'PYTHONNOUSERSITE': '1',
        'pythonLocation': '/usr',
    }
    command = [
        str(work / 'bin/awf'),
        '--container-workdir',
        str(workspace),
        '--mount',
        f'{workspace}:{workspace}:rw',
        '--mount',
        f'{work / "runner"}:{work / "runner"}:ro',
        '--mount',
        '/tmp/gh-aw:/tmp/gh-aw:rw',
        '--work-dir',
        str(work / f'awf-{case}'),
        '--capture-command-output',
        str(work / f'capture-{case}'),
        '--capture-command-output-max-bytes',
        '4194304',
        '--agent-timeout',
        '3',
        '--memory-limit',
        '4g',
        '--pids-limit',
        '256',
        '--image-tag',
        IMAGE_TAG,
    ]
    # /opt toolcache is a documented implicit read-only host mount. Mount
    # other setup-node roots explicitly rather than assuming their visibility.
    if not str(node).startswith(('/opt/', '/usr/', '/bin/')):
        command += ['--mount', f'{node.parent.parent}:{node.parent.parent}:ro']
    for key, value in explicit.items():
        command += ['-e', f'{key}={value}']
    # exec eliminates the intermediate shell that would retain the capture fd.
    inner = f'exec {shlex.quote(str(node))} {shlex.quote(str(helpers / "pydantic-ai_harness.cjs"))} clai2 --database {shlex.quote(str(workspace / "settings.db"))}'
    command += ['--', '/bin/bash', '-c', inner]
    case_env = {**env, 'COMPOSE_PROJECT_NAME': project}
    (case_evidence / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    try:
        awf = execute(command, cwd=workspace, env=case_env, log=case_evidence / 'awf.log', timeout=240, expected=None)
        metadata_path = work / f'capture-{case}' / 'metadata.json'
        if metadata_path.exists():
            shutil.copyfile(metadata_path, case_evidence / 'metadata.json')
        for probe in workspace.glob('hostile-probe-*.json'):
            shutil.copyfile(probe, case_evidence / probe.name)
        if awf.returncode != expected_exit:
            raise RuntimeError(f'{case}: AWF exited {awf.returncode}, expected app exit {expected_exit}; see awf.log')
        metadata = json.loads(metadata_path.read_text())
        if metadata.get('command', {}).get('exitCode') != expected_exit:
            raise RuntimeError(f'{case}: unexpected native command exit code; inspect metadata.json')
        post_env = {
            **env,
            'GH_AW_AGENT_OUTPUT': str(work / f'capture-{case}'),
            'GH_AW_PROTECTED_SESSION_CAPTURE': '1',
            'GH_AW_SECRET_NAMES': 'CAPTURE_FIXTURE_SECRET',
            'SECRET_CAPTURE_FIXTURE_SECRET': FIXTURE_SECRET,
            'RUNNER_TEMP': str(work / 'runner'),
        }
        execute(
            [
                str(node),
                str(fixture / 'postprocess.cjs'),
                str(helpers),
                case,
                str(case_evidence),
                str(expected_exit),
                (evidence / 'python-version.json').read_text().strip(),
                (Path('/tmp/gh-aw') / '.native-capture-owner').read_text(),
            ],
            env=post_env,
            log=case_evidence / 'postprocess.log',
            timeout=60,
        )
        conclusion = json.loads((case_evidence / 'conclusion-step.json').read_text())
        if conclusion.get('status') != 'passed':
            raise RuntimeError(f'{case}: postprocessing did not finish its acceptance assertions')
        if not (case_evidence / 'actions-log.json').is_file():
            raise RuntimeError(f'{case}: postprocessing exited before its completion receipt')
        result: dict[str, object] = {
            'case': case,
            'awf_exit_code': awf.returncode,
            'native_exit_code': expected_exit,
            'status': 'passed',
        }
    finally:
        clean_docker(project, env, case_evidence)
    return result


def check_assets(assets: Path, script: str, raw_parser: str) -> None:
    """Validate test assets and report exact source hashes without claiming a native pass."""
    for fixture in assets.glob('*.py'):
        compile(fixture.read_text(encoding='utf-8'), str(fixture), 'exec')
    subprocess.run(['node', '--check', str(assets / 'postprocess.cjs')], check=True)
    print(
        json.dumps(
            {
                'status': 'static-check-only',
                'harness_sha256': hashlib.sha256(script.encode()).hexdigest(),
                'parser_sha256': hashlib.sha256(raw_parser.encode()).hexdigest(),
            }
        )
    )


def artifact_upload(lock: str, upload_name: str) -> dict[str, object]:
    """Read one real emitted upload path list and require its parser/redaction publication gate."""
    start = lock.index(f'      - name: {upload_name}\n')
    remainder = lock[start:]
    stop = re.search(r'\n(?:      - name:|  [a-zA-Z_][a-zA-Z_0-9-]*:)', remainder[1:])
    section = remainder[: stop.start() + 1] if stop else remainder
    condition_match = re.search(r'^        if: (.+)$', section, re.M)
    if condition_match is None:
        raise RuntimeError(f'Released upload {upload_name} has no publication condition')
    condition = condition_match.group(1)
    if (
        "steps.parse_agent_logs.outcome == 'success'" not in condition
        or "steps.redact_secrets.outcome == 'success'" not in condition
    ):
        raise RuntimeError('Released artifact publication lacks the protected parser/redaction gate')
    paths = re.findall(r'^            /tmp/gh-aw/(.+)$', section, re.M)
    return {'name': upload_name, 'condition': condition, 'paths': paths}


def main() -> None:
    """Run static checks or all three native protected-capture acceptance cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument(
        '--check-only', action='store_true', help='Validate fixture syntax/extraction; never claim native acceptance'
    )
    arguments = parser.parse_args()
    assets = Path(__file__).with_name('gh_aw_native_capture')
    if not assets.is_dir():
        assets = Path(__file__).with_name('assets')
    repo = arguments.repo.resolve()
    definition = (repo / 'src/pydantic_ai_harness/gh-aw/pydantic.md').read_text(encoding='utf-8')
    script = block(definition, 'harness-script')
    raw_parser = block(definition, 'log-parser')
    check_assets(assets, script, raw_parser)
    if arguments.check_only:
        return
    if sys.platform != 'linux' or os.getuid() == 0:
        raise RuntimeError(
            'Native acceptance requires a non-root Linux runner with Docker; this is a failure, not a skip'
        )
    if platform.machine() not in GHAW_HASHES:
        raise RuntimeError(f'Unsupported native acceptance architecture: {platform.machine()}')
    work = arguments.work_dir.resolve()
    if work == repo or repo in work.parents or work in repo.parents:
        raise RuntimeError('The acceptance work directory must be outside the candidate checkout')
    work.mkdir(mode=0o700, parents=True, exist_ok=False)
    evidence = work / 'evidence'
    evidence.mkdir(mode=0o700)
    home = work / 'home'
    home.mkdir(mode=0o700)
    env = {
        'PATH': os.environ['PATH'],
        'HOME': str(home),
        'LANG': 'C.UTF-8',
        'LC_ALL': 'C.UTF-8',
        'PYTHONNOUSERSITE': '1',
        'PIP_DISABLE_PIP_VERSION_CHECK': '1',
        'PIP_NO_INPUT': '1',
    }
    # Docker daemon routing is configuration, not arbitrary job environment.
    if 'DOCKER_HOST' in os.environ:
        env['DOCKER_HOST'] = os.environ['DOCKER_HOST']
    shared = Path('/tmp/gh-aw')
    if shared.exists():
        raise RuntimeError('/tmp/gh-aw already exists; refusing to replace resources from another task')
    for network in ('awf-net', 'awf-ext'):
        probe = execute(
            ['docker', 'network', 'inspect', network],
            env=env,
            log=evidence / f'preflight-{network}.log',
            expected=None,
            timeout=20,
        )
        if probe.returncode == 0:
            raise RuntimeError(f'{network} already exists; refusing to start across another task network')
    execute(['docker', 'info', '--format', '{{.OSType}}'], env=env, log=evidence / 'docker-info.log', timeout=30)
    node = Path(shutil.which('node') or '').resolve()
    execute(
        [str(node), '-e', 'if(typeof process.execve!=="function")process.exit(1);console.log(process.version)'],
        env=env,
        log=evidence / 'node-version.log',
        timeout=20,
    )
    host_python = Path('/usr/bin/python3')
    execute(
        [str(host_python), '-c', 'import json,sys; print(json.dumps(list(sys.version_info[:2])))'],
        env=env,
        log=evidence / 'python-version.json',
        timeout=20,
    )
    workspace = work / 'workspace'
    workspace.mkdir(mode=0o700)
    fixture = workspace / 'fixture'
    shutil.copytree(assets, fixture)
    packages = workspace / 'packages'
    helpers = work / 'runner' / 'gh-aw' / 'actions'
    helpers.mkdir(parents=True)
    checksums: dict[str, str] = json.loads((assets / 'released_helpers.json').read_text())
    download_jobs: list[tuple[str, Path, str]] = [
        (
            f'https://raw.githubusercontent.com/github/gh-aw/{GHAW_SOURCE}/actions/setup/js/{name}',
            helpers / name,
            digest,
        )
        for name, digest in checksums.items()
    ]
    for name, values in [('gh-aw', GHAW_HASHES), ('awf', AWF_HASHES)]:
        asset, digest = values[platform.machine()]
        project = 'gh-aw' if name == 'gh-aw' else 'gh-aw-firewall'
        version = GHAW_VERSION if name == 'gh-aw' else AWF_VERSION
        download_jobs.append(
            (f'https://github.com/github/{project}/releases/download/v{version}/{asset}', work / 'bin' / name, digest)
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda job: download(*job), download_jobs))
    for name in ('gh-aw', 'awf'):
        (work / 'bin' / name).chmod(0o755)
    execute([str(work / 'bin/gh-aw'), 'version'], env=env, log=evidence / 'gh-aw-version.log', timeout=20)
    execute([str(work / 'bin/awf'), '--version'], env=env, log=evidence / 'awf-version.log', timeout=20)
    compile_root = work / 'compiler'
    workflows = compile_root / '.github/workflows'
    (workflows / 'shared').mkdir(parents=True)
    (workflows / 'shared/pydantic.md').write_text(definition)
    (workflows / 'capture.md').write_text("""---
name: Native recorder acceptance compiler fixture
on: push
permissions:
  contents: read
imports:
  - shared/pydantic.md
engine:
  id: pydantic-ai
  model: openai/fixture-model
steps:
  - name: Declare a dummy configured secret
    run: ':'
    env:
      CAPTURE_FIXTURE_SECRET: ${{ secrets.CAPTURE_FIXTURE_SECRET }}
network:
  allowed:
    - defaults
safe-outputs:
  add-comment:
---

Run the deterministic recorder fixture.
""")
    execute(['git', 'init', '-q', str(compile_root)], env=env, log=evidence / 'git-init.log', timeout=20)
    execute(
        ['git', 'remote', 'add', 'origin', 'https://github.com/pydantic/pydantic-ai.git'],
        cwd=compile_root,
        env=env,
        log=evidence / 'git-remote.log',
        timeout=20,
    )
    execute(
        [str(work / 'bin/gh-aw'), 'compile', 'capture', '--no-check-update', '--no-logo'],
        cwd=compile_root,
        env=env,
        log=evidence / 'compile.log',
        timeout=120,
    )
    lock = (workflows / 'capture.lock.yml').read_text()
    shutil.copyfile(workflows / 'capture.lock.yml', evidence / 'capture.lock.yml')
    compiled_harness = heredoc(lock, 'pydantic-ai_harness.cjs', script)
    compiled_parser = heredoc(lock, 'pydantic-ai_log_parser.cjs', raw_parser)
    required_contract = (
        '--capture-command-output',
        'exec "$GH_AW_NODE_EXEC"',
        'SECRET_CAPTURE_FIXTURE_SECRET:',
        "steps.parse_agent_logs.outcome == 'success'",
    )
    if not all(required in lock for required in required_contract):
        raise RuntimeError('Released compiler omitted a required protected-capture/publication contract')
    transfers = [
        artifact_upload(lock, name) for name in ('Upload agent artifacts', 'Upload agent output fallback artifact')
    ]
    (helpers / 'artifact-transfer.json').write_text(json.dumps(transfers, indent=2) + '\n')
    (helpers / 'pydantic-ai_harness.cjs').write_text(compiled_harness)
    (helpers / 'pydantic-ai_log_parser.cjs').write_text(compiled_parser)
    # Publish unchanged released code for the separate artifact-download consumer.
    shutil.copytree(helpers, evidence / 'released-helpers')
    engine_match = re.search(r'^  version: "([^"]+)"$', definition, re.M)
    slim_match = re.search(r'pydantic-ai-slim\[anthropic,openai,mcp,spec\]>=([0-9.]+)', definition)
    assert engine_match is not None, 'Engine definition must specify its runtime version'
    assert slim_match is not None, 'Engine definition must specify its framework minimum'
    engine_version = engine_match.group(1)
    slim_version = slim_match.group(1)
    execute(
        [
            str(host_python),
            '-m',
            'pip',
            'install',
            '--target',
            str(packages),
            f'pydantic-ai-harness[coder]=={engine_version}',
            f'pydantic-clai2=={engine_version}',
            f'pydantic-ai-slim[anthropic,openai,mcp,spec]>={slim_version}',
            'cryptography>=48.0.1',
        ],
        env=env,
        log=evidence / 'install.log',
        timeout=300,
    )
    install_env = {**env, 'PYTHONPATH': str(packages)}
    execute(
        [str(host_python), '-P', str(fixture / 'prepare_settings.py'), str(workspace)],
        env=install_env,
        log=evidence / 'prepare-settings.log',
        timeout=45,
    )
    marker = f'gh-aw-native-capture-{os.getpid()}-{time.time_ns()}'
    shared.mkdir(mode=0o700)
    (shared / '.native-capture-owner').write_text(marker)
    results: list[dict[str, object]] = []
    try:
        for case, fail, interrupt, expected_exit in [
            ('success', '0', '0', 0),
            ('failure', '1', '0', 1),
            ('incomplete', '0', '1', 137),
        ]:
            results.append(run_case(work, evidence, env, node, case, fail, interrupt, expected_exit))
            clean_shared(shared, marker, remove_root=False)
            for probe in workspace.glob('hostile-probe-*.json'):
                probe.unlink()
        (evidence / 'result.json').write_text(
            json.dumps(
                {
                    'status': 'passed',
                    'cases': results,
                    'gh_aw': GHAW_VERSION,
                    'awf': AWF_VERSION,
                    'source_commit': GHAW_SOURCE,
                },
                indent=2,
            )
            + '\n'
        )
        print(f'Native protected capture acceptance passed: {evidence / "result.json"}')
    finally:
        clean_shared(shared, marker, remove_root=True)


if __name__ == '__main__':
    main()
