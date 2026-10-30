import ctypes
import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pydantic_ai.models
from pydantic_ai import Agent, ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

# This module is a deterministic acceptance fixture, loaded through PAI_AGENT.
pydantic_ai.models.ALLOW_MODEL_REQUESTS = False
agent = Agent(name='native-recorder-acceptance')
_request_count = 0
_probe_failed = False


@agent.tool_plain
def hostile_probe() -> str:
    """Inspect a same-UID descendant's access to the protected recorder."""
    global _probe_failed
    import gh_aw_recording

    parent_dumpable = ctypes.CDLL(None).prctl(3, 0, 0, 0, 0)
    sentinel = ctypes.create_string_buffer(b'native-recorder-controlled-memory-sentinel')
    capture_fd = gh_aw_recording.output.fileno()
    capture_target = os.readlink(f'/proc/self/fd/{capture_fd}')
    workspace = Path(os.environ['GITHUB_WORKSPACE'])
    report_path = workspace / f'hostile-probe-{os.getpid()}-{os.environ.get("FIXTURE_FAILURE", "0")}.json'
    child = subprocess.run(
        [
            sys.executable,
            '-P',
            str(Path(__file__).with_name('hostile_child.py')),
            str(os.getpid()),
            str(os.getuid()),
            str(capture_fd),
            capture_target,
            str(ctypes.addressof(sentinel)),
            str(ctypes.sizeof(sentinel)),
            str(report_path),
            str(parent_dumpable),
        ],
        close_fds=False,
        check=False,
    )
    report = json.loads(report_path.read_text(encoding='utf-8'))
    _probe_failed = child.returncode != 0 or bool(report['violations'])
    report['child_exit_code'] = child.returncode
    report['parent_dumpable_before_child'] = parent_dumpable
    report['python_executable'] = sys.executable
    report['python_version'] = list(sys.version_info[:2])
    report['parent_capture_inheritable'] = os.get_inheritable(capture_fd)
    if report['parent_capture_inheritable']:
        _probe_failed = True
        report['violations'].append('recorder capture descriptor is inheritable')
    report['parent_dumpable'] = ctypes.CDLL(None).prctl(3, 0, 0, 0, 0)
    if parent_dumpable != 0 or report['parent_dumpable'] != 0:
        _probe_failed = True
        report['violations'].append('recorder parent remains dumpable')
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return json.dumps(report, sort_keys=True)


async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    """Emit one tool call and a deterministic complete, failed, or interrupted response."""
    global _request_count
    _request_count += 1
    if _request_count == 1:
        yield {0: DeltaToolCall(name='hostile_probe', json_args='{}')}
    else:
        yield 'acceptance-secret-unique-123'
        if os.environ.get('FIXTURE_INTERRUPT') == '1':
            os._exit(137)
        if _probe_failed:
            raise RuntimeError('native-fixture hostile subprocess checks failed; inspect the signed probe report')
        if os.environ.get('FIXTURE_FAILURE') == '1':
            raise RuntimeError('intentional-fixture-failure')


# Keep this test-only context alive while the real CLI supplies its workflow model.
_override = agent.override(model=FunctionModel(stream_function=stream))
_override.__enter__()
