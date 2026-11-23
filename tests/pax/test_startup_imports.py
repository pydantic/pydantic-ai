"""Check startup import boundaries in fresh interpreters, without timing thresholds.

Each check lists the third-party distributions its entry point may load; `tests/loaded_distributions.py` says how a
new or stale entry fails.
"""

import logging
import subprocess
import sys
from pathlib import Path

import pytest

from pydantic_ai_pax import warm_imports
from tests.loaded_distributions import (
    SNAPSHOT_PRELUDE,
    Allowlist,
    Loads,
    MayLoad,
    assert_loads_only,
    child_env,
    snapshots,
)

pytestmark = pytest.mark.subprocess(reason='asserts what a fresh `pydantic_ai_pax` process imports at startup')

_PYDANTIC = Loads('Pydantic imports it')
_LOGFIRE = Loads('the Logfire SDK imports it, directly or through its dependencies')
_HTTPCORE2 = Loads("`httpcore2`, HTTPX2's transport, imports it when it is installed")
_TRIO = Loads('Trio imports it')
_KEYRING = Loads('`keyring` imports it')

IMPORT_PYDANTIC_AI: Allowlist = {
    'annotated-types': _PYDANTIC,
    'anyio': Loads('`pydantic_ai` runs its concurrency on AnyIO'),
    'attrs': Loads('Trio, and Rich when it is installed, import it'),
    'backports-tarfile': Loads(
        "on Python 3.11, `urllib3` and HTTPX2 run this distribution's `backports/__init__.py` as they look for "
        '`backports.zstd`'
    ),
    'brotli': Loads('`urllib3` and HTTPX2 import it when it is installed, to decode Brotli responses'),
    'certifi': _LOGFIRE,
    'charset-normalizer': _LOGFIRE,
    'click': Loads('HTTPX imports its command line, which needs Click, when Click is installed'),
    'colorama': Loads("Griffe's command line imports it"),
    'executing': _LOGFIRE,
    'genai-prices': Loads('`pydantic_ai` prices usage with it'),
    'griffecli': Loads('Griffe imports it'),
    'griffelib': Loads('`pydantic_ai` reads tool docstrings with Griffe'),
    'h11': _HTTPCORE2,
    'h2': _HTTPCORE2,
    'hpack': Loads('`h2` imports it'),
    'httpcore2': Loads("`pydantic_ai._http` imports HTTPX2's transport"),
    'httpx': Loads('`pydantic_ai._http` imports HTTPX when it is installed, to accept its clients'),
    'httpx2': Loads("`pydantic_ai`'s HTTP client"),
    'hyperframe': Loads('`h2` imports it'),
    'idna': Loads('HTTPX2 imports it'),
    'logfire': MayLoad(
        'older Logfire releases, such as the one the lowest-versions CI job installs, ship the SDK that '
        '`logfire-api` imports in this distribution; newer ones ship it in `logfire-sdk`'
    ),
    'logfire-api': Loads('`pydantic_ai` and `pydantic_graph` instrument through it'),
    'logfire-sdk': Loads('`logfire-api` imports the Logfire SDK when it is installed'),
    'opentelemetry-api': _LOGFIRE,
    'opentelemetry-exporter-http-transport': _LOGFIRE,
    'opentelemetry-exporter-otlp-common': _LOGFIRE,
    'opentelemetry-exporter-otlp-proto-common': _LOGFIRE,
    'opentelemetry-exporter-otlp-proto-http': _LOGFIRE,
    'opentelemetry-proto': _LOGFIRE,
    'opentelemetry-sdk': _LOGFIRE,
    'opentelemetry-semantic-conventions': _LOGFIRE,
    'outcome': _TRIO,
    'packaging': _LOGFIRE,
    'protobuf': _LOGFIRE,
    'psutil': _LOGFIRE,
    'pydantic': Loads('`pydantic_ai` defines its types with Pydantic'),
    'pydantic-core': _PYDANTIC,
    'pygments': _LOGFIRE,
    'requests': _LOGFIRE,
    'rich': _LOGFIRE,
    'sniffio': Loads('AnyIO imports it'),
    'socksio': _HTTPCORE2,
    'sortedcontainers': _TRIO,
    'trio': _HTTPCORE2,
    'truststore': Loads('`httpcore2` verifies TLS with it'),
    'typing-extensions': _PYDANTIC,
    'typing-inspection': _PYDANTIC,
    'urllib3': _LOGFIRE,
    'zstandard': Loads('HTTPX imports it when it is installed, to decode Zstandard responses'),
}

ARGUMENT_PARSING: Allowlist = {
    'python-dotenv': Loads('`pydantic_ai_pax.__main__` loads `.env` before it parses arguments'),
}

# Building the first Pydantic model loads every installed Pydantic plugin, Logfire's among them, so the splash
# paints before anything loads Pydantic.
SPLASH: Allowlist = {}

PROMPT_READY: Allowlist = {
    **IMPORT_PYDANTIC_AI,
    **ARGUMENT_PARSING,
    'cffi': Loads('`cryptography` imports it'),
    'cryptography': Loads('the credential store encrypts saved credentials with it'),
    'importlib-metadata': Loads('`keyring` imports it on Python 3.11'),
    'jaraco-classes': _KEYRING,
    'jaraco-context': _KEYRING,
    'jaraco-functools': _KEYRING,
    'json-repair': Loads("the `coder` plugin's tool-argument repair imports it"),
    'keyring': Loads('the key menu and the credential store keep keys in the OS keyring'),
    'lazy-importer': Loads('newer `importlib_metadata` releases, which `keyring` imports on Python 3.11, import it'),
    'more-itertools': Loads('`jaraco.functools` imports it'),
    'opentelemetry-instrumentation': Loads('the `observability` plugin instruments HTTPX and HTTPX2 clients'),
    'opentelemetry-instrumentation-httpx': Loads('the `observability` plugin instruments HTTPX and HTTPX2 clients'),
    'opentelemetry-util-http': Loads('the OpenTelemetry HTTPX instrumentation imports it'),
    'pillow': Loads('`pydantic_ai_pax.runtime.image_codec` encodes pasted images with it'),
    'prompt-toolkit': Loads('the prompt'),
    'pyfiglet': Loads('the branding banner'),
    'pyyaml': Loads("the `coder` plugin's skills loader reads skill front matter with it"),
    'termflow-md': Loads("`pydantic_ai_pax.plugins` builds command completions on Termflow's terminal UI"),
    'wcwidth': Loads('`prompt_toolkit` imports it'),
    'wrapt': Loads('the OpenTelemetry HTTPX instrumentation imports it'),
    'zipp': Loads('`importlib_metadata` imports it on Python 3.11'),
}

# Modules that only name provider settings types must not load the SDKs: test modules import them, and loading the
# Anthropic SDK used to add about half a second to every collecting worker.
PROVIDER_SETTINGS_TYPES: Allowlist = {
    **PROMPT_READY,
    'pydantic-monty-client': Loads("speculative mode runs CodeMode's Monty sandbox"),
}


@pytest.mark.parametrize(
    ('args', 'exit_code'),
    [(['--help'], 0), (['--unknown-option'], 2), (['-p', 'hello', '--resume'], 2)],
)
def test_argument_parsing_does_not_load_agent(args: list[str], exit_code: int, tmp_path: Path) -> None:
    script = """
import sys
from pydantic_ai_pax.__main__ import main
try:
    main()
finally:
    snapshot('argument parsing')
    assert 'pydantic_ai' not in sys.modules
    assert 'pydantic_ai_pax.app' not in sys.modules
    assert 'pydantic_ai_pax.cli.headless' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, '-c', SNAPSHOT_PRELUDE + script, *args],
        cwd=tmp_path,
        env=child_env(),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == exit_code, result.stderr
    assert 'AssertionError' not in result.stderr
    loaded = snapshots(result.stdout)['argument parsing']
    assert_loads_only('Argument parsing', loaded, ARGUMENT_PARSING, table_name='ARGUMENT_PARSING')


def test_splash_paints_before_settings_load_pydantic() -> None:
    script = """
import sys
import pydantic_ai_pax.ui.rendering.splash
snapshot('splash')
assert 'pydantic_ai_pax.config' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, '-c', SNAPSHOT_PRELUDE + script],
        env=child_env(),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert_loads_only('The splash', snapshots(result.stdout)['splash'], SPLASH, table_name='SPLASH')


def test_prompt_ready_without_provider_or_model_menu_imports(tmp_path: Path) -> None:
    """Also checks a bare `import pydantic_ai`, which the child runs first, before anything else loads."""
    script = """
import pydantic_ai
snapshot('import pydantic_ai')
import sys
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from pydantic_ai import models
from pydantic_ai_pax import warm_imports
from pydantic_ai_pax.__main__ import main
models.ALLOW_MODEL_REQUESTS = False
requested = []
start = warm_imports.start
def hold_warming(modules=warm_imports.FIRST_USE_MODULES):
    requested.append(modules)
    return start(())
warm_imports.start = hold_warming
async def read(self, *args, **kwargs):
    completions = self.completer.get_completions(Document('/login ', 7), CompleteEvent())
    assert {item.text for item in completions} == {'openai-codex', 'github-copilot'}
    warmed = ('openai', 'anthropic', *warm_imports.FIRST_USE_MODULES)
    for name in (*warmed, 'pydantic_ai_pax.cli.headless'):
        assert name not in sys.modules, name
    # The MCP client stack, FastMCP and the MCP SDK, takes most of a second and is not in `PROMPT_READY`;
    # the `mcp` and `observability` plugins load without it.
    assert 'pydantic_ai_pax.mcp' in sys.modules and 'pydantic_ai_pax.builtin_plugins.logfire' in sys.modules
    assert requested == [warm_imports.FIRST_USE_MODULES]
    snapshot('prompt ready')
    print('PROMPT_READY')
    import pydantic_ai_pax.models.model_options
    import pydantic_ai_pax.runtime.speculative_mode
    snapshot('provider settings types')
    start().join()
    for name in warmed:
        assert name in sys.modules, name
    assert 'pydantic_ai_pax.cli.headless' not in sys.modules
    print('WARMED')
    return '/exit'
PromptSession.prompt_async = read
main()
"""
    result = subprocess.run(
        [sys.executable, '-c', SNAPSHOT_PRELUDE + script, '-m', 'test'],
        cwd=tmp_path,
        env=child_env(CLAI_NO_SPLASH='1'),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert 'PROMPT_READY' in result.stdout
    assert 'WARMED' in result.stdout
    loaded = snapshots(result.stdout)
    assert_loads_only(
        '`import pydantic_ai`', loaded['import pydantic_ai'], IMPORT_PYDANTIC_AI, table_name='IMPORT_PYDANTIC_AI'
    )
    assert_loads_only('The ready prompt', loaded['prompt ready'], PROMPT_READY, table_name='PROMPT_READY')
    assert_loads_only(
        'Importing the provider settings types',
        loaded['provider settings types'],
        PROVIDER_SETTINGS_TYPES,
        table_name='PROVIDER_SETTINGS_TYPES',
    )


def test_warming_failure_is_logged_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger=warm_imports.__name__):
        warm_imports.start(('pydantic_ai_pax._no_such_module', 'pydantic_ai_pax.models.vllm')).join()
    assert 'pydantic_ai_pax.models.vllm' in sys.modules
    assert [record.getMessage() for record in caplog.records] == ['Could not warm pydantic_ai_pax._no_such_module']
