"""Check source-based reload planning in a fresh process, using only its public entry point."""

import importlib
import os
import py_compile
import sys
from graphlib import CycleError
from pathlib import Path

import pytest

import pydantic_clai2
from pydantic_clai2.reloading import reload_clai


def conditional_source(mode: str) -> str:
    guards = {
        'platform': ('import sys', f'sys.platform == {sys.platform!r}', True),
        'platform_alias': ('import sys as runtime', 'runtime.platform != "nonexistent"', True),
        'os_dotted': ('import os.path', f'os.name == {os.name!r}', True),
        'annotation_only': ('from typing import TYPE_CHECKING\nTYPE_CHECKING: bool', 'TYPE_CHECKING', False),
        'os_alias': ('from os import name as system_name', f'system_name == {os.name!r}', True),
        'version': ('import sys', 'sys.version_info >= (3, 10)', True),
        'version_lt': ('import sys', 'sys.version_info < (100,)', True),
        'version_le': ('import sys', 'sys.version_info <= (100,)', True),
        'version_gt': ('import sys', 'sys.version_info > (1,)', True),
        'constant': ('', 'True', True),
        'main': ('', '__name__ == "__main__"', False),
        'module_name': ('', '__name__ == "pydantic_clai2.reload_consumer"', True),
        'package_name': ('', '__package__ == "pydantic_clai2"', True),
        'false': ('', 'False', False),
        'not': ('from typing import TYPE_CHECKING as checking', 'not checking', True),
    }
    imports, guard, selected = guards[mode]
    first, second = ('reload_provider', 'reload_inactive') if selected else ('reload_inactive', 'reload_provider')
    return f'{imports}\nif {guard}:\n    from .{first} import NEW\nelse:\n    from .{second} import NEW\nVALUE = NEW\n'


def main(root: Path, mode: str) -> None:
    sys.path.insert(0, str(root))
    pydantic_clai2.__path__.insert(0, str(root / 'pydantic_clai2'))

    package = root / 'pydantic_clai2'
    provider_path = package / 'reload_provider.py'
    consumer_path = package / 'reload_consumer.py'
    (package / 'reload_leaf.py').write_text("VALUE = 'old'\n")
    provider_path.write_text('from . import reload_leaf\nVALUE = reload_leaf.VALUE\n')
    consumer_path.write_text("VALUE = 'old'\n")
    consumer = importlib.import_module('pydantic_clai2.reload_consumer')
    provider = importlib.import_module('pydantic_clai2.reload_provider')
    original_consumer = vars(consumer).copy()
    original_provider = vars(provider).copy()
    provider_sources = {
        'invalid_guard': 'if 1 < "invalid":\n    pass\n',
        'cycle': 'from .reload_consumer import VALUE\nNEW = VALUE\n',
        'syntax': 'invalid syntax!\n',
        'annotated': 'from .reload_leaf import VALUE\nNEW = "new"\n',
        'augmented': 'from .reload_leaf import VALUE\nNEW = "new"\n',
    }
    provider_path.write_text(provider_sources.get(mode, "NEW = 'new'\n"))
    consumer_sources = {
        'annotated': 'from typing import TYPE_CHECKING\nTYPE_CHECKING: bool = True\n'
        'if TYPE_CHECKING:\n    from .reload_provider import NEW\nVALUE = NEW\n',
        'augmented': 'from typing import TYPE_CHECKING\nTYPE_CHECKING |= True\n'
        'if TYPE_CHECKING:\n    from .reload_provider import NEW\nVALUE = NEW\n',
        'absolute': 'from pydantic_clai2.reload_provider import NEW\nVALUE = NEW\n',
        'module': 'import pydantic_clai2.reload_provider as dependency\nVALUE = dependency.NEW\n',
        'relative_module': 'from . import reload_provider as dependency\nVALUE = dependency.NEW\n',
        'class': 'class Values:\n    from .reload_provider import NEW\nVALUE = Values.NEW\n',
    }
    new_consumer = consumer_sources.get(mode, 'from .reload_provider import NEW\nVALUE = NEW\n')

    if mode.startswith('guard:'):
        (package / 'reload_inactive.py').write_text('from .reload_consumer import VALUE as NEW\n')
        new_consumer = conditional_source(mode.removeprefix('guard:'))
    elif mode in ('branch_alias', 'agreed_alias'):
        provider_path.write_text('from .reload_leaf import VALUE\nNEW = "new"\n')
        second = 'sys.platform' if mode == 'agreed_alias' else 'os.name'
        module, attr = second.split('.')
        new_consumer = (
            'if bool(1):\n    from sys import platform as runtime_platform\n'
            f'else:\n    from {module} import {attr} as runtime_platform\n'
            f'if runtime_platform == {sys.platform!r}:\n    from .reload_provider import NEW\nVALUE = NEW\n'
        )
    elif mode == 'class_scope':
        provider_path.write_text(
            'import typing\n'
            'class Local:\n    typing = object()\n'
            'if typing.TYPE_CHECKING:\n    from .reload_consumer import VALUE\n'
            "NEW = 'new'\n"
        )
    elif mode == 'unknown_guards':
        provider_path.write_text(
            'import sys\n'
            'sys = object()\n'
            'from typing import TYPE_CHECKING\n'
            'TYPE_CHECKING = False\n'
            'if not bool(0):\n    from . import reload_leaf\n'
            'if (True, object()):\n    from . import reload_leaf\n'
            'if 0 < 1 < 2:\n    from . import reload_leaf\n'
            'if 1 is not None:\n    from . import reload_leaf\n'
            "NEW = 'new'\n"
        )
    elif mode == 'reverse':
        consumer_path.write_text('from . import reload_provider\nVALUE = reload_provider.VALUE\n')
        importlib.reload(consumer)
        new_consumer = "NEW = 'new'\nVALUE = NEW\n"
        provider_path.write_text('from .reload_consumer import NEW\n')
    elif mode == 'lazy':
        provider_path.write_text(
            'import typing\n'
            'from typing import TYPE_CHECKING\n'
            'if TYPE_CHECKING:\n    from .reload_consumer import VALUE\n'
            'else:\n    CONSTANT = 1\n'
            'if typing.TYPE_CHECKING:\n    from .reload_consumer import VALUE\n'
            'def lazy():\n    from .reload_consumer import VALUE\n    return VALUE\n'
            'async def async_lazy():\n    from .reload_consumer import VALUE\n    return VALUE\n'
            "NEW = 'new'\n"
        )
    elif mode == 'inactive':
        (package / 'unused.py').write_text('raise RuntimeError("must not import inactive modules")\n')
        new_consumer += 'if False:\n    from .unused import VALUE\n'
    elif mode in ('new_package', 'import_error', 'build_error'):
        bridge = package / 'reload_bridge'
        bridge.mkdir()
        (bridge / '__init__.py').write_text('from ..reload_provider import NEW\nfrom .bridge import VALUE\n')
        bridge_path = bridge / 'bridge.py'
        bridge_path.write_text("from ..reload_provider import NEW\nVALUE = 'old'\n")
        py_compile.compile(str(bridge_path), doraise=True)
        timestamp = bridge_path.stat()
        bridge_path.write_text('from ..reload_provider import NEW\nVALUE = NEW  \n')
        assert bridge_path.stat().st_size == timestamp.st_size
        os.utime(bridge_path, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))
        new_consumer = 'from .reload_bridge.bridge import VALUE\n'

    consumer_path.write_text(new_consumer)
    if mode == 'import_error':
        consumer_path.write_text(new_consumer + 'raise RuntimeError("failed import")\n')

    def build() -> object:
        if mode == 'build_error':
            raise RuntimeError('failed build')
        return consumer.VALUE

    if mode in ('import_error', 'build_error', 'cycle', 'syntax', 'invalid_guard'):
        expected = {'cycle': CycleError, 'syntax': SyntaxError, 'invalid_guard': TypeError}.get(mode, RuntimeError)
        with pytest.raises(expected):
            reload_clai(build)
        assert vars(consumer) == original_consumer
        assert vars(provider) == original_provider
        assert 'pydantic_clai2.reload_bridge' not in sys.modules
        assert 'pydantic_clai2.reload_bridge.bridge' not in sys.modules
        assert 'reload_bridge' not in vars(sys.modules['pydantic_clai2'])
        provider_path.write_text("NEW = 'new'\n")
        consumer_path.write_text(new_consumer)

    assert reload_clai(lambda: consumer.VALUE) == 'new'
    assert sys.modules['pydantic_clai2.reload_consumer'] is consumer
    assert sys.modules['pydantic_clai2.reload_provider'] is provider
    assert 'pydantic_clai2.unused' not in sys.modules


main(Path(sys.argv[1]), sys.argv[2])
