"""Release smoke validation rejects incomplete and stale installed browser bundles."""

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

import pytest


@runtime_checkable
class WebAssetsScript(Protocol):
    check_web_assets: Callable[[Path, Path], None]


def test_packaged_web_assets(tmp_path: Path) -> None:
    script = Path(__file__).parents[2] / 'src' / 'pydantic_clai2' / 'scripts' / 'check_web_assets.py'
    spec = importlib.util.spec_from_file_location('check_web_assets', script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert isinstance(module, WebAssetsScript)

    source = tmp_path / 'source'
    installed = tmp_path / 'installed'
    for directory in (source, installed):
        (directory / 'assets').mkdir(parents=True)
        (directory / 'index.html').write_text('<script src="/assets/app.js"></script><link href="/assets/index.css">')
        (directory / 'assets' / 'app.js').write_text('render()')
        (directory / 'assets' / 'index.css').write_text('body {}')

    module.check_web_assets(installed, source)

    (installed / 'assets' / 'app.js').write_text('stale()')
    with pytest.raises(ValueError, match='differ'):
        module.check_web_assets(installed, source)

    (installed / 'assets' / 'app.js').write_text('')
    with pytest.raises(ValueError, match='empty'):
        module.check_web_assets(installed, source)

    (installed / 'assets' / 'app.js').unlink()
    with pytest.raises(FileNotFoundError):
        module.check_web_assets(installed, source)

    (installed / 'index.html').write_text('<div id="root"></div>')
    with pytest.raises(ValueError, match='JavaScript and CSS'):
        module.check_web_assets(installed, source)
