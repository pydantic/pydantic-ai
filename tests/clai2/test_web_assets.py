"""Release smoke validation rejects incomplete and stale installed browser bundles."""

from pathlib import Path

import pytest

from src.pydantic_clai2.scripts.check_web_assets import check_web_assets


def test_packaged_web_assets(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    installed = tmp_path / 'installed'
    for directory in (source, installed):
        (directory / 'assets').mkdir(parents=True)
        (directory / 'index.html').write_text('<script src="/assets/app.js"></script><link href="/assets/index.css">')
        (directory / 'assets' / 'app.js').write_text('render()')
        (directory / 'assets' / 'index.css').write_text('body {}')

    check_web_assets(installed, source)

    (installed / 'assets' / 'app.js').write_text('stale()')
    with pytest.raises(ValueError, match='differ'):
        check_web_assets(installed, source)

    (installed / 'assets' / 'app.js').write_text('')
    with pytest.raises(ValueError, match='empty'):
        check_web_assets(installed, source)

    (installed / 'assets' / 'app.js').unlink()
    with pytest.raises(FileNotFoundError):
        check_web_assets(installed, source)

    (installed / 'index.html').write_text('<div id="root"></div>')
    with pytest.raises(ValueError, match='JavaScript and CSS'):
        check_web_assets(installed, source)
