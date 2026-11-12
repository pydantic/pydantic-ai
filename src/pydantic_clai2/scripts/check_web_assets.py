"""Check an installed CLAI 2's web assets against the validated source bundle.

Run with the installed environment's Python and `-I` to exclude the checkout:

```bash
/path/to/venv/bin/python -I src/pydantic_clai2/scripts/check_web_assets.py SOURCE_STATIC
```
"""

import argparse
import re
from pathlib import Path

import pydantic_clai2


def check_web_assets(static: Path, expected: Path) -> None:
    """Require a complete web entry point and byte-identical installed assets."""
    page = (static / 'index.html').read_text(encoding='utf-8')
    assets: list[str] = re.findall(r'(?:src|href)="(/assets/[^"?]+)"', page)
    if not any(asset.endswith('.js') for asset in assets) or not any(asset.endswith('.css') for asset in assets):
        raise ValueError('The packaged web page must reference JavaScript and CSS assets.')
    for asset in assets:
        if not (static / asset.removeprefix('/')).read_bytes():
            raise ValueError(f'The packaged web asset is empty: {asset}')
    packaged: dict[Path, bytes] = {
        path.relative_to(static): path.read_bytes() for path in static.rglob('*') if path.is_file()
    }
    source: dict[Path, bytes] = {
        path.relative_to(expected): path.read_bytes() for path in expected.rglob('*') if path.is_file()
    }
    if packaged != source:
        raise ValueError('Packaged web assets differ from the validated source bundle.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('expected', type=Path)
    args = parser.parse_args()
    assert pydantic_clai2.__file__ is not None
    check_web_assets(Path(pydantic_clai2.__file__).parent / 'builtin_plugins' / 'web' / 'static', args.expected)
