"""`clai2` and `pydantic-clai2`: the deprecated names of `pax`."""

import sys

from pydantic_ai_pax.__main__ import main as pax_main

DEPRECATION = (
    '`clai2` is now `pax`: install it with `uv tool install pydantic-ai-pax`. '
    'The `clai2` and `pydantic-clai2` commands will be removed in a future release.'
)


def main() -> None:
    """Run `pax` after pointing at its new name on stderr."""
    print(DEPRECATION, file=sys.stderr)
    pax_main()


if __name__ == '__main__':
    main()
