"""Entry point for `python -m pydantic_ai_gh_aw_shim` / `runpy.run_module`.

All shim logic lives in `cli.py` (which tests import directly). Keeping
this file a one-call stub avoids the `runpy.run_module(..., run_name="__main__")`
+ PEP-563 corner case where pydantic-ai's runtime annotation inspection
can't find `RunContext` in a module loaded under `__name__ == "__main__"`.
"""

import sys

from .cli import main

if __name__ == '__main__':
    sys.exit(main())
