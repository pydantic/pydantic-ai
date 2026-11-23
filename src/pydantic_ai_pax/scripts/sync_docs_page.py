"""Regenerate `docs/harness/clai2.md` from `src/pydantic_ai_pax/README.md`.

The README is the source; the docs page adds front matter and drops the badges.
Exits 1 when the page was out of date, after rewriting it, so pre-commit and CI catch drift:

```bash
python src/pydantic_ai_pax/scripts/sync_docs_page.py
```
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parents[3]
README = ROOT / 'src' / 'pydantic_ai_pax' / 'README.md'
PAGE = ROOT / 'docs' / 'harness' / 'clai2.md'
HEADER = """\
---
title: CLAI
description: "CLAI is a coding agent for your terminal, built on Pydantic AI and the Pydantic AI Harness."
---

<!-- Generated from src/pydantic_ai_pax/README.md by src/pydantic_ai_pax/scripts/sync_docs_page.py. Edit the README instead. -->

"""


def main() -> int:
    """Rewrite the page from the README; return 1 if it changed."""
    readme = README.read_text(encoding='utf-8')
    page = HEADER + re.sub(r'^(?:\[!\[.*\n)+\n', '', readme, flags=re.MULTILINE)
    if PAGE.read_text(encoding='utf-8') == page:
        return 0
    PAGE.write_text(page, encoding='utf-8')
    print(f'Regenerated {PAGE.relative_to(ROOT)} from {README.relative_to(ROOT)}.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
