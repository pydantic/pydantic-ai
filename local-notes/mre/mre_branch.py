# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pydantic-ai @ file:///home/pydanty/pydantic-ai",
# ]
# ///

"""MRE for the fixed evals link in the local branch.

Reads docs/index.md from the local worktree and asserts the fixed absolute
link `[evaluate](https://ai.pydantic.dev/evals)` is present.
"""

from pathlib import Path
import sys

def main() -> None:
    # The repo root is alongside this script's great-grandparent (local-notes/mre/)
    repo_root = Path(__file__).parent.parent.parent
    index_md = repo_root / "docs" / "index.md"
    content = index_md.read_text()
    if "[evaluate](https://ai.pydantic.dev/evals)" in content:
        print("FIX VERIFIED: docs/index.md contains absolute evals link.")
        sys.exit(0)
    else:
        print("FIX MISSING: docs/index.md still uses relative evals link.")
        sys.exit(1)

if __name__ == "__main__":
    main()
