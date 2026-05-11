# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "httpx",
# ]
# ///

"""MRE for the broken evals link bug in the published docs.

Fetches docs/index.md from the main branch and asserts the buggy relative
link `[evaluate](evals.md)` is present.
"""

import httpx
import sys

URL = "https://raw.githubusercontent.com/pydantic/pydantic-ai/main/docs/index.md"

def main() -> None:
    resp = httpx.get(URL, follow_redirects=True)
    resp.raise_for_status()
    content = resp.text
    if "[evaluate](evals.md)" in content:
        print("BUG REPRODUCED: docs/index.md contains broken relative evals link.")
        sys.exit(1)
    else:
        print("BUG NOT FOUND: docs/index.md already uses an absolute evals link.")
        sys.exit(0)

if __name__ == "__main__":
    main()
