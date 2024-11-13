from __future__ import annotations as _annotations

import re
from pathlib import Path

from mkdocs.config import Config
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page


def on_page_markdown(markdown: str, page: Page, config: Config, files: Files) -> str:
    """Called on each file after it is read and before it is converted to HTML."""
    markdown = replace_uv_python_run(markdown)
    markdown = render_examples(markdown)
    return markdown


def replace_uv_python_run(markdown: str) -> str:
    return re.sub(r'```bash\n(.*?)python/uv[\- ]run (.+?)\n```', sub_run, markdown)


def sub_run(m: re.Match[str]) -> str:
    prefix = m.group(1)
    command = m.group(2)
    return f"""\
=== "pip"

    ```bash
    {prefix}python {command}
    ```

=== "uv"

    ```bash
    {prefix}uv run {command}
    ```"""


EXAMPLES_DIR = Path(__file__).parent.parent.parent / 'pydantic_ai_examples'


def render_examples(markdown: str) -> str:
    return re.sub(r'^#! *pydantic_ai_examples/(.+)', sub_example, markdown, flags=re.M)


def sub_example(m: re.Match[str]) -> str:
    example_path = EXAMPLES_DIR / m.group(1)
    content = example_path.read_text().strip()
    # remove leading docstring which duplicates what's in the docs page
    content = re.sub(r'^""".*?"""', '', content, count=1, flags=re.S).strip()

    return content
