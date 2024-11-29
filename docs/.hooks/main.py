from __future__ import annotations as _annotations

import re
import urllib.parse
from pathlib import Path

from mkdocs.config import Config
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page


def on_page_markdown(markdown: str, page: Page, config: Config, files: Files) -> str:
    """Called on each file after it is read and before it is converted to HTML."""
    markdown = replace_uv_python_run(markdown)
    markdown = render_examples(markdown)
    markdown = render_video(markdown)
    return markdown


def replace_uv_python_run(markdown: str) -> str:
    return re.sub(r'```bash\n(.*?)(python/uv[\- ]run|pip/uv[\- ]add)(.+?)\n```', sub_run, markdown)


def sub_run(m: re.Match[str]) -> str:
    prefix = m.group(1)
    command = m.group(2)
    install = 'pip' in command
    suffix = m.group(3)
    return f"""\
=== "pip"

    ```bash
    {prefix}{'pip install' if install else 'python'}{suffix}
    ```

=== "uv"

    ```bash
    {prefix}uv {'add' if install else 'run'}{suffix}
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


def render_video(markdown: str) -> str:
    return re.sub(r'\{\{ *video\((["\'])(.+?)\1(?:, (\d+))?(?:, (\d+))?\) *\}\}', sub_cf_video, markdown)


def sub_cf_video(m: re.Match[str]) -> str:
    video_id = m.group(2)
    time = m.group(3)
    time = f'{time}s' if time else ''
    padding_top = m.group(4) or '67'

    domain = 'https://customer-nmegqx24430okhaq.cloudflarestream.com'
    poster = f'{domain}/{video_id}/thumbnails/thumbnail.jpg?time={time}&height=600'
    return f"""
<div style="position: relative; padding-top: {padding_top}%;">
  <iframe
    src="{domain}/{video_id}/iframe?poster={urllib.parse.quote_plus(poster)}"
    loading="lazy"
    style="border: none; position: absolute; top: 0; left: 0; height: 100%; width: 100%;"
    allow="accelerometer; gyroscope; autoplay; encrypted-media; picture-in-picture;"
    allowfullscreen="true"
  ></iframe>
</div>
"""
