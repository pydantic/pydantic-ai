from __future__ import annotations as _annotations

import re
import time
import urllib.parse
from pathlib import Path

from jinja2 import Environment
from mkdocs.config import Config
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page
from snippets import inject_snippets

DOCS_ROOT = Path(__file__).parent.parent


def on_page_markdown(markdown: str, page: Page, config: Config, files: Files) -> str:
    """Called on each file after it is read and before it is converted to HTML."""
    markdown = inject_snippets(markdown, (DOCS_ROOT / page.file.src_uri).parent)
    markdown = replace_uv_python_run(markdown)
    markdown = render_examples(markdown)
    markdown = render_video(markdown)
    markdown = create_gateway_toggle(markdown)
    return markdown


# path to the main mkdocs material bundle file, found during `on_env`
bundle_path: Path | None = None


def on_env(env: Environment, config: Config, files: Files) -> Environment:
    global bundle_path
    for file in files:
        if re.match('assets/javascripts/bundle.[a-z0-9]+.min.js', file.src_uri):
            bundle_path = Path(file.dest_dir) / file.src_uri

    env.globals['build_timestamp'] = str(int(time.time()))
    return env


def on_post_build(config: Config) -> None:
    """Inject extra CSS into mermaid styles to avoid titles being the same color as the background in dark mode."""
    assert bundle_path is not None
    if bundle_path.exists():
        content = bundle_path.read_text()
        content, _ = re.subn(r'}(\.statediagram)', '}.statediagramTitleText{fill:#888}\1', content, count=1)
        bundle_path.write_text(content)


def replace_uv_python_run(markdown: str) -> str:
    return re.sub(r'```bash\n(.*?)(python/uv[\- ]run|pip/uv[\- ]add|py-cli)(.+?)\n```', sub_run, markdown)


def sub_run(m: re.Match[str]) -> str:
    prefix = m.group(1)
    command = m.group(2)
    if 'pip' in command:
        pip_base = 'pip install'
        uv_base = 'uv add'
    elif command == 'py-cli':
        pip_base = ''
        uv_base = 'uv run'
    else:
        pip_base = 'python'
        uv_base = 'uv run'
    suffix = m.group(3)
    return f"""\
=== "pip"

    ```bash
    {prefix}{pip_base}{suffix}
    ```

=== "uv"

    ```bash
    {prefix}{uv_base}{suffix}
    ```"""


EXAMPLES_DIR = Path(__file__).parent.parent.parent / 'examples'


def render_examples(markdown: str) -> str:
    return re.sub(r'^#! *examples/(.+)', sub_example, markdown, flags=re.M)


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


def create_gateway_toggle(markdown: str) -> str:
    """Transform Python code blocks with Agent() calls to show both Pydantic AI and Gateway versions."""
    # Pattern matches Python code blocks with or without attributes, and optional annotation definitions after
    # Annotation definitions are numbered list items like "1. Some text" that follow the code block
    return re.sub(
        r'```python(?:\s+\{([^}]*)\})?\n(.*?)\n```(\n\n(?:\d+\..+(?:\n(?:   .+|\n))*)+)?',
        transform_gateway_code_block,
        markdown,
        flags=re.MULTILINE | re.DOTALL,
    )


# Models that should get gateway transformation
GATEWAY_MODELS = ('anthropic', 'openai', 'openai-responses', 'openai-chat', 'bedrock', 'google-vertex', 'groq')


def transform_gateway_code_block(m: re.Match[str]) -> str:
    """Transform a single code block to show both versions if it contains Agent() calls."""
    attrs = m.group(1) or ''
    code = m.group(2)
    annotations = m.group(3) or ''  # Capture annotation definitions if present

    # Simple check: does the code contain both "Agent(" and a quoted string?
    if 'Agent(' not in code:
        attrs_str = f' {{{attrs}}}' if attrs else ''
        return f'```python{attrs_str}\n{code}\n```{annotations}'

    # Check if code contains Agent() with a model that should be transformed
    # Look for Agent(...'model:...' or Agent(..."model:..."
    agent_pattern = r'Agent\((?:(?!["\']).)*([\"\'])([^"\']+)\1'
    agent_match = re.search(agent_pattern, code, flags=re.DOTALL)

    if not agent_match:
        # No Agent() with string literal found
        attrs_str = f' {{{attrs}}}' if attrs else ''
        return f'```python{attrs_str}\n{code}\n```{annotations}'

    model_string = agent_match.group(2)
    # Check if model starts with one of the gateway-supported models
    should_transform = any(model_string.startswith(f'{model}:') for model in GATEWAY_MODELS)

    if not should_transform:
        # Model doesn't match gateway models, return original
        attrs_str = f' {{{attrs}}}' if attrs else ''
        return f'```python{attrs_str}\n{code}\n```{annotations}'

    # Transform the code for gateway version
    def replace_agent_model(match: re.Match[str]) -> str:
        """Replace model string with gateway/ prefix."""
        full_match = match.group(0)
        quote = match.group(1)
        model = match.group(2)

        # Replace the model string while preserving the rest
        return full_match.replace(f'{quote}{model}{quote}', f'{quote}gateway/{model}{quote}', 1)

    # This pattern finds: "Agent(" followed by anything (lazy), then the first quoted string
    gateway_code = re.sub(
        agent_pattern,
        replace_agent_model,
        code,
        flags=re.DOTALL,
    )

    # Build attributes string
    attrs_str = f' {{{attrs}}}' if attrs else ''

    # Indent code lines for proper markdown formatting within tabs
    # Always add 4 spaces to every line (even empty ones) to preserve annotations
    code_lines = code.split('\n')
    indented_code = '\n'.join('    ' + line for line in code_lines)

    gateway_code_lines = gateway_code.split('\n')
    indented_gateway_code = '\n'.join('    ' + line for line in gateway_code_lines)

    # Indent annotation definitions if present (need to be inside tabs for Material to work)
    indented_annotations = ''
    if annotations:
        # Remove leading newlines and indent each line with 4 spaces
        annotation_lines = annotations.strip().split('\n')
        indented_annotations = '\n\n' + '\n'.join('    ' + line for line in annotation_lines)

    return f"""\
=== "Pydantic AI Gateway"

    ```python{attrs_str}
{indented_gateway_code}
    ```{indented_annotations}

=== "Pydantic AI"

    ```python{attrs_str}
{indented_code}
    ```{indented_annotations}"""
