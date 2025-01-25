# example_loader.py
import re
from collections import defaultdict
from pathlib import Path

# Cache for discovered examples
_examples_cache: dict[str, dict[str, Path]] = {}
_example_pattern = re.compile(r'(.+?)(?:_(gradio|streamlit))?\.py$')


def on_startup(**kwargs):
    """Load examples on startup to avoid repeated directory scans"""
    discover_examples()


def discover_examples() -> dict[str, dict[str, Path]]:
    """
    Discover all example files.

    Returns a dictionary where the keys are example names and the values are dictionaries
    with python framework names as keys and file paths as values.

    Example:
    {
        'bank_support': {
            'python': Path('examples/pydantic_ai_examples/bank_support.py'),
        },
        'weather_agent': {
            'python': Path('examples/pydantic_ai_examples/weather_agent.py'),
            'gradio': Path('examples/pydantic_ai_examples/weather_agent_gradio.py'),
        },
    }

    Real examples are located at @examples/pydantic_at_examples/*.py
    """
    if _examples_cache:
        return _examples_cache

    examples: dict[str, dict[str, Path]] = defaultdict(dict)
    examples_path = Path('examples/pydantic_ai_examples')

    for file in examples_path.glob('*.py'):
        match = _example_pattern.match(file.name)
        if match:
            example_name, framework = match.groups()
            examples[example_name][framework or 'python'] = file

    _examples_cache.update(examples)
    return examples


def load_example_code(filepath: Path) -> str:
    """Load code from a Python file"""
    try:
        with open(filepath, encoding='utf-8') as f:
            return f.read().rstrip()
    except FileNotFoundError:
        return f'Error: File {filepath} not found'


def create_tabs(example_name: str) -> str:
    """Create markdown tabs for an example"""
    examples = discover_examples()

    if example_name not in examples:
        return f"Error: Example '{example_name}' not found"

    frameworks = examples[example_name]
    sorted_frameworks = sorted(frameworks.keys(), key=lambda x: (x != 'python', x))

    tab_content: list[str] = []
    for framework in sorted_frameworks:
        filepath = frameworks[framework]
        tab_title = f'{"/".join(filepath.parts[-2:])} ({framework.title()})'
        code = load_example_code(filepath)

        # Each line needs to be indented with 4 spaces for proper tab formatting
        tab = f'=== "{tab_title}"\n'
        tab += '    ```python\n'
        # Indent each line of code with 4 spaces
        tab += '\n'.join(f'    {line}' for line in code.splitlines())
        tab += '\n    ```'
        tab_content.append(tab)

    return '\n'.join(tab_content)


def on_page_markdown(markdown: str, **kwargs) -> str:
    """Process the markdown content"""
    pattern = re.compile(r'{example\("([^"]+)"\)}')

    def replace_match(match: re.Match[str]) -> str:
        example_name = match.group(1)
        return create_tabs(example_name)

    return pattern.sub(replace_match, markdown)
