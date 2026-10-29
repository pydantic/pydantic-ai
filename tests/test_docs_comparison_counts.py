import re
from pathlib import Path

import pytest

DOCS = Path(__file__).parent.parent / 'docs'
COMPARISON_PAGES = sorted((DOCS / 'comparisons').glob('vs-*.md'))


def table_row_count(path: Path, start: str, end: str = '\n\n') -> int:
    section = path.read_text(encoding='utf-8').split(start, 1)[1].split(end, 1)[0]
    return sum(line.startswith('| [') for line in section.splitlines())


COUNTS = {
    'durable execution integrations': table_row_count(DOCS / 'durable_execution' / 'overview.md', '| Engine |'),
    'Harness capabilities': table_row_count(DOCS / 'harness' / 'index.md', '## Capabilities', '\n## When do you need'),
    'model providers': table_row_count(DOCS / 'models' / 'overview.md', '| Provider and setup |'),
    'realtime providers': table_row_count(DOCS / 'realtime' / 'overview.md', '| Provider | Audio output |'),
}

NUMBER_WORDS = {
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
}


def asserted_count(match: re.Match[str], actual: int, label: str) -> None:
    stated = match.group('count').lower()
    claimed = NUMBER_WORDS.get(stated, int(stated) if stated.isdigit() else 0)
    if match.groupdict().get('plus'):
        assert claimed <= actual < claimed + 10, f'{label} claims {claimed}+, but its source lists {actual}'
    else:
        assert claimed == actual, f'{label} claims {claimed}, but its source lists {actual}'


@pytest.mark.parametrize('page', COMPARISON_PAGES, ids=lambda path: path.stem)
def test_comparison_page_counts(page: Path):
    text = page.read_text(encoding='utf-8')

    durable = re.search(r'\[(?P<count>\w+) integrations\]\(\.\./durable_execution/overview\.md\)', text)
    harness = re.search(r'\[(?P<count>\d+)(?P<plus>\+) with the Harness SDK\]', text)
    assert durable and harness
    asserted_count(durable, COUNTS['durable execution integrations'], f'{page}: durable execution integrations')
    asserted_count(harness, COUNTS['Harness capabilities'], f'{page}: Harness capabilities')

    model_providers = re.search(
        r'\[(?P<count>\d+)(?P<plus>\+)?(?: model)? providers\]\(\.\./models/overview\.md\)', text
    )
    if model_providers:
        asserted_count(model_providers, COUNTS['model providers'], f'{page}: model providers')

    for realtime in re.finditer(r'(?P<count>\w+) providers', text, re.IGNORECASE):
        if realtime.group('count').lower() in NUMBER_WORDS:
            asserted_count(realtime, COUNTS['realtime providers'], f'{page}: realtime providers')
