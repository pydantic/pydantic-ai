from pathlib import Path


def test_index_evals_link_is_absolute():
    """The evals link in docs/index.md must be absolute so it works from pydantic.dev/docs/ai/."""
    index_md = Path(__file__).parent.parent / 'docs' / 'index.md'
    content = index_md.read_text()
    assert '[evaluate](https://ai.pydantic.dev/evals)' in content
    assert '[evaluate](evals.md)' not in content
