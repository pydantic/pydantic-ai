"""Shell commands split into Bash and the code they embed, each with its own language."""

import pytest
from rich.syntax import Syntax

from pydantic_clai2.ui.rendering import command_syntax
from pydantic_clai2.ui.rendering.command_syntax import command_segments


@pytest.mark.parametrize(
    ('command', 'expected'),
    [
        pytest.param(
            "cat > /tmp/probe.py <<'PYEOF'\nimport os\nPYEOF\npython3 /tmp/probe.py",
            [
                ("cat > /tmp/probe.py <<'PYEOF'\n", 'bash'),
                ('import os\n', 'python'),
                ('PYEOF\npython3 /tmp/probe.py', 'bash'),
            ],
            id='written-to-a-file',
        ),
        pytest.param(
            "python - <<'EOF'\nprint(1)\nEOF",
            [("python - <<'EOF'\n", 'bash'), ('print(1)\n', 'python'), ('EOF', 'bash')],
            id='read-by-an-interpreter',
        ),
        pytest.param(
            'uv run --with rich /usr/bin/python3.12 <<EOF\nx = 1\nEOF',
            [('uv run --with rich /usr/bin/python3.12 <<EOF\n', 'bash'), ('x = 1\n', 'python'), ('EOF', 'bash')],
            id='interpreter-behind-a-runner',
        ),
        pytest.param(
            'sqlite3 data.db <<SQL\nSELECT 1;\nSQL',
            [('sqlite3 data.db <<SQL\n', 'bash'), ('SELECT 1;\n', 'sql'), ('SQL', 'bash')],
            id='interpreter-with-an-argument',
        ),
        pytest.param(
            'python3 run.py <<EOF\ndata\nEOF',
            [('python3 run.py <<EOF\n', 'bash'), ('data\n', None), ('EOF', 'bash')],
            id='script-reads-data',
        ),
        pytest.param(
            'python3 run.py <<JSON\n{}\nJSON',
            [('python3 run.py <<JSON\n', 'bash'), ('{}\n', 'json'), ('JSON', 'bash')],
            id='script-reads-data-the-delimiter-names',
        ),
        pytest.param(
            'cat <<EOF > out.sql\nSELECT 1;\nEOF',
            [('cat <<EOF > out.sql\n', 'bash'), ('SELECT 1;\n', 'sql'), ('EOF', 'bash')],
            id='redirect-after-the-operator',
        ),
        pytest.param(
            "sudo tee -a /etc/app.yaml > /dev/null <<'EOF'\nkey: 1\nEOF",
            [("sudo tee -a /etc/app.yaml > /dev/null <<'EOF'\n", 'bash'), ('key: 1\n', 'yaml'), ('EOF', 'bash')],
            id='tee',
        ),
        pytest.param(
            'cat > notes.txt <<EOF\nhello\nEOF',
            [('cat > notes.txt <<EOF\n', 'bash'), ('hello\n', None), ('EOF', 'bash')],
            id='unknown-extension',
        ),
        pytest.param(
            'cat <<-PYEOF\n\tx = 1\n\tPYEOF\necho done',
            [('cat <<-PYEOF\n', 'bash'), ('\tx = 1\n', 'python'), ('\tPYEOF\necho done', 'bash')],
            id='delimiter-with-tabs-stripped',
        ),
        pytest.param(
            'cat <<"EOF_SQL" | psql -f -\nSELECT 1;\nEOF_SQL',
            [('cat <<"EOF_SQL" | psql -f -\n', 'bash'), ('SELECT 1;\n', 'sql'), ('EOF_SQL', 'bash')],
            id='delimiter',
        ),
        pytest.param(
            'cat <<A <<B\na\nA\nb\nB',
            [('cat <<A <<B\n', 'bash'), ('a\n', None), ('A\n', 'bash'), ('b\n', None), ('B', 'bash')],
            id='two-heredocs',
        ),
        pytest.param(
            "bash <<'X'\necho hi",
            [("bash <<'X'\n", 'bash'), ('echo hi', 'bash')],
            id='unterminated',
        ),
        pytest.param(
            'echo $((1<<2)); python3 <<< "1"', [('echo $((1<<2)); python3 <<< "1"', 'bash')], id='not-heredocs'
        ),
        pytest.param('', [], id='empty'),
    ],
)
def test_heredoc_bodies_take_their_language(command: str, expected: list[tuple[str, str | None]]) -> None:
    segments = command_segments(command)
    assert [(segment.text, segment.language) for segment in segments] == expected
    assert ''.join(segment.text for segment in segments) == command


@pytest.mark.parametrize(
    ('command', 'expected'),
    [
        pytest.param(
            """python3 -c 'import sys' && node -e "console.log(1)" """,
            ['import sys', 'console.log(1)'],
            id='single-and-double-quoted',
        ),
        pytest.param("psql -X -c 'SELECT 1'", ['SELECT 1'], id='after-other-flags'),
        pytest.param('node -e "console.log($HOME)"', [], id='expanded-by-the-shell'),
        pytest.param("python -m 'http.server'", [], id='other-flag'),
        pytest.param("grep -e 'import' file", [], id='not-an-interpreter'),
    ],
)
def test_inline_code_takes_its_language(command: str, expected: list[str]) -> None:
    (segment,) = command_segments(command)
    assert segment.language == 'bash'
    assert [command[start:end] for start, end, _ in segment.embedded] == expected


def test_every_language_has_a_lexer() -> None:
    languages = {*command_syntax._INTERPRETERS.values(), *command_syntax._NAMES.values()}  # pyright: ignore[reportPrivateUsage]
    assert all(Syntax('', language).lexer is not None for language in languages)
    assert set(command_syntax._INTERPRETERS.values()) <= set(command_syntax._INLINE_FLAGS)  # pyright: ignore[reportPrivateUsage]
