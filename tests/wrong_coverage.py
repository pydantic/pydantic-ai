import re
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel, Field
from rich.console import Console

EXCLUDE_COMMENT = 'pragma: no cover'


def main() -> int:
    with NamedTemporaryFile(suffix='.json') as coverage_json:
        with NamedTemporaryFile(mode='w', suffix='.toml') as config_file:
            config_file.write(f"[tool.coverage.report]\nexclude_lines = ['{EXCLUDE_COMMENT}']\n")
            config_file.flush()
            subprocess.run(
                ['uv', 'run', 'coverage', 'json', f'--rcfile={config_file.name}', '-o', coverage_json.name],
                check=True,
            )

        r = CoverageReport.model_validate_json(coverage_json.read())

    blocks: list[str] = []
    total_lines = 0
    cwd = Path.cwd()
    for file_name, file_coverage in r.files.items():
        # Find lines that are both excluded and executed
        common_lines = sorted(set(file_coverage.excluded_lines) & set(file_coverage.executed_lines))

        if not common_lines:
            continue

        code_analysise: CodeAnalyzer | None = None

        def add_block(start: int, end: int):
            nonlocal code_analysise, total_lines

            if code_analysise is None:
                code_analysise = CodeAnalyzer(file_name)

            if all(code_analysise.is_expression_start(line_no) for line_no in range(start, end + 1)):
                return

            b = str(start) if start == end else f'{start}-{end}'
            if not blocks or blocks[-1] != b:
                total_lines += end - start + 1
                blocks.append(f'  [link=file://{cwd / file_name}]{file_name} {b}[/link]')

        first_line, *rest = common_lines
        current_start = current_end = first_line

        for line in rest:
            if line == current_end + 1:
                current_end = line
            else:
                # Start a new block
                add_block(current_start, current_end)
                current_start = current_end = line

        add_block(current_start, current_end)

    console = Console()
    if blocks:
        console.print(f"❎ {total_lines} lines marked with '{EXCLUDE_COMMENT}' and covered")
        for block in blocks:
            console.print(block)
        return 1
    else:
        console.print(f"✅ No lines wrongly marked with '{EXCLUDE_COMMENT}'")
        return 0


class FunctionSummary(BaseModel):
    covered_lines: int
    num_statements: int
    percent_covered: float
    percent_covered_display: str
    missing_lines: int
    excluded_lines: int
    num_branches: int
    num_partial_branches: int
    covered_branches: int
    missing_branches: int


class FileCoverage(BaseModel):
    executed_lines: list[int]
    summary: FunctionSummary
    missing_lines: list[int]
    excluded_lines: list[int]
    executed_branches: list[list[int]] = Field(default_factory=list)
    missing_branches: list[list[int]] = Field(default_factory=list)


class CoverageReport(BaseModel):
    files: dict[str, FileCoverage]


EXPRESSION_START_REGEX = re.compile(r'\s*(?:def|async def|@|class|if|elif|else)')


class CodeAnalyzer:
    def __init__(self, file_path: str) -> None:
        with open(file_path) as f:
            content = f.read()
        self.lines: dict[int, str] = dict(enumerate(content.splitlines(), start=1))

    def is_expression_start(self, line_no: int) -> bool:
        line = self.lines[line_no]
        return bool(EXPRESSION_START_REGEX.match(line))


if __name__ == '__main__':
    sys.exit(main())
