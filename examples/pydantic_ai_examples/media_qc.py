"""Media QC agent — a fix-and-verify loop over a local stdio MCP server.

An agent that checks a media file for loudness compliance against a formal
broadcast standard (EBU R 128) using the loudcheck MCP server, applies the
exact gain correction the verdict recommends via a native tool, and
re-checks until the file passes — returning a structured QC report.

Requires ffmpeg >= 5 on PATH and `uv` (the MCP server is fetched with uvx
on first run, no install step). If no file is given, an intentionally
too-loud test tone is generated so the example is fully self-contained.

Run with:

    uv run -m pydantic_ai_examples.media_qc [path/to/file.wav]
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from fastmcp.client.transports import StdioTransport
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset


class QCReport(BaseModel):
    """Structured result of the compliance check."""

    file: str
    standard: str
    verdict: str
    """pass | fail — the final verdict after any corrections."""
    integrated_lufs: float
    corrections_applied: list[str]
    summary: str


# loudcheck's MCP server exposes check_loudness(path, standard) and
# list_standards(); uvx fetches it from PyPI on first run.
loudcheck = MCPToolset(
    StdioTransport(
        command='uvx', args=['--from', 'loudcheck[mcp]', 'loudcheck', '--mcp']
    )
)

agent = Agent(
    'openai:gpt-5.2',
    output_type=QCReport,
    toolsets=[loudcheck],
    instructions=(
        'You are a media QC operator. Check the file for EBU R 128 loudness '
        'compliance with check_loudness. If it fails, the verdict includes a '
        'remediation with the exact gain delta — apply it with apply_gain, '
        'then re-check the corrected file. Report the final state.'
    ),
)


@agent.tool_plain
def apply_gain(path: str, gain_db: float) -> str:
    """Apply a gain correction to a media file with ffmpeg.

    Writes a corrected copy next to the original and returns its path.
    """
    src = Path(path)
    dst = src.with_name(f'{src.stem}.fixed{src.suffix}')
    subprocess.run(
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(src), '-af', f'volume={gain_db}dB', str(dst)],
        check=True,
    )
    return str(dst)


def demo_file() -> str:
    """Generate a test tone ~3.6 LU louder than the EBU R 128 target."""
    path = Path(tempfile.mkdtemp()) / 'master.wav'
    subprocess.run(
        ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i', 'sine=frequency=997:duration=5', '-af', 'volume=1.7dB', str(path)],
        check=True,
    )
    return str(path)


if __name__ == '__main__':
    media = sys.argv[1] if len(sys.argv) > 1 else demo_file()
    result = agent.run_sync(f'Run loudness QC on {media}')
    print(result.output.model_dump_json(indent=2))
