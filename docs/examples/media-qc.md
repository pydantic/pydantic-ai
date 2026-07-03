Media QC agent — a fix-and-verify loop over a local stdio MCP server.

Demonstrates:

* [MCP toolsets over stdio](../mcp/client.md#stdio)
* [structured `output_type`](../output.md)
* mixing MCP tools with a native [tool](../tools.md) in one agent

The agent checks a media file for loudness compliance against a formal
broadcast standard (EBU R 128) using the
[loudcheck](https://github.com/chaoz23/loudcheck) MCP server (spawned with
`uvx`, no install step), applies the exact gain correction the verdict
recommends via a native `ffmpeg` tool, and re-checks until the file passes —
returning a structured QC report.

The interesting shape here is the loop: the MCP tool's output (a verdict
with an exact remediation delta) feeds a native tool (`apply_gain`), whose
output feeds the MCP tool again. The model plans the loop; each measurement
is deterministic.

## Running the Example

Requires `ffmpeg` >= 5 on `PATH` and [`uv`](https://docs.astral.sh/uv/).
With [dependencies installed and environment variables set](./setup.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.media_qc
```

With no argument it generates an intentionally too-loud test tone, so the
run is fully self-contained; pass a path to check your own file:

```bash
python/uv-run -m pydantic_ai_examples.media_qc path/to/master.wav
```

Output looks like:

```json
{
  "file": "/tmp/.../master.fixed.wav",
  "standard": "EBU R 128",
  "verdict": "pass",
  "integrated_lufs": -23.0,
  "corrections_applied": ["applied -3.6 dB gain"],
  "summary": "master.wav was 3.6 LU over target; corrected copy passes EBU R 128."
}
```

## Example Code

```snippet {path="/examples/pydantic_ai_examples/media_qc.py"}```
