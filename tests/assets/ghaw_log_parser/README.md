# Vendored gh-aw Claude log parser

[GitHub Agentic Workflows](https://github.com/github/gh-aw) (gh-aw) renders the step summary of a
`engine: claude` run by parsing the Claude Code CLI's `stream-json` output. Running that parser over
a stream produced by [`ClaudeCodeEventStream`][pydantic_ai.ui.claude_code.ClaudeCodeEventStream] is
what proves compatibility rather than claiming it, so `test_claude_code_ui.py` uses it as an oracle.

| | |
|---|---|
| Upstream | [`github/gh-aw`](https://github.com/github/gh-aw) |
| Pinned commit | `fafef5837db7134eb1931954423f5c9d6e0bec3a` |
| Upstream directory | `actions/setup/js/` |
| License | MIT, Copyright GitHub, Inc. — see `LICENSE` |

Every `.cjs` file is verbatim upstream content with a three-line provenance header prepended.

## Files

`parse_claude_log.cjs` is the entry point; the other files are its `require()` closure at the pinned
commit. `main` is not usable standalone (it needs the `actions/github-script` `core` global), so the
test drives the pure `parseClaudeLog(logContent)` export, which reads no argv, stdin, or network.

- `parse_claude_log.cjs` — entry point, exports `parseClaudeLog`
- `log_parser_shared.cjs` — line parsing, legacy → canonical conversion, initialization + information sections
- `log_parser_format.cjs` — conversation/tool rendering and statistics
- `log_parser_bootstrap.cjs` — required eagerly by `parse_claude_log.cjs` at import time
- `log_parser_step_summary_builder.cjs`, `error_codes.cjs`, `error_helpers.cjs`,
  `markdown_unfencing.cjs`, `redact_secrets.cjs` — leaf helpers

## Refreshing the pin

```bash
SHA=<new sha>
for f in parse_claude_log log_parser_shared log_parser_format log_parser_bootstrap \
         log_parser_step_summary_builder error_codes error_helpers markdown_unfencing redact_secrets; do
  curl -sSfL -o "$f.cjs" "https://raw.githubusercontent.com/github/gh-aw/$SHA/actions/setup/js/$f.cjs"
done
curl -sSfL -o LICENSE "https://raw.githubusercontent.com/github/gh-aw/$SHA/LICENSE"
```

Then re-add the provenance headers, update the sha above, and run `node -e "require('./parse_claude_log.cjs')"`
to confirm the `require()` closure is still complete.
