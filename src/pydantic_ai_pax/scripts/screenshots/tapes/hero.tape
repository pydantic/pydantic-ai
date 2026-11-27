# The startup banner with a real model turn: tool calls grouped on one line, a short answer, and the
# status row with the model, its context window and the cost. Rendered with a real model, not the
# scripted demo: run `PAX_SHOTS_MODEL=anthropic:claude-opus-5-5 capture.sh hero` with
# ANTHROPIC_API_KEY set. The terminal is tall so nothing scrolls off; capture.sh trims the empty rows
# below the status row afterwards.
Source tapes/common.tape
Set Height 2400
Set WaitTimeout 240s
Output shots/hero.txt
Hide
Type 'pax-demo' Enter
Wait+Screen /\| ready/
Type '/set display.tool_calls grouped' Enter
Sleep 2s
Ctrl+D
Sleep 1s
Type 'clear; pax-demo' Enter
Wait+Screen /\| ready/
Show
Type 'Briefly: how is a forecast fetched? Three numbered steps' Enter
Wait+Screen /3\. /
Sleep 30s
Screenshot shots/hero.png
Hide
Ctrl+D
