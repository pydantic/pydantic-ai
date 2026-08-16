Example of a Pydantic AI agent that answers questions about professional tennis using the [Live Tennis API](https://livetennisapi.com).

Demonstrates:

- [tools](../tools.md)
- [agent dependencies](../dependencies.md)
- wrapping a plain REST API as tools with `httpx`

In this case the idea is a "live tennis" agent — the user asks what's happening on the
ATP/WTA tours, and the agent uses the `get_live_matches`, `get_upcoming_fixtures` and
`search_players` tools to look up matches in progress (including who is serving and
whether there's a break point), upcoming fixtures, and player rankings. Each tool makes
one HTTP call and returns a trimmed-down JSON summary so the LLM sees only the fields
it needs.

## Running the Example

You'll need a Live Tennis API key set via `LIVETENNIS_API_KEY`. The free tier
(30 requests/minute, 100/day, no card required) is enough for this example — grab a key
at [livetennisapi.com/subscribe/free](https://livetennisapi.com/subscribe/free).

The example agent runs on `openai:gpt-5-mini`, so you'll also need an OpenAI API key set
via `OPENAI_API_KEY`.

With [dependencies installed and environment variables set](./setup.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.live_tennis_agent
```

## Example Code
```snippet {path="/examples/pydantic_ai_examples/live_tennis_agent.py"}```
