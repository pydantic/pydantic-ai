---
description: "Let a Pydantic AI agent drive a Chrome tab with raw Chrome DevTools Protocol commands, using the Browser capability."
---

# Browser

The [`Browser`][pydantic_ai.capabilities.Browser] [capability](overview.md) gives your agent a Chrome tab, driven with raw [Chrome DevTools Protocol](https://chromedevtools.github.io/devtools-protocol/) (CDP) commands through one `browser_cdp` tool. Each run launches its own Chrome with a throwaway profile, and closes it when the run ends.

It needs Chrome or Chromium installed, and the `browser` optional group:

```bash
pip install "pydantic-ai-slim[browser]"
```

```python {title="browser.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Browser

agent = Agent('{{ anthropic:medium }}', capabilities=[Browser()])
result = agent.run_sync('List the titles of the books in the Mystery category on books.toscrape.com.')
print(result.output)
```

Set `headless=False` to watch it work, and `chrome_path` (or the `CHROME_PATH` environment variable) to choose the binary.

## Writing code against the browser

Raw CDP is a low-level API: a single click is several commands, and extracting data means evaluating JavaScript. Models do best with it when they can write code that calls `browser_cdp` in loops, rather than calling the tool once per command. Combine `Browser` with a code execution capability such as [`CodeMode`](../harness/code-mode.md) from [Pydantic AI Harness](../harness/index.md), which runs the model's Python in the [Monty](https://github.com/pydantic/monty) sandbox:

```python {title="browser_code_mode.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Browser
from pydantic_ai_harness import CodeMode

agent = Agent('{{ anthropic:medium }}', capabilities=[Browser(), CodeMode()])
```

## Security

Only the commands in `allowed_methods` can be sent. The defaults read and interact with the page; they leave out commands that reach beyond the tab (`Browser.*`, `Target.*`) or take host file paths (`DOM.setFileInputFiles`, `Input.dispatchDragEvent`), `Page.navigate` only opens `http://` and `https://` URLs, and downloads are denied.

The page can still reach anything Chrome can reach on the network, including services on your machine and private networks, and whatever the model reads from a page is untrusted content that can try to steer it. Run it where that's acceptable.
