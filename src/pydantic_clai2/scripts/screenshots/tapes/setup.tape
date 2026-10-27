# Not a screenshot: saves the models, sign-ins, and keys the menu screenshots show. Every value is fake.
Source tapes/common.tape
Hide
Type 'clai-demo' Enter
Wait+Screen /\| ready/
Type '/model anthropic:claude-opus-5-5' Enter
Sleep 500ms
Type '/model anthropic:claude-sonnet-4-6' Enter
Sleep 500ms
Type '/model openai:gpt-5.2' Enter
Sleep 500ms
Type '/model openai-codex:gpt-6-astra' Enter
Sleep 500ms
Type '/model demo:tour' Enter
Sleep 500ms
Type '/set run.instructions "Prefer small, focused changes. Run the tests after every edit."' Enter
Sleep 500ms
Type '/login openai@work' Enter
Wait+Screen /API key for/
Type 'sk-demo-not-a-real-key' Enter
Wait+Screen /openai@work connected/
Type '/login openai@personal' Enter
Sleep 2s
Type 'sk-demo-not-a-real-key' Enter
Wait+Screen /openai@personal connected/
Type '/login anthropic@work' Enter
Sleep 2s
Type 'sk-ant-demo-not-a-real-key' Enter
Wait+Screen /anthropic@work connected/
Type '/keys' Enter
Wait+Screen /Add API key/
Type 'a'
Wait+Screen /Name:/
Type 'LINEAR_API_KEY' Enter
Sleep 1s
Type 'lin-demo-not-a-real-key' Enter
Wait+Screen /LINEAR_API_KEY/
Type 'a'
Wait+Screen /Name:/
Type 'GITHUB_TOKEN' Enter
Sleep 1s
Type 'ghp-demo-not-a-real-key' Enter
Wait+Screen /GITHUB_TOKEN/
Type 'a'
Wait+Screen /Name:/
Type 'NOTION_API_KEY' Enter
Sleep 1s
Type 'ntn-demo-not-a-real-key' Enter
Wait+Screen /NOTION_API_KEY/
Escape
Sleep 1s
Type '/exit' Enter
Wait
Show
Sleep 200ms
