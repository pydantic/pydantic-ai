# /model add: a provider, then its models with context size and prices.
Source tapes/common.tape
Output shots/model-add.txt
Hide
Type 'clai-demo' Enter
Wait+Screen /\| ready/
Type '/model add' Enter
Wait+Screen /Enter browse models/
Type 'anthropic'
Sleep 500ms
Enter
Wait+Screen /claude-/
Type 'sonnet'
Sleep 500ms
Down 6
Show
Sleep 1s
Screenshot shots/model-add.png
Hide
Escape
Escape
Hide
Ctrl+D
