# Speculative execution on (Ctrl+X Ctrl+S): read-only calls start while the model writes.
Source tapes/common.tape
Output shots/code-mode.txt
Hide
Type 'clai-demo' Enter
Wait+Screen /\| ready/
Ctrl+X
Ctrl+S
Sleep 500ms
Show
Type 'Explain how a forecast is fetched' Enter
Wait+Screen /tests run offline/
Wait+Screen /\| ready/
Sleep 1s
Screenshot shots/code-mode.png
Hide
Ctrl+D
