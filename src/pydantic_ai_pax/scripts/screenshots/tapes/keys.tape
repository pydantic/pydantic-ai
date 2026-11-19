# /keys: saved API keys, values masked.
Source tapes/common.tape
Output shots/keys.txt
Hide
Type 'clai-demo' Enter
Wait+Screen /\| ready/
Type '/keys' Enter
Wait+Screen /GITHUB_TOKEN/
Show
Sleep 1s
Screenshot shots/keys.png
Hide
Escape
Hide
Ctrl+D
