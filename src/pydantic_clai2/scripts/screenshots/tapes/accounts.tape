# /accounts: signed-in accounts by provider.
Source tapes/common.tape
Output shots/accounts.txt
Hide
Type 'clai-demo' Enter
Wait+Screen /\| ready/
Type '/accounts' Enter
Wait+Screen /personal/
Show
Sleep 1500ms
Screenshot shots/accounts.png
Hide
Escape
Hide
Ctrl+D
