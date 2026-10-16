# /usage, /cost, and /compact after two turns.
Source tapes/common.tape
Output shots/usage.txt
Hide
Type 'clai-demo' Enter
Wait+Screen /\| ready/
Type 'Add a --units flag for Fahrenheit' Enter
Wait+Screen /All three tests still pass/
Type 'Explain how a forecast is fetched' Enter
Wait+Screen /tests run offline/
Wait+Screen /\| ready/
Ctrl+L
Sleep 500ms
Type '/usage' Enter
Sleep 1s
Type '/cost' Enter
Sleep 1s
Type '/compact' Enter
Wait+Screen /Compacted|smaller/
Show
Sleep 1s
Screenshot shots/usage.png
Hide
Ctrl+D
