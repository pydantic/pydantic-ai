# /fork runs a copy of the conversation in the background.
Source tapes/common.tape
Output shots/fork.txt
Hide
Type 'clai-demo' Enter
Wait+Screen /\| ready/
Type 'Add a --units flag for Fahrenheit' Enter
Wait+Screen /All three tests still pass/
Wait+Screen /\| ready/
Ctrl+L
Show
Type '/fork write tests for the parser' Enter
Wait+Screen /FORK #1 RESPONSE/
Sleep 1s
Screenshot shots/fork.png
Hide
Ctrl+D
