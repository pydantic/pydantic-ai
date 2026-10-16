# The startup banner with a real turn underneath: a thought, tool calls, a diff, and the answer.
Source tapes/common.tape
Set Height 2200
Output shots/hero.txt
Hide
Type 'clai-demo' Enter
Wait+Screen /\| ready/
Show
Type 'Add a --units flag for Fahrenheit'
Enter
Wait+Screen /All three tests still pass/
Wait+Screen /\| ready/
Sleep 1500ms
Screenshot shots/hero.png
Hide
Ctrl+D
