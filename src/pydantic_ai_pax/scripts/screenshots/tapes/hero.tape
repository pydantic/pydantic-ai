# The startup banner with a real turn underneath: a thought, tool calls, a diff, and the answer.
Source tapes/common.tape
Set Height 2500
Output shots/hero.txt
Hide
Type 'pax-demo' Enter
Wait+Screen /\| ready/
Show
Type 'Add a --units flag for Fahrenheit'
Enter
Wait+Screen /All three tests still\s+pass/
Wait+Screen /\| ready/
Sleep 1500ms
Screenshot shots/hero.png
Hide
Ctrl+D
