# The /set menu: every setting, with its value, default, and description.
Source tapes/common.tape
Output shots/set.txt
Hide
Type 'clai-demo' Enter
Wait+Screen /\| ready/
Type '/set' Enter
Wait+Screen /model/
Down 2
Show
Sleep 1s
Screenshot shots/set.png
Hide
Escape
Hide
Ctrl+D
