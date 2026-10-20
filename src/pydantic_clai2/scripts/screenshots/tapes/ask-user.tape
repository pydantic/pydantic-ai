# The model asks a multiple-choice question inline, then writes the file you chose.
Source tapes/common.tape
Output shots/ask-user.txt
Hide
Type 'clai-demo' Enter
Wait+Screen /\| ready/
Show
Type 'Set up CI for this repo' Enter
Wait+Screen /Where should the tests run/
Sleep 1s
Screenshot shots/ask-user.png
Hide
Ctrl+D
