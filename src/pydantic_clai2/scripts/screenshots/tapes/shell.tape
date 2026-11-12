# !command runs in your shell; the output is saved for the next prompt.
Source tapes/common.tape
Output shots/shell.txt
Hide
Type 'clai-demo' Enter
Wait+Screen /\| ready/
Show
Type '!git log --oneline' Enter
Wait+Screen /Done/
Type '!python3 -m weather forecast lisbon' Enter
Sleep 2s
Screenshot shots/shell.png
Hide
Ctrl+D
