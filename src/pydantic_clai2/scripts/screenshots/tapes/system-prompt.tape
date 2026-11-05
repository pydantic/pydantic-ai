# /system_prompt: what the model is told, and your own instructions.
Source tapes/common.tape
Output shots/system-prompt.txt
Hide
Type 'clai-demo' Enter
Wait+Screen /\| ready/
Type '/system_prompt' Enter
Wait+Screen /instructions/
Show
Sleep 1s
Screenshot shots/system-prompt.png
Hide
Escape
Hide
Ctrl+D
