# Double Esc between turns opens the rewind picker.
Source tapes/common.tape
Output shots/rewind.txt
Hide
Type 'clai-demo' Enter
Wait+Screen /\| ready/
Type 'Explain how a forecast is fetched' Enter
Wait+Screen /tests run offline/
Wait+Screen /\| ready/
Type 'Write tests for the parser' Enter
Wait+Screen /assertEqual/
Wait+Screen /\| ready/
Sleep 500ms
Escape 2
Wait+Screen /Rewind/
Show
Sleep 1s
Screenshot shots/rewind.png
Hide
Escape
Hide
Ctrl+D
