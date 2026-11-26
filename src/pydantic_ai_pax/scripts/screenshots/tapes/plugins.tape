# The /plugins menu: built-ins, a drop-in, and what each one provides.
Source tapes/common.tape
# Tall enough for every built-in plugin, so the menu's title stays on screen.
Set Height 1800
Output shots/plugins.txt
Hide
Type 'pax-demo' Enter
Wait+Screen /\| ready/
Type '/plugins' Enter
Wait+Screen /coder/
Down 2
Show
Sleep 1s
Screenshot shots/plugins.png
Hide
Escape
Hide
Ctrl+D
