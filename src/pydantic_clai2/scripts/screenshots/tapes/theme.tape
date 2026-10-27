# The /theme picker previewing two bundled palettes. Nothing is applied.
Source tapes/common.tape
Output shots/theme.txt
Hide
Type 'clai-demo' Enter
Wait+Screen /\| ready/
Type '/theme' Enter
Wait+Screen /enter apply/
Type '/tokyo_night'
Show
Sleep 1500ms
Screenshot shots/theme-tokyo-night.png
Hide
Escape
Sleep 300ms
Escape
Sleep 300ms
Escape
Sleep 500ms
Sleep 1s
Type '/theme' Enter
Wait+Screen /enter apply/
Type '/github_light'
Show
Sleep 1500ms
Screenshot shots/theme-github-light.png
Hide
Escape
Sleep 300ms
Escape
Sleep 300ms
Escape
Sleep 500ms
Hide
Ctrl+D
