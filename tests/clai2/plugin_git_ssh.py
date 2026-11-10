"""An SSH transport that reports its Git parent and waits for cancellation."""

import os
import signal
import socket
import sys

assert os.environ['GIT_TERMINAL_PROMPT'] == '0'
with socket.socket(socket.AF_UNIX) as ready:
    ready.connect(sys.argv[1])
    ready.sendall(f'{os.getppid()} {os.getpid()}'.encode())

while True:
    signal.pause()
