"""Ports shared by tests. It imports only `socket`, so any test module, fixture or helper script can use it."""

import socket


def reserve_loopback_port() -> int:
    """A loopback port that no other `bind(0)` is handed while it is in `TIME_WAIT`, but a server can still bind.

    A port read from a socket and released goes back to the pool, where another `pytest-xdist` worker can take it
    before the server binds it. Closing an accepted connection first leaves the port itself in `TIME_WAIT`: the
    kernel then skips it when allocating a port, while a server that sets `SO_REUSEADDR` (uvicorn, `http.server`
    and Go's `net.Listen` do) can bind it. Linux allows that bind only if the socket that left the port in
    `TIME_WAIT` set `SO_REUSEADDR` too, so the listener sets it, as Rust's `TcpListener::bind` does for the Temporal
    SDK, which reserves its dev-server ports the same way.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        port: int = listener.getsockname()[1]
        with socket.create_connection(('127.0.0.1', port)):
            accepted, _ = listener.accept()
            accepted.close()
    return port
