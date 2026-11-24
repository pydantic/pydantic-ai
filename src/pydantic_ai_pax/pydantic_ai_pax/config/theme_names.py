"""The theme choices shared by settings validation and the terminal picker."""


def names() -> tuple[str, ...]:
    """Offer the unchanged default appearance and Termflow's bundled palettes."""
    from termflow.themes import PALETTES

    return ('default', *PALETTES)
