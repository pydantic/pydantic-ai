import click

@click.group()
def mcp():
    """Manage Model Context Protocol (MCP) servers."""
    pass

@mcp.command()
@click.argument('server_url')
def add(server_url):
    """Add a new MCP server."""
    click.echo(f"Adding MCP server: {server_url}")
    # Logic to register server URL in config

@mcp.command()
def list():
    """List registered MCP servers."""
    click.echo("Listing MCP servers...")
    # Logic to read from config
