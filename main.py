"""CLI entry point for agentify-design2code."""

import typer
import asyncio

from src.green_agent import start_green_agent
from src.white_agent import start_white_agent
from src.launcher import launch_evaluation

# Main Typer app for CLI interface
app = typer.Typer(help="Agentified Design2Code - HTML generation assessment framework")


@app.command()
def green():
    """Start the green agent (assessment manager for Design2Code tasks)."""
    start_green_agent()


@app.command()
def white():
    """Start the white agent (target being tested for HTML generation)."""
    start_white_agent()


@app.command()
def launch():
    """Launch the complete Design2Code evaluation workflow."""
    asyncio.run(launch_evaluation())


if __name__ == "__main__":
    app()
