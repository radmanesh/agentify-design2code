"""CLI entry point for agentify-design2code."""

import typer
import asyncio
import os
import dotenv

from src.green_agent import start_green_agent
from src.white_agent import start_white_agent
from src.launcher import launch_evaluation

# Debug: Load and check environment variables
print("[DEBUG] Loading .env in main.py...")
dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print(f"[DEBUG] OPENAI_API_KEY in main.py: {'Found' if api_key else 'NOT FOUND'}")
if api_key:
    print(f"[DEBUG] API key starts with: {api_key[:7]}... (length: {len(api_key)})")
else:
    print("[DEBUG] WARNING: OPENAI_API_KEY not found in environment!")

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
