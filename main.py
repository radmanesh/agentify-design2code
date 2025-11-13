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


@app.command()
def langserve():
    """Start the white agent with LangServe web interface (OpenAI GPT-4o Vision)."""
    import uvicorn
    print("\n" + "="*60)
    print("Starting LangServe Web Interface")
    print("="*60)
    print("\n🚀 White Agent with LangChain + OpenAI GPT-4o Vision")
    print("\n🌐 Access points:")
    print("  • Agent Playground:  http://localhost:8000/agent/playground")
    print("  • Simple Playground: http://localhost:8000/simple/playground")
    print("  • API Docs:          http://localhost:8000/docs")
    print("  • Health Check:      http://localhost:8000/health")
    print("\n💡 Tip: Open the playground in your browser to interact with the agent!")
    print("="*60 + "\n")

    uvicorn.run(
        "src.white_agent.langserve_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    app()
