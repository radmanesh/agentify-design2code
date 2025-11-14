"""CLI entry point for agentify-design2code."""

import os
import typer
import asyncio

from src.green_agent import start_green_agent
from src.white_agent import start_white_agent
from src.launcher import launch_evaluation

# Main Typer app for CLI interface
app = typer.Typer(help="Agentified Design2Code - HTML generation assessment framework")


@app.command()
def green(
    host: str = typer.Option(None, "--host", help="Green agent host (default: localhost or GREEN_AGENT_HOST env)"),
    port: int = typer.Option(None, "--port", "-p", help="Green agent port (default: 10001 or GREEN_AGENT_PORT env)"),
):
    """
    Start the green agent (assessment manager for Design2Code tasks).

    Address can be configured via:
    1. Command-line arguments (highest priority)
    2. Environment variables (GREEN_AGENT_HOST, GREEN_AGENT_PORT)
    3. Default values (localhost:10001)
    """
    # Get address (CLI args > env vars > defaults)
    host = host or os.getenv("GREEN_AGENT_HOST", "localhost")
    port = port or int(os.getenv("GREEN_AGENT_PORT", "10001"))

    print(f"Starting green agent at http://{host}:{port}...")
    start_green_agent(host=host, port=port)


@app.command()
def white(
    host: str = typer.Option(None, "--host", help="White agent host (default: localhost or WHITE_AGENT_HOST env)"),
    port: int = typer.Option(None, "--port", "-p", help="White agent port (default: 10002 or WHITE_AGENT_PORT env)"),
):
    """
    Start the white agent (target being tested for HTML generation).

    Address can be configured via:
    1. Command-line arguments (highest priority)
    2. Environment variables (WHITE_AGENT_HOST, WHITE_AGENT_PORT)
    3. Default values (localhost:10002)
    """
    # Get address (CLI args > env vars > defaults)
    host = host or os.getenv("WHITE_AGENT_HOST", "localhost")
    port = port or int(os.getenv("WHITE_AGENT_PORT", "10002"))

    print(f"Starting white agent at http://{host}:{port}...")
    start_white_agent(host=host, port=port)


@app.command()
def launch(
    green_host: str = typer.Option(None, "--green-host", "-gh", help="Green agent host (default: localhost or GREEN_AGENT_HOST env)"),
    green_port: int = typer.Option(None, "--green-port", "-gp", help="Green agent port (default: 10001 or GREEN_AGENT_PORT env)"),
    white_host: str = typer.Option(None, "--white-host", "-wh", help="White agent host (default: localhost or WHITE_AGENT_HOST env)"),
    white_port: int = typer.Option(None, "--white-port", "-wp", help="White agent port (default: 10002 or WHITE_AGENT_PORT env)"),
):
    """
    Launch the complete Design2Code evaluation workflow.

    Agent addresses can be configured via:
    1. Command-line arguments (highest priority)
    2. Environment variables (GREEN_AGENT_HOST, GREEN_AGENT_PORT, etc.)
    3. Default values (localhost:10001, localhost:10002)
    """
    asyncio.run(launch_evaluation(
        green_host=green_host,
        green_port=green_port,
        white_host=white_host,
        white_port=white_port
    ))


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
