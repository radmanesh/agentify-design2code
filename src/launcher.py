"""Launcher module - initiates and coordinates the Design2Code evaluation process."""

import multiprocessing
import json
import os
from typing import Optional
from src.green_agent.agent import start_green_agent
from src.white_agent.agent import start_white_agent
from src.my_util import my_a2a


async def launch_evaluation(
    green_host: Optional[str] = None,
    green_port: Optional[int] = None,
    white_host: Optional[str] = None,
    white_port: Optional[int] = None
):
    """
    Launch the complete Design2Code evaluation workflow with green and white agents.

    Args:
        green_host: Green agent host (default: from GREEN_AGENT_HOST env or "localhost")
        green_port: Green agent port (default: from GREEN_AGENT_PORT env or 10001)
        white_host: White agent host (default: from WHITE_AGENT_HOST env or "localhost")
        white_port: White agent port (default: from WHITE_AGENT_PORT env or 10002)
    """
    # Get green agent address (CLI args > env vars > defaults)
    green_host = green_host or os.getenv("GREEN_AGENT_HOST", "localhost")
    green_port = green_port or int(os.getenv("GREEN_AGENT_PORT", "10001"))
    green_address = (green_host, green_port)
    green_url = f"http://{green_address[0]}:{green_address[1]}"

    # start green agent
    print(f"Launching green agent for Design2Code assessment at {green_url}...")
    # Create green agent process
    p_green = multiprocessing.Process(
        target=start_green_agent, args=("design2code_green_agent", *green_address)
    )
    p_green.start()
    assert await my_a2a.wait_agent_ready(green_url), "Green agent not ready in time"
    print("Green agent is ready.")

    # Get white agent address (CLI args > env vars > defaults)
    white_host = white_host or os.getenv("WHITE_AGENT_HOST", "localhost")
    white_port = white_port or int(os.getenv("WHITE_AGENT_PORT", "10002"))
    white_address = (white_host, white_port)
    white_url = f"http://{white_address[0]}:{white_address[1]}"

    # start white agent
    print(f"Launching white agent for HTML generation at {white_url}...")
    # Create white agent process
    p_white = multiprocessing.Process(
        target=start_white_agent, args=("html_generation_white_agent", *white_address)
    )
    p_white.start()
    assert await my_a2a.wait_agent_ready(white_url), "White agent not ready in time"
    print("White agent is ready.")

    # send the task description for Design2Code evaluation
    print("Sending Design2Code task description to green agent...")
        # Configuration for the assessment task
    task_config = {
        "data_folder": "data",  # Folder containing HTML files and screenshots
        "task_ids": [6, 11],  # HTML file IDs to evaluate (6.html, 11.html, etc.)
    }
    # Format task message with XML-like tags for parsing
    task_text = f"""
Your task is to instantiate Design2Code assessment to test the agent located at:
<white_agent_url>
http://{white_address[0]}:{white_address[1]}/
</white_agent_url>
You should use the following configuration:
<env_config>
{json.dumps(task_config, indent=2)}
</env_config>
    """
    print("Task description:")
    print(task_text)
    print("Sending...")
    # Send task to green agent via A2A protocol
    response = await my_a2a.send_message(green_url, task_text)
    print("Response from green agent:")
    print(response)

    print("Evaluation complete. Terminating agents...")

    # Clean up agent processes with timeout
    # Use kill() instead of terminate() if terminate doesn't work
    p_green.terminate()
    p_white.terminate()

    # Wait with timeout to avoid blocking forever
    p_green.join(timeout=5)
    if p_green.is_alive():
        print("Green agent didn't terminate, forcing kill...")
        p_green.kill()
        p_green.join()

    p_white.join(timeout=5)
    if p_white.is_alive():
        print("White agent didn't terminate, forcing kill...")
        p_white.kill()
        p_white.join()

    print("Agents terminated.")
