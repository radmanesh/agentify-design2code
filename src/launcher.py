"""Launcher module - initiates and coordinates the Design2Code evaluation process."""

import multiprocessing
import json
from src.green_agent.agent import start_green_agent
from src.white_agent.agent import start_white_agent
from src.my_util import my_a2a


async def launch_evaluation():
    """Launch the complete Design2Code evaluation workflow with green and white agents."""
    # start green agent
    print("Launching green agent for Design2Code assessment...")
    green_address = ("localhost", 10001)
    green_url = f"http://{green_address[0]}:{green_address[1]}"
    # Create green agent process
    p_green = multiprocessing.Process(
        target=start_green_agent, args=("design2code_green_agent", *green_address)
    )
    p_green.start()
    assert await my_a2a.wait_agent_ready(green_url), "Green agent not ready in time"
    print("Green agent is ready.")

    # start white agent
    print("Launching white agent for HTML generation...")
    white_address = ("localhost", 10002)
    white_url = f"http://{white_address[0]}:{white_address[1]}"
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
    # Clean up agent processes
    p_green.terminate()
    p_green.join()
    p_white.terminate()
    p_white.join()
    print("Agents terminated.")
