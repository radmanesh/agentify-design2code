"""Green agent implementation - manages Design2Code assessment and evaluation."""

import uvicorn
import tomllib
import dotenv
import json
import time
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, SendMessageSuccessResponse, Message
from a2a.utils import new_agent_text_message, get_text_parts
from src.my_util import parse_tags, my_a2a
from src.green_agent.design2code_env import Design2CodeEnvironment

dotenv.load_dotenv()


def load_agent_card_toml(agent_name):
    """Load agent card configuration from TOML file."""
    current_dir = __file__.rsplit("/", 1)[0]
    with open(f"{current_dir}/{agent_name}.toml", "rb") as f:
        return tomllib.load(f)


async def ask_agent_to_generate_html(white_agent_url, env, task_id):
    """
    Ask the white agent to generate HTML from a screenshot.

    Args:
        white_agent_url: URL of the white agent to test
        env: Design2Code environment instance
        task_id: ID of the task to evaluate

    Returns:
        Dictionary with evaluation results
    """
    # Reset environment to the specified task
    env_reset_res = env.reset(task_id=task_id)
    obs = env_reset_res.observation
    info = env_reset_res.info

    # Get the full screenshot data for the agent
    current_task = env.current_task
    if current_task is None or current_task.screenshot_base64 is None:
        raise ValueError(f"Task {task_id} not properly loaded")

    # Prepare task description with wiki and screenshot
    task_description = f"""
{env.get_wiki()}

Here is the screenshot (base64-encoded PNG):
<screenshot_base64>
{current_task.screenshot_base64}
</screenshot_base64>

Please analyze this screenshot and generate the corresponding HTML code.
Wrap your HTML code in <html_code>...</html_code> tags.
"""

    print(f"@@@ Green agent: Sending task {task_id} to white agent...")
    print(f"Screenshot size: {len(current_task.screenshot_base64)} bytes (base64)")

    # Send message to white agent
    white_agent_response = await my_a2a.send_message(
        white_agent_url, task_description
    )

    # Parse response
    res_root = white_agent_response.root
    assert isinstance(res_root, SendMessageSuccessResponse), (
        "Expected SendMessageSuccessResponse from white agent"
    )
    res_result = res_root.result
    assert isinstance(res_result, Message), (
        "Expected Message result from white agent"
    )

    # Extract text from response
    text_parts = get_text_parts(res_result.parts)
    assert len(text_parts) >= 1, (
        "Expecting at least one text part from the white agent"
    )
    white_text = text_parts[0]
    print(f"@@@ White agent response received (length: {len(white_text)} chars)")

    # Parse the HTML code from response
    white_tags = parse_tags(white_text)
    if "html_code" not in white_tags:
        print("Warning: No <html_code> tag found in response, using full text")
        html_code = white_text
    else:
        html_code = white_tags["html_code"]

    print(f"Generated HTML length: {len(html_code)} chars")

    # Evaluate the generated HTML
    env_response = env.step(html_code)
    reward = env_response.reward
    info = {**info, **env_response.info}

    return {
        "reward": reward,
        "info": info,
        "generated_html": html_code,
    }


class Design2CodeGreenAgentExecutor(AgentExecutor):
    """Agent executor for Design2Code assessment."""

    def __init__(self):
        pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute Design2Code assessment workflow.

        Args:
            context: Request context with user input
            event_queue: Queue for sending events back to user
        """
        # Parse the task
        print("Green agent: Received a task, parsing...")
        user_input = context.get_user_input()
        tags = parse_tags(user_input)
        white_agent_url = tags["white_agent_url"]
        env_config_str = tags["env_config"]
        env_config = json.loads(env_config_str)

        # Set up the Design2Code environment
        print("Green agent: Setting up Design2Code environment...")
        data_folder = env_config["data_folder"]
        task_ids = env_config["task_ids"]

        env = Design2CodeEnvironment(
            data_folder=data_folder,
            task_ids=task_ids,
        )

        print(f"Green agent: Loaded {len(env.tasks)} tasks: {list(env.tasks.keys())}")

        # Run evaluation for each task
        print("Green agent: Starting evaluation...")
        timestamp_started = time.time()

        results = []
        for task_id in task_ids:
            if task_id not in env.tasks:
                print(f"Warning: Task {task_id} not found, skipping...")
                continue

            print(f"\n=== Evaluating Task {task_id} ===")
            task_result = await ask_agent_to_generate_html(
                white_agent_url, env, task_id
            )
            results.append(task_result)

            reward = task_result["reward"]
            result_emoji = "✅" if reward > 0.5 else "⚠️" if reward > 0.0 else "❌"
            print(f"Task {task_id} result: {result_emoji} (reward: {reward:.2f})")

        # Calculate overall metrics
        elapsed_time = time.time() - timestamp_started
        avg_reward = sum(r["reward"] for r in results) / len(results) if results else 0.0
        success_count = sum(1 for r in results if r["reward"] > 0.5)

        metrics = {
            "time_used": elapsed_time,
            "average_reward": avg_reward,
            "success_count": success_count,
            "total_tasks": len(results),
            "success_rate": success_count / len(results) if results else 0.0,
        }

        print("\n=== Evaluation Complete ===")
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Success rate: {metrics['success_rate']:.1%}")

        # Send results back
        result_message = f"""Design2Code Evaluation Complete!

Summary:
- Tasks evaluated: {metrics['total_tasks']}
- Success count: {success_count}/{metrics['total_tasks']}
- Success rate: {metrics['success_rate']:.1%}
- Average reward: {avg_reward:.2f}
- Time used: {elapsed_time:.2f}s

Task Results:
"""
        for i, (task_id, result) in enumerate(zip(task_ids, results), 1):
            reward = result["reward"]
            emoji = "✅" if reward > 0.5 else "⚠️" if reward > 0.0 else "❌"
            result_message += f"{i}. Task {task_id}: {emoji} (reward: {reward:.2f})\n"

        await event_queue.enqueue_event(
            new_agent_text_message(result_message)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the current execution (not implemented)."""
        raise NotImplementedError


def start_green_agent(agent_name="design2code_green_agent", host="localhost", port=10001):
    """
    Start the green agent HTTP server.

    Args:
        agent_name: Name of the agent (used to load TOML config)
        host: Host to bind to
        port: Port to bind to
    """
    print("Starting green agent...")
    agent_card_dict = load_agent_card_toml(agent_name)
    url = f"http://{host}:{port}"
    agent_card_dict["url"] = url  # Complete all required card fields

    request_handler = DefaultRequestHandler(
        agent_executor=Design2CodeGreenAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)
