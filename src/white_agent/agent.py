"""White agent implementation - generates HTML from screenshots."""

import uvicorn
import dotenv
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message
from litellm import completion
import os

dotenv.load_dotenv()


# Debug: verify API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")


def prepare_white_agent_card(url):
    """
    Prepare the agent card for the white agent.

    Args:
        url: The URL where the agent is hosted

    Returns:
        AgentCard instance
    """
    skill = AgentSkill(
        id="html_generation",
        name="HTML Generation from Screenshots",
        description="Generates HTML code from screenshot images",
        tags=["html", "generation", "design2code"],
        examples=[],
    )
    card = AgentCard(
        name="html_generation_agent",
        description="Agent that generates HTML code from screenshots using vision models",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


class HtmlGenerationWhiteAgentExecutor(AgentExecutor):
    """Agent executor for HTML generation from screenshots."""

    def __init__(self):
        # Store conversation history per context
        self.ctx_id_to_messages = {}
        # System prompt for HTML generation
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Create the system prompt for HTML generation from screenshots."""
        prompt = ""
        prompt += "You are an expert web developer who specializes in HTML and CSS.\n"
        prompt += "A user will provide you with a screenshot of a webpage.\n"
        prompt += "You need to return a single html file that uses HTML and CSS to reproduce the given website.\n"
        prompt += "Include all CSS code in the HTML file itself.\n"
        prompt += "If it involves any images, use \"rick.jpg\" as the placeholder.\n"
        prompt += "Some images on the webpage are replaced with a blue rectangle as the placeholder, use \"rick.jpg\" for those as well.\n"
        prompt += "Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions.\n"
        prompt += "Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.\n"
        prompt += "Respond with the content of the HTML+CSS file:\n"
        return prompt

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute HTML generation task.

        Args:
            context: Request context with user input
            event_queue: Queue for sending events back
        """
        # Get user input
        user_input = context.get_user_input()

        # Initialize or retrieve message history for this context
        if context.context_id not in self.ctx_id_to_messages:
            # Initialize with system prompt
            self.ctx_id_to_messages[context.context_id] = [
                {"role": "system", "content": self.system_prompt}
            ]
        messages = self.ctx_id_to_messages[context.context_id]

        # Check if the input contains a base64-encoded screenshot
        if "<screenshot_base64>" in user_input and "</screenshot_base64>" in user_input:
            # Extract screenshot and task description
            import re
            screenshot_match = re.search(
                r"<screenshot_base64>(.*?)</screenshot_base64>",
                user_input,
                re.DOTALL
            )
            screenshot_base64 = screenshot_match.group(1).strip() if screenshot_match else None

            # Remove screenshot from text and add as separate content
            text_without_screenshot = re.sub(
                r"<screenshot_base64>.*?</screenshot_base64>",
                "[Screenshot provided as image]",
                user_input,
                flags=re.DOTALL
            )

            # Create message with vision content
            if screenshot_base64:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text_without_screenshot
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot_base64}"
                            }
                        }
                    ]
                })
            else:
                messages.append({
                    "role": "user",
                    "content": user_input,
                })
        else:
            # Regular text message
            messages.append({
                "role": "user",
                "content": user_input,
            })

        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Call LLM with vision capability
        print(f"White agent: Calling LLM with {len(messages)} messages...")
        response = completion(
            messages=messages,
            model="openai/gpt-4o",  # GPT-4o supports vision
            temperature=0.0,
            api_key=api_key,
        )

        # Extract response
        next_message = response.choices[0].message.model_dump()  # type: ignore
        assistant_content = next_message["content"]

        # Add to message history
        messages.append({
            "role": "assistant",
            "content": assistant_content,
        })

        print(f"White agent: Generated response ({len(assistant_content)} chars)")

        # Send response back
        await event_queue.enqueue_event(
            new_agent_text_message(
                assistant_content, context_id=context.context_id
            )
        )

    async def cancel(self, context, event_queue) -> None:
        """Cancel the current execution (not implemented)."""
        raise NotImplementedError


def start_white_agent(agent_name="html_generation_white_agent", host="localhost", port=9002):
    """
    Start the white agent HTTP server.

    Args:
        agent_name: Name of the agent
        host: Host to bind to
        port: Port to bind to
    """
    print("Starting white agent...")
    url = f"http://{host}:{port}"
    card = prepare_white_agent_card(url)

    request_handler = DefaultRequestHandler(
        agent_executor=HtmlGenerationWhiteAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)
