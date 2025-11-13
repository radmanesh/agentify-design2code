"""ADK client wrapper for the white agent.

This module provides a way for ADK agents to call the white agent's
A2A HTTP server as a tool. The white agent runs as an independent
HTTP server using OpenAI, and this wrapper allows ADK agents (which
may use Gemini) to call it via HTTP.
"""

from google.adk.tools import FunctionTool
from src.my_util import my_a2a
from a2a.types import SendMessageSuccessResponse, Message
from a2a.utils import get_text_parts


async def call_white_agent_http(
    screenshot_base64: str,
    white_agent_url: str = "http://localhost:10002",
    description: str = "Generate HTML from this screenshot"
) -> str:
    """
    Call the white agent's A2A HTTP server to generate HTML.

    This function acts as a client to the white agent's A2A server,
    allowing ADK agents to delegate HTML generation tasks.

    Args:
        screenshot_base64: Base64-encoded PNG screenshot
        white_agent_url: URL of the white agent's A2A HTTP server
        description: Description or instructions for HTML generation

    Returns:
        Generated HTML code as a string
    """
    # Format message for A2A protocol
    message = f"""{description}

<screenshot_base64>
{screenshot_base64}
</screenshot_base64>

Please wrap your HTML code in <html_code>...</html_code> tags.
"""

    # Call white agent via A2A HTTP
    response = await my_a2a.send_message(white_agent_url, message)

    # Parse response
    res_root = response.root
    if not isinstance(res_root, SendMessageSuccessResponse):
        raise ValueError("Expected SendMessageSuccessResponse from white agent")

    res_result = res_root.result
    if not isinstance(res_result, Message):
        raise ValueError("Expected Message result from white agent")

    # Extract text from response
    text_parts = get_text_parts(res_result.parts)
    if not text_parts:
        raise ValueError("No text parts found in white agent response")

    white_text = text_parts[0]

    # Parse the HTML code from response
    if "<html_code>" in white_text and "</html_code>" in white_text:
        start_idx = white_text.find("<html_code>") + len("<html_code>")
        end_idx = white_text.find("</html_code>")
        html_code = white_text[start_idx:end_idx].strip()
    else:
        # If no tags found, assume the entire response is HTML
        html_code = white_text

    return html_code


def create_white_agent_tool(white_agent_url: str = "http://localhost:10002"):
    """
    Create an ADK FunctionTool that calls the white agent via HTTP.

    This tool can be used in ADK agents to delegate HTML generation
    to the white agent server. The white agent must be running at the
    specified URL.

    Args:
        white_agent_url: URL where the white agent server is running

    Returns:
        FunctionTool that can be used in ADK agents

    Example:
        ```python
        from google.adk.agents import Agent
        from src.white_agent import create_white_agent_tool

        # White agent must be running
        white_tool = create_white_agent_tool("http://localhost:10002")

        # Use in an ADK agent
        my_agent = Agent(
            model="gemini-2.0-flash",
            tools=[white_tool]
        )
        ```
    """
    async def wrapper(screenshot_base64: str, description: str = "Generate HTML from this screenshot") -> str:
        """Wrapper function with proper signature for ADK."""
        return await call_white_agent_http(screenshot_base64, white_agent_url, description)

    return FunctionTool(
        name="generate_html_from_screenshot",
        description="Generate HTML code from a screenshot by calling the white agent HTTP server",
        func=wrapper
    )

