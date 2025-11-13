"""LangChain agent implementation for HTML generation from screenshots.

This module provides a LangChain-based agent using OpenAI GPT-4o Vision
for generating HTML code from screenshot images.
"""

import os
import dotenv
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from pydantic import BaseModel, Field

# Load environment variables
dotenv.load_dotenv()


class HTMLGenerationInput(BaseModel):
    """Input schema for HTML generation tool."""
    screenshot_base64: str = Field(description="Base64-encoded PNG screenshot image")
    description: str = Field(
        default="Generate HTML from this screenshot",
        description="Description or instructions for HTML generation"
    )


def create_system_prompt() -> str:
    """Create the system prompt for HTML generation."""
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


def generate_html_from_screenshot_impl(screenshot_base64: str, description: str = "Generate HTML from this screenshot") -> str:
    """
    Generate HTML from a base64-encoded screenshot using GPT-4o Vision.

    Args:
        screenshot_base64: Base64-encoded PNG screenshot
        description: Task description for HTML generation

    Returns:
        Generated HTML code as a string
    """
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Create ChatOpenAI with vision capability
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.0,
        api_key=api_key
    )

    # Create system prompt
    system_prompt = create_system_prompt()

    # Create messages with vision content
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=[
                {"type": "text", "text": description},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{screenshot_base64}"
                    }
                }
            ]
        )
    ]

    # Generate HTML
    print(f"LangChain agent: Generating HTML from screenshot...")
    response = llm.invoke(messages)
    html_code = response.content

    print(f"LangChain agent: Generated HTML ({len(html_code)} chars)")

    return html_code


def create_html_generation_tool() -> StructuredTool:
    """
    Create a LangChain StructuredTool for HTML generation.

    Returns:
        StructuredTool that can be used in LangChain agents
    """
    return StructuredTool.from_function(
        func=generate_html_from_screenshot_impl,
        name="generate_html_from_screenshot",
        description="Generate HTML code from a base64-encoded PNG screenshot using GPT-4o Vision model. Accepts a screenshot and optional description.",
        args_schema=HTMLGenerationInput,
        return_direct=False
    )


def create_white_agent():
    """
    Create a LangChain runnable for HTML generation.

    This creates a simple chain that processes user input and generates HTML.
    It uses OpenAI GPT-4o for understanding the request and GPT-4o Vision for HTML generation.

    Returns:
        A runnable chain configured for HTML generation tasks
    """
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Create LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.0,
        api_key=api_key
    )

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert HTML generation assistant. You help users convert screenshots into HTML code.

When the user provides input (which may contain base64-encoded screenshot data), analyze it and generate HTML code.

Always:
1. Generate clean, well-structured HTML
2. Include CSS inline in the HTML
3. Use placeholder images (rick.jpg) where needed
4. Explain what you're doing
5. Be professional and thorough

Respond with the generated HTML code and helpful feedback."""),
        ("human", "{input}")
    ])

    # Create chain
    chain = prompt | llm

    return chain


def create_simple_chain():
    """
    Create a simple LangChain chain (no agent) for direct HTML generation.

    This is useful when you want direct HTML generation without agent orchestration.

    Returns:
        A simple runnable chain that takes screenshot_base64 and description as input
    """
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Create a runnable lambda that wraps the generation function
    def generate_wrapper(inputs: Dict[str, Any]) -> Dict[str, Any]:
        screenshot_base64 = inputs.get("screenshot_base64", "")
        description = inputs.get("description", "Generate HTML from this screenshot")

        html_code = generate_html_from_screenshot_impl(screenshot_base64, description)

        return {"output": html_code}

    # Return a RunnableLambda which is a proper Runnable
    return RunnableLambda(generate_wrapper)

