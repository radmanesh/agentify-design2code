"""White agent module - Multiple interfaces for HTML generation.

This module provides three interfaces:
1. A2A HTTP Server - Original interface (port 10002)
2. LangChain/LangServe - Web UI and REST API (port 8000)
3. ADK Client Wrapper - Optional wrapper for ADK agents

All interfaces use OpenAI GPT-4o Vision for HTML generation.
"""

# Core A2A server
from .agent import start_white_agent

# LangChain components
try:
    from .langchain_agent import (
        create_white_agent,
        create_simple_chain,
        generate_html_from_screenshot_impl,
        create_html_generation_tool
    )
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# ADK client wrapper (optional, requires white agent server running)
try:
    from .adk_client import create_white_agent_tool, call_white_agent_http
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False

# Build __all__ based on what's available
__all__ = ["start_white_agent"]

if LANGCHAIN_AVAILABLE:
    __all__.extend([
        "create_white_agent",
        "create_simple_chain",
        "generate_html_from_screenshot_impl",
        "create_html_generation_tool"
    ])

if ADK_AVAILABLE:
    __all__.extend(["create_white_agent_tool", "call_white_agent_http"])
