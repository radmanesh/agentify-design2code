"""White agent module - A2A server with optional ADK client wrapper."""

from .agent import start_white_agent

# Optional: ADK client wrapper (requires white agent server running)
try:
    from .adk_client import create_white_agent_tool, call_white_agent_http
    __all__ = ["start_white_agent", "create_white_agent_tool", "call_white_agent_http"]
except ImportError:
    # ADK not installed, only A2A available
    __all__ = ["start_white_agent"]
