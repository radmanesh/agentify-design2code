"""Example: Using the white agent as a tool in ADK multi-agent systems.

This example demonstrates how to use the white agent client wrapper as a tool
in ADK agents, enabling complex multi-agent workflows where different agents
specialize in different tasks.

IMPORTANT: The white agent A2A server must be running before using this wrapper.
Start it with: uv run python main.py white
"""

import asyncio
from google.adk.agents import Agent
from src.white_agent import create_white_agent_tool


def example_create_tool():
    """
    Example of creating a white agent tool for use in ADK agents.

    This tool wraps the white agent's HTTP server, allowing ADK agents
    to delegate HTML generation tasks to it.
    """
    print("=" * 60)
    print("Example: Creating White Agent Tool")
    print("=" * 60)

    # Create the tool
    white_tool = create_white_agent_tool(white_agent_url="http://localhost:10002")

    print(f"\n✓ White agent tool created:")
    print(f"  Name: {white_tool.name}")
    print(f"  Description: {white_tool.description}")

    print("\nHow it works:")
    print("  1. Tool is called with screenshot and description")
    print("  2. Wrapper sends HTTP request to white agent server")
    print("  3. White agent (OpenAI) generates HTML")
    print("  4. Tool returns HTML to the calling agent")

    return white_tool


def example_coordinator_agent():
    """
    Example of creating a coordinator agent that uses the white agent as a tool.

    This demonstrates a multi-agent architecture where a coordinator agent
    delegates HTML generation to the specialized white agent.
    """
    print("\n" + "=" * 60)
    print("Example: Coordinator Agent with White Agent Tool")
    print("=" * 60)

    # Create the white agent tool
    white_tool = create_white_agent_tool()

    # Create a coordinator agent that uses Gemini for orchestration
    # but delegates HTML generation to the white agent (OpenAI)
    coordinator_agent = Agent(
        model="gemini-2.0-flash",  # Gemini for orchestration
        name="design_coordinator",
        description="Coordinates design-to-code workflows using specialized agents",
        instruction="""You are a design coordinator that manages HTML generation tasks.
When asked to generate HTML from a design or screenshot, use the HTML generation tool.
Provide clear feedback about the generated HTML and suggest improvements if needed.""",
        tools=[white_tool],  # Delegates to white agent via HTTP
    )

    print(f"\n✓ Coordinator agent created:")
    print(f"  Name: {coordinator_agent.name}")
    print(f"  Description: {coordinator_agent.description}")
    print(f"  Model: {coordinator_agent.model} (for orchestration)")
    print(f"  Tools:")
    for tool in coordinator_agent.tools:
        print(f"    - {tool.name}")

    print("\nArchitecture:")
    print("  ┌─────────────────────────────────────┐")
    print("  │   Coordinator Agent (Gemini)       │")
    print("  │   - Orchestrates workflow          │")
    print("  │   - Makes decisions                │")
    print("  └──────────┬──────────────────────────┘")
    print("             │ HTTP")
    print("             ▼")
    print("  ┌─────────────────────────────────────┐")
    print("  │   White Agent Server (OpenAI)      │")
    print("  │   - Generates HTML from screenshots │")
    print("  │   - Specialized vision + code task │")
    print("  └─────────────────────────────────────┘")

    return coordinator_agent


def example_complex_workflow():
    """
    Example of a complex multi-agent workflow with multiple specialized agents.

    This shows how you might combine multiple specialized agents,
    including the white agent, to handle complex tasks.
    """
    print("\n" + "=" * 60)
    print("Example: Complex Multi-Agent Workflow")
    print("=" * 60)

    # White agent for HTML generation (via HTTP)
    white_tool = create_white_agent_tool()

    # In a real system, you might have more specialized tools:
    # - Code review tool
    # - Accessibility checker tool
    # - Performance optimizer tool
    # - etc.

    # Main orchestrator coordinates everything
    orchestrator = Agent(
        model="gemini-2.0-flash",
        name="design2code_orchestrator",
        description="Orchestrates the complete design-to-code workflow",
        instruction="""You orchestrate the design-to-code workflow.

Your responsibilities:
1. Analyze design requirements
2. Delegate HTML generation to the HTML generator tool
3. Review the generated code
4. Provide comprehensive feedback
5. Suggest improvements

When generating HTML, always use the available HTML generation tool.""",
        tools=[white_tool],
    )

    print(f"\n✓ Orchestrator created:")
    print(f"  Name: {orchestrator.name}")
    print(f"  Model: {orchestrator.model}")
    print(f"  Available tools: {[t.name for t in orchestrator.tools]}")

    print("\nWorkflow Benefits:")
    print("  ✓ Task delegation to specialized agents")
    print("  ✓ Clear separation of concerns")
    print("  ✓ Scalable multi-agent systems")
    print("  ✓ Reusable agent components")
    print("  ✓ Mix different LLMs (Gemini + OpenAI)")

    return orchestrator


def example_api_key_requirements():
    """
    Example explaining the API key requirements for this architecture.
    """
    print("\n" + "=" * 60)
    print("Example: API Key Requirements")
    print("=" * 60)

    print("""
This hybrid architecture requires different API keys for different components:

1. **White Agent Server** (A2A HTTP Server)
   - Uses: OpenAI GPT-4o via LiteLLM
   - Requires: OPENAI_API_KEY environment variable
   - Purpose: Vision model for HTML generation
   - Location: Runs independently as HTTP server

2. **ADK Agents** (Coordinator/Orchestrator)
   - Uses: Google Gemini models
   - Requires: GOOGLE_API_KEY environment variable
   - Purpose: Agent orchestration and decision-making
   - Location: Your ADK agent code

Architecture Benefits:
✓ Best tool for each job (Gemini for orchestration, OpenAI for vision)
✓ Independent scaling (white agent can be on different server)
✓ Clear separation (white agent doesn't need ADK)
✓ Flexible integration (any HTTP client can call white agent)

Required Environment Variables:
.env file should contain:
  OPENAI_API_KEY=sk-...    # For white agent server
  GOOGLE_API_KEY=...        # For ADK agents (if using them)
""")


def main():
    """Run all examples."""
    print("\nWhite Agent - ADK Tool Integration Examples")
    print("=" * 60)
    print("\nThese examples show how to use the white agent as a tool")
    print("in ADK multi-agent systems via HTTP client wrapper.")
    print("\n⚠️  PREREQUISITE: White agent server must be running!")
    print("   Start it with: uv run python main.py white")
    print()

    white_tool = example_create_tool()
    coordinator = example_coordinator_agent()
    orchestrator = example_complex_workflow()
    example_api_key_requirements()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  • White agent = A2A HTTP server (OpenAI)")
    print("  • ADK wrapper = HTTP client that calls white agent")
    print("  • ADK agents = Use Gemini for orchestration")
    print("  • Delegation = ADK agents call white agent via HTTP")
    print("  • Benefits = Mix best models for each task")


if __name__ == "__main__":
    main()
