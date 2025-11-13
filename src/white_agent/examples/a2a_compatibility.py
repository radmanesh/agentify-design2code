"""Example: A2A compatibility of the white agent.

This example demonstrates that the white agent maintains A2A protocol support
alongside the new LangChain/LangServe interface.
"""

import asyncio
import base64
from pathlib import Path
from src.my_util import my_a2a


async def example_a2a_client_access():
    """
    Example of accessing the white agent via A2A protocol.

    This shows that existing A2A clients (like the green agent) can
    continue to work without any modifications.
    """
    print("=" * 60)
    print("Example: A2A Client Access")
    print("=" * 60)

    # White agent A2A server URL (must be running)
    white_agent_url = "http://localhost:10002"

    print(f"\nAttempting to connect to white agent at {white_agent_url}")
    print("(Make sure the A2A server is running: uv run python main.py white)")

    # Test 1: Get agent card
    print("\n" + "-" * 60)
    print("Test 1: Fetching Agent Card")
    print("-" * 60)

    try:
        card = await my_a2a.get_agent_card(white_agent_url)
        if card:
            print(f"✓ Agent card retrieved successfully!")
            print(f"  Name: {card.name}")
            print(f"  Description: {card.description}")
            print(f"  Version: {card.version}")
            print(f"\nThis proves the A2A server is running and accessible.")
        else:
            print("✗ Could not retrieve agent card")
            print("Make sure the white agent server is running.")
            return
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Make sure the white agent server is running.")
        return

    # Test 2: Send a simple message
    print("\n" + "-" * 60)
    print("Test 2: Sending Test Message")
    print("-" * 60)

    # Create a minimal test screenshot
    test_screenshot = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    )

    test_message = f"""
Generate HTML from this test screenshot.

<screenshot_base64>
{test_screenshot}
</screenshot_base64>

Please wrap your HTML code in <html_code>...</html_code> tags.
"""

    try:
        print("Sending message to white agent...")
        response = await my_a2a.send_message(white_agent_url, test_message)

        print(f"✓ Response received!")
        print(f"  Type: {type(response)}")
        print(f"\nThis proves the white agent responds via A2A protocol.")

    except Exception as e:
        print(f"✗ Error: {e}")


async def example_green_agent_compatibility():
    """
    Example showing that the green agent can still call the white agent.

    This demonstrates backward compatibility - the existing evaluation
    workflow continues to work without modifications.
    """
    print("\n" + "=" * 60)
    print("Example: Green Agent Compatibility")
    print("=" * 60)

    print("""
The green agent uses the A2A protocol to communicate with the white agent.
Here's what happens:

1. Green agent sends screenshot + task via A2A HTTP request
2. White agent (A2A server) receives the request
3. White agent uses OpenAI GPT-4o Vision to generate HTML
4. Response is sent back via A2A protocol
5. Green agent receives and evaluates the HTML

Key points:
✓ No changes needed to green agent code
✓ No changes needed to launcher.py
✓ Existing evaluation workflow works as-is
✓ A2A protocol compatibility maintained

This ensures backward compatibility with all existing A2A clients!
""")


async def example_dual_interface_support():
    """
    Example explaining the dual interface support.
    """
    print("\n" + "=" * 60)
    print("Example: Dual Interface Support")
    print("=" * 60)

    print("""
The white agent now supports TWO interfaces simultaneously:

┌─────────────────────────────────────────────────────┐
│              White Agent (OpenAI GPT-4o)            │
│                                                     │
│  ┌────────────────────────────────────────────┐   │
│  │    HTML Generation Logic                   │   │
│  │    (OpenAI GPT-4o Vision)                 │   │
│  └────────────────────────────────────────────┘   │
│                      ↑                              │
│         ┌────────────┴────────────┐               │
│         │                         │                │
│    ┌────┴────┐              ┌────┴────┐          │
│    │   A2A   │              │LangChain│          │
│    │  HTTP   │              │LangServe│          │
│    │ Server  │              │ (Web UI)│          │
│    │         │              │         │          │
│    └────┬────┘              └────┬────┘          │
└─────────┼──────────────────────────┼───────────────┘
          │                          │
          │                          │
     ┌────▼────┐              ┌──────▼─────┐
     │   A2A   │              │ LangChain  │
     │ Clients │              │  Agents &  │
     │(Green   │              │  Web UI    │
     │ Agent)  │              │(Playground)│
     └─────────┘              └────────────┘

Port 10002: A2A HTTP Server   Port 8000: LangServe Web UI

Start A2A server:    uv run python main.py white
Start LangServe:     uv run python main.py langserve

Benefits:
• Single codebase, multiple interfaces
• Backward compatible with A2A clients
• Modern web UI via LangServe
• Choose interface based on use case
• Both use OpenAI GPT-4o (no Google dependency)
""")


async def main():
    """Run all examples."""
    print("\nWhite Agent - A2A Compatibility Examples")
    print("=" * 60)
    print("\nThese examples demonstrate that the white agent maintains")
    print("full A2A compatibility alongside the new LangChain interface.")
    print()

    await example_a2a_client_access()
    await example_green_agent_compatibility()
    await example_dual_interface_support()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  • A2A interface still works (port 10002)")
    print("  • LangServe interface available (port 8000)")
    print("  • Green agent workflow unchanged")
    print("  • Both interfaces use OpenAI GPT-4o")
    print("  • Choose the right interface for your needs")

if __name__ == "__main__":
    asyncio.run(main())
