"""Example: A2A compatibility of the white agent.

This example demonstrates that the white agent, despite being a native ADK agent,
is fully compatible with the A2A protocol through ADK's to_a2a() function.
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
    print("(Make sure the white agent server is running: uv run python main.py white)")

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
2. White agent (now ADK-native) receives via to_a2a() wrapper
3. ADK automatically handles protocol conversion
4. White agent generates HTML using vision model
5. Response is sent back via A2A protocol
6. Green agent receives and evaluates the HTML

Key points:
✓ No changes needed to green agent code
✓ No changes needed to launcher.py
✓ Existing evaluation workflow works as-is
✓ A2A protocol is automatically handled by ADK

This is the power of ADK's to_a2a() function - it provides
A2A compatibility automatically for native ADK agents!
""")


async def example_dual_protocol_support():
    """
    Example explaining the dual protocol support.
    """
    print("\n" + "=" * 60)
    print("Example: Dual Protocol Support")
    print("=" * 60)

    print("""
The white agent now supports BOTH protocols simultaneously:

┌─────────────────────────────────────────────────────┐
│                  White Agent                        │
│            (Native ADK Agent)                       │
│                                                     │
│  ┌────────────────────────────────────────────┐   │
│  │    HTML Generation Logic                   │   │
│  │    (LiteLLM + Vision Model)               │   │
│  └────────────────────────────────────────────┘   │
│                      ↑                              │
│         ┌────────────┴────────────┐               │
│         │                         │                │
│    ┌────┴────┐              ┌────┴────┐          │
│    │   ADK   │              │   A2A   │          │
│    │Interface│              │Interface│          │
│    │         │              │(via     │          │
│    │         │              │to_a2a() )│          │
│    └────┬────┘              └────┬────┘          │
└─────────┼──────────────────────────┼───────────────┘
          │                          │
          │                          │
     ┌────▼────┐              ┌──────▼─────┐
     │   ADK   │              │    A2A     │
     │ Agents  │              │  Clients   │
     │(AgentTool)│            │ (Green Agent)│
     └─────────┘              └────────────┘

Benefits:
• Single implementation serves both protocols
• No code duplication
• Easier to maintain
• More flexible integration options
• Future-proof architecture
""")


async def main():
    """Run all examples."""
    print("\nWhite Agent - A2A Compatibility Examples")
    print("=" * 60)
    print("\nThese examples demonstrate that the native ADK white agent")
    print("maintains full A2A compatibility through ADK's to_a2a() function.")
    print()

    await example_a2a_client_access()
    await example_green_agent_compatibility()
    await example_dual_protocol_support()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  • Native ADK agent with automatic A2A support")
    print("  • Green agent works without modifications")
    print("  • Dual protocol support (ADK + A2A)")
    print("  • Single implementation, multiple interfaces")
    print("  • Best of both worlds!")


if __name__ == "__main__":
    asyncio.run(main())

