"""Example: Basic ADK client wrapper usage for the white agent.

This example shows how to use the white agent from ADK workflows by calling
its A2A HTTP server via the client wrapper.

IMPORTANT: The white agent A2A server must be running before using this wrapper.
Start it with: uv run python main.py white
"""

import asyncio
import base64
from pathlib import Path


async def example_using_function_directly():
    """
    Example of calling the white agent HTTP server directly using the wrapper function.

    This approach is useful for simple integrations where you just need
    to call the white agent without building a full ADK agent.
    """
    print("=" * 60)
    print("Example: Direct Function Call to White Agent Server")
    print("=" * 60)

    # Import the client wrapper
    from src.white_agent import call_white_agent_http

    # Prepare a test screenshot
    # For demo, using a minimal 1x1 pixel PNG
    test_screenshot = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    )

    print("\nCalling white agent HTTP server...")
    print("White agent URL: http://localhost:10002")

    try:
        # Call the white agent via HTTP
        html_code = await call_white_agent_http(
            screenshot_base64=test_screenshot,
            white_agent_url="http://localhost:10002",
            description="Generate a simple HTML page from this screenshot"
        )

        print(f"\n✓ HTML generated successfully!")
        print(f"  Length: {len(html_code)} characters")
        print(f"\nFirst 300 characters:")
        print(html_code[:300])
        if len(html_code) > 300:
            print("...")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure the white agent server is running:")
        print("  uv run python main.py white")


async def example_using_adk_tool():
    """
    Example of creating an ADK FunctionTool from the white agent wrapper.

    This creates a tool that can be used in ADK agents.
    """
    print("\n" + "=" * 60)
    print("Example: Creating ADK FunctionTool")
    print("=" * 60)

    from src.white_agent import create_white_agent_tool

    # Create the tool
    white_tool = create_white_agent_tool(white_agent_url="http://localhost:10002")

    print(f"\n✓ ADK FunctionTool created:")
    print(f"  Name: {white_tool.name}")
    print(f"  Description: {white_tool.description}")

    print("\nThis tool can now be used in ADK agents:")
    print("  - Add it to an Agent's tools list")
    print("  - The agent can call it to generate HTML from screenshots")
    print("  - Communication happens via HTTP to the white agent server")


async def example_with_real_screenshot():
    """
    Example of using the white agent with a real screenshot file.
    """
    print("\n" + "=" * 60)
    print("Example: Processing Real Screenshot")
    print("=" * 60)

    # Look for PNG files in the data directory
    data_folder = Path("data")
    if not data_folder.exists():
        print(f"\n⚠ Data folder not found: {data_folder}")
        print("Skipping real screenshot example.")
        return

    screenshot_files = list(data_folder.glob("*.png"))
    if not screenshot_files:
        print(f"\n⚠ No PNG screenshots found in {data_folder}")
        print("Skipping real screenshot example.")
        return

    # Use the first screenshot
    screenshot_path = screenshot_files[0]
    print(f"\nLoading screenshot: {screenshot_path}")

    # Read and encode the screenshot
    with open(screenshot_path, "rb") as f:
        screenshot_bytes = f.read()
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")

    print(f"  Size: {len(screenshot_bytes):,} bytes")
    print(f"  Base64 size: {len(screenshot_base64):,} characters")

    # Import the wrapper
    from src.white_agent import call_white_agent_http

    print("\nCalling white agent HTTP server...")

    try:
        html_code = await call_white_agent_http(
            screenshot_base64=screenshot_base64,
            white_agent_url="http://localhost:10002",
            description=f"Generate HTML that recreates the design from {screenshot_path.name}"
        )

        # Save the output
        output_path = Path("generated_from_adk_client.html")
        with open(output_path, "w") as f:
            f.write(html_code)

        print(f"\n✓ HTML generated successfully!")
        print(f"  Output: {output_path}")
        print(f"  Length: {len(html_code):,} characters")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure the white agent server is running:")
        print("  uv run python main.py white")


async def main():
    """Run all examples."""
    print("\nWhite Agent - ADK Client Wrapper Examples")
    print("=" * 60)
    print("\nThese examples show how to call the white agent's A2A HTTP")
    print("server from ADK workflows using the client wrapper.")
    print("\n⚠️  PREREQUISITE: White agent server must be running!")
    print("   Start it with: uv run python main.py white")
    print()

    await example_using_function_directly()
    await example_using_adk_tool()
    await example_with_real_screenshot()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nKey Points:")
    print("  • White agent runs as A2A HTTP server (OpenAI)")
    print("  • ADK wrapper calls it via HTTP")
    print("  • No Google API key needed for white agent")
    print("  • Works with any ADK agent that needs HTML generation")


if __name__ == "__main__":
    asyncio.run(main())
