"""Example: Using LangChain white agent for HTML generation.

This example demonstrates how to use the LangChain-based white agent
with OpenAI GPT-4o Vision for generating HTML from screenshots.

IMPORTANT: Set OPENAI_API_KEY in your .env file before running.
"""

import asyncio
import base64
from pathlib import Path


async def example_direct_function():
    """
    Example 1: Direct function call for HTML generation.

    This is the simplest way to generate HTML - just call the function directly.
    No agent orchestration, just pure HTML generation.
    """
    print("=" * 70)
    print("Example 1: Direct Function Call")
    print("=" * 70)

    from src.white_agent.langchain_agent import generate_html_from_screenshot_impl

    # Create a minimal test screenshot (1x1 pixel PNG)
    test_screenshot = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    )

    print("\n📸 Generating HTML from test screenshot...")
    print(f"Screenshot size: {len(test_screenshot)} bytes (base64)")

    try:
        html_code = generate_html_from_screenshot_impl(
            screenshot_base64=test_screenshot,
            description="Generate a simple HTML page"
        )

        print(f"\n✅ Success!")
        print(f"Generated HTML length: {len(html_code)} characters")
        print(f"\nFirst 200 characters:")
        print(html_code[:200])
        if len(html_code) > 200:
            print("...")

    except Exception as e:
        print(f"\n❌ Error: {e}")


async def example_agent_executor():
    """
    Example 2: Using LangChain Agent Executor.

    This uses the full agent with tool calling and reasoning capabilities.
    The agent can understand natural language and decide when to use tools.
    """
    print("\n" + "=" * 70)
    print("Example 2: LangChain Agent Executor")
    print("=" * 70)

    from src.white_agent.langchain_agent import create_white_agent

    # Create the agent
    print("\n🤖 Creating LangChain agent...")
    agent_executor = create_white_agent()
    print("✅ Agent created!")

    # Test screenshot
    test_screenshot = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    )

    # Natural language input - the agent will understand and use the tool
    user_input = f"""I have a screenshot of a webpage (base64: {test_screenshot[:50]}...).
Can you generate HTML code for it? Make it a simple landing page."""

    print("\n💬 Sending request to agent...")
    print(f"Input: {user_input[:100]}...")

    try:
        # Invoke the agent
        result = agent_executor.invoke({"input": user_input})

        print(f"\n✅ Agent response:")
        print(result["output"])

    except Exception as e:
        print(f"\n❌ Error: {e}")


async def example_with_real_screenshot():
    """
    Example 3: Processing a real screenshot file.

    This loads an actual screenshot from the data directory and generates HTML.
    """
    print("\n" + "=" * 70)
    print("Example 3: Real Screenshot Processing")
    print("=" * 70)

    from src.white_agent.langchain_agent import generate_html_from_screenshot_impl

    # Look for PNG files in the data directory
    data_folder = Path("data")
    if not data_folder.exists():
        print(f"\n⚠️  Data folder not found: {data_folder}")
        print("Skipping real screenshot example.")
        return

    screenshot_files = list(data_folder.glob("*.png"))
    if not screenshot_files:
        print(f"\n⚠️  No PNG screenshots found in {data_folder}")
        print("Skipping real screenshot example.")
        return

    # Use the first screenshot
    screenshot_path = screenshot_files[0]
    print(f"\n📸 Loading screenshot: {screenshot_path}")

    # Read and encode
    with open(screenshot_path, "rb") as f:
        screenshot_bytes = f.read()
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")

    print(f"   Size: {len(screenshot_bytes):,} bytes")
    print(f"   Base64 size: {len(screenshot_base64):,} characters")

    # Generate HTML
    print("\n🔄 Generating HTML...")

    try:
        html_code = generate_html_from_screenshot_impl(
            screenshot_base64=screenshot_base64,
            description=f"Generate HTML that recreates the design from {screenshot_path.name}"
        )

        # Save output
        output_path = Path("generated_langchain.html")
        with open(output_path, "w") as f:
            f.write(html_code)

        print(f"\n✅ Success!")
        print(f"   Generated: {len(html_code):,} characters")
        print(f"   Saved to: {output_path}")
        print(f"\n💡 Open {output_path} in your browser to see the result!")

    except Exception as e:
        print(f"\n❌ Error: {e}")


async def example_simple_chain():
    """
    Example 4: Using the simple chain (no agent).

    This bypasses agent orchestration for direct HTML generation.
    Faster and simpler when you don't need agent capabilities.
    """
    print("\n" + "=" * 70)
    print("Example 4: Simple Chain (No Agent)")
    print("=" * 70)

    from src.white_agent.langchain_agent import create_simple_chain

    # Create simple chain
    print("\n⚡ Creating simple chain...")
    chain = create_simple_chain()
    print("✅ Chain created!")

    # Test screenshot
    test_screenshot = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    )

    print("\n🔄 Running chain...")

    try:
        result = chain({
            "screenshot_base64": test_screenshot,
            "description": "Generate a minimal HTML page"
        })

        html_code = result["output"]

        print(f"\n✅ Success!")
        print(f"Generated HTML length: {len(html_code)} characters")
        print(f"\nFirst 200 characters:")
        print(html_code[:200])

    except Exception as e:
        print(f"\n❌ Error: {e}")


async def example_api_usage():
    """
    Example 5: Using LangServe API endpoints.

    Shows how to call the web API programmatically.
    Server must be running: uv run python main.py langserve
    """
    print("\n" + "=" * 70)
    print("Example 5: LangServe API Usage")
    print("=" * 70)

    print("\n⚠️  This example requires the LangServe server to be running!")
    print("   Start it with: uv run python main.py langserve")
    print("\n📡 Example API calls:\n")

    # Show example curl commands
    print("# Agent endpoint:")
    print('curl -X POST "http://localhost:8000/agent/invoke" \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"input": {"input": "Generate HTML from screenshot: <base64>"}}\'\n')

    print("# Simple endpoint:")
    print('curl -X POST "http://localhost:8000/simple/invoke" \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"input": {"screenshot_base64": "<base64>", "description": "Generate HTML"}}\'\n')

    print("# Health check:")
    print('curl http://localhost:8000/health\n')

    # Try to make actual API call if server is running
    try:
        import httpx

        print("🔍 Checking if server is running...")
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:8000/health")
            if response.status_code == 200:
                print("✅ Server is running!")
                print(f"Response: {response.json()}")
            else:
                print(f"⚠️  Server responded with status {response.status_code}")
    except Exception as e:
        print(f"ℹ️  Server not running or not accessible: {e}")
        print("   Start it with: uv run python main.py langserve")


async def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("LangChain White Agent - Usage Examples")
    print("=" * 70)
    print("\n🚀 These examples show different ways to use the LangChain agent")
    print("   with OpenAI GPT-4o Vision for HTML generation.\n")
    print("📋 Requirements:")
    print("   • OPENAI_API_KEY environment variable (in .env)")
    print("   • LangServe server (for Example 5 only)")
    print("\n" + "=" * 70)

    # Run all examples
    await example_direct_function()
    await example_agent_executor()
    await example_with_real_screenshot()
    await example_simple_chain()
    await example_api_usage()

    print("\n" + "=" * 70)
    print("Examples Completed!")
    print("=" * 70)
    print("\n📚 Key Takeaways:")
    print("   • Direct function: Simplest, no agent needed")
    print("   • Agent executor: Full reasoning and tool use")
    print("   • Simple chain: Fast, no orchestration overhead")
    print("   • LangServe API: Web interface + REST API")
    print("\n💡 Next Steps:")
    print("   • Start LangServe: uv run python main.py langserve")
    print("   • Open playground: http://localhost:8000/agent/playground")
    print("   • Read docs: http://localhost:8000/docs")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

