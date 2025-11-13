"""LangServe FastAPI application for the white agent.

This module provides a web server with REST API and interactive playground UI
for the HTML generation agent.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from src.white_agent.langchain_agent import create_white_agent, create_simple_chain

# Create FastAPI application
app = FastAPI(
    title="White Agent - HTML Generation Service",
    version="1.0.0",
    description="""
    # White Agent - HTML Generation from Screenshots

    This service uses OpenAI GPT-4o Vision to generate HTML code from screenshot images.

    ## Features

    - **Vision-based HTML Generation**: Convert screenshots to HTML using GPT-4o Vision
    - **Interactive Playground**: Test the agent in your browser
    - **REST API**: Programmatic access via HTTP
    - **Streaming Responses**: Real-time response streaming
    - **Agent Mode**: Intelligent agent with tool use and reasoning
    - **Simple Mode**: Direct HTML generation without agent orchestration

    ## Endpoints

    - `/agent/*` - Full agent with orchestration
    - `/simple/*` - Direct HTML generation
    - `/agent/playground` - Interactive web UI for the agent
    - `/simple/playground` - Interactive web UI for simple mode

    ## Usage

    ### Agent Mode (Recommended)

    ```python
    import requests

    response = requests.post(
        "http://localhost:8000/agent/invoke",
        json={
            "input": {
                "input": "Generate HTML from this screenshot: <base64_data>"
            }
        }
    )
    html = response.json()["output"]["output"]
    ```

    ### Simple Mode

    ```python
    response = requests.post(
        "http://localhost:8000/simple/invoke",
        json={
            "input": {
                "screenshot_base64": "<base64_data>",
                "description": "Generate a landing page"
            }
        }
    )
    html = response.json()["output"]["output"]
    ```

    ## Authentication

    Set `OPENAI_API_KEY` environment variable before starting the server.
    """,
)

# Add CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint - service information."""
    return {
        "service": "White Agent - HTML Generation",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "agent_api": "/agent/*",
            "agent_playground": "/agent/playground",
            "simple_api": "/simple/*",
            "simple_playground": "/simple/playground",
            "docs": "/docs",
            "openapi": "/openapi.json"
        },
        "description": "OpenAI GPT-4o Vision-based HTML generation from screenshots"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# Create agent and chain instances
print("Initializing LangChain agent...")
agent_executor = create_white_agent()
print("✓ Agent initialized")

print("Initializing simple chain...")
simple_chain = create_simple_chain()
print("✓ Simple chain initialized")


# Add agent routes (full agent with orchestration)
add_routes(
    app,
    agent_executor,
    path="/agent",
    enabled_endpoints=["invoke", "batch", "stream", "stream_log", "playground"],
    playground_type="default",
)
print("✓ Agent routes added at /agent")


# Add simple chain routes (direct HTML generation)
add_routes(
    app,
    simple_chain,
    path="/simple",
    enabled_endpoints=["invoke", "batch", "stream", "playground"],
    playground_type="default",
)
print("✓ Simple chain routes added at /simple")


# Custom endpoints for convenience
@app.post("/generate")
async def generate_html_endpoint(screenshot_base64: str, description: str = "Generate HTML from this screenshot"):
    """
    Convenience endpoint for direct HTML generation.

    Args:
        screenshot_base64: Base64-encoded PNG screenshot
        description: Optional description for HTML generation

    Returns:
        Generated HTML code
    """
    from src.white_agent.langchain_agent import generate_html_from_screenshot_impl

    try:
        html_code = generate_html_from_screenshot_impl(screenshot_base64, description)
        return {
            "success": True,
            "html": html_code,
            "length": len(html_code)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("Starting LangServe server...")
    print("="*60)
    print("\n🌐 Access points:")
    print("  • Agent Playground:  http://localhost:8000/agent/playground")
    print("  • Simple Playground: http://localhost:8000/simple/playground")
    print("  • API Docs:          http://localhost:8000/docs")
    print("  • Health Check:      http://localhost:8000/health")
    print("\n" + "="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)

