# Agentify Design2Code

Assessment framework for evaluating HTML generation from screenshots using A2A standards and Google ADK integration.

## Project Structure

```
src/
├── green_agent/    # Assessment manager agent
├── white_agent/    # Target agent being tested
├── my_util/        # Utility functions
└── launcher.py     # Evaluation coordinator
data/               # HTML files and screenshots
```

## Installation

1. Install dependencies:

```bash
uv sync
```

2. Install Playwright browser for screenshot generation:

```bash
uv run playwright install chromium
```

This downloads the Chromium browser (~130MB) needed for generating screenshots from HTML files during evaluation.

## Configuration

Create a `.env` file with the following variables:

```bash
# Required: OpenAI API Key for white agent
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Google API Key for ADK agents (only if using ADK)
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Evaluation Debug Level
# Options: INFO, DEBUG, TRACE, or leave empty to disable detailed logging
# - INFO: High-level metrics (block counts, final scores)
# - DEBUG: Per-pair details, matching statistics
# - TRACE: All intermediate calculations, raw data
EVAL_DEBUG_LEVEL=INFO
```

## Usage

After configuring `.env`, run:

```bash
# Launch complete evaluation
uv run python main.py launch

# Start green agent only
uv run python main.py green

# Start white agent only (A2A HTTP server)
uv run python main.py white

# Start white agent with LangServe web interface
uv run python main.py langserve
```

## Dataset

The Design2Code dataset contains HTML files paired with screenshots. The green agent loads these pairs and creates assessment tasks for the white agent to generate HTML from screenshots.

## Architecture

The white agent is a **native A2A HTTP server** using OpenAI for HTML generation, with an optional ADK client wrapper for integration with ADK workflows.

```
┌─────────────────────────────────────┐
│   White Agent A2A HTTP Server      │
│   (OpenAI GPT-4o via LiteLLM)     │
│                                     │
│  ┌──────────────────────────────┐  │
│  │  HTML Generation Logic       │  │
│  │  - Vision model processing   │  │
│  │  - HTML/CSS generation       │  │
│  └──────────────────────────────┘  │
└──────────┬──────────────────────────┘
           │ HTTP (A2A Protocol)
           │
    ┌──────┴───────┐
    │              │
┌───▼────┐    ┌────▼─────────────┐
│  A2A   │    │ ADK Client       │
│Clients │    │ Wrapper          │
│(Green  │    │ (Optional)       │
│ Agent) │    └────┬─────────────┘
└────────┘         │
                   ▼
            ┌─────────────┐
            │ ADK Agents  │
            │ (Gemini)    │
            └─────────────┘
```

**Key Points:**
- **Primary**: A2A HTTP server using OpenAI (no Google credentials needed)
- **Optional**: ADK client wrapper for calling from ADK workflows
- **Separation**: White agent doesn't depend on ADK
- **Flexibility**: Can be called by any HTTP client (A2A or custom)

## For A2A Users (Green Agent)

Just start the server and call via A2A. The green agent does this automatically:

```bash
# Start white agent server
uv run python main.py white

# Run evaluation (green agent calls white agent)
uv run python main.py launch
```

**Requirements**: Only `OPENAI_API_KEY` needed

## For ADK Users

The white agent runs as an HTTP server. ADK users can call it using the client wrapper:

### Basic Usage

```python
from src.white_agent import call_white_agent_http
import asyncio

async def generate_html():
    # White agent server must be running
    html_code = await call_white_agent_http(
        screenshot_base64="...",  # Your base64-encoded PNG
        white_agent_url="http://localhost:10002",
        description="Generate HTML from this screenshot"
    )
    return html_code

asyncio.run(generate_html())
```

### As a Tool in ADK Agents

```python
from google.adk.agents import Agent
from src.white_agent import create_white_agent_tool

# Create tool that calls white agent via HTTP
white_tool = create_white_agent_tool(
    white_agent_url="http://localhost:10002"
)

# Use in an ADK agent (Gemini for orchestration)
coordinator = Agent(
    model="gemini-2.0-flash",  # Gemini orchestrates
    name="design_coordinator",
    description="Coordinates design-to-code workflows",
    instruction="Use the HTML generation tool for design tasks.",
    tools=[white_tool]  # Delegates to white agent via HTTP
)
```

**Requirements**:
- `OPENAI_API_KEY` for white agent server
- `GOOGLE_API_KEY` for your ADK agents (Gemini)

### Running Examples

The project includes comprehensive examples:

```bash
# Start white agent server first
uv run python main.py white

# Then run examples (in another terminal)
uv run python -m src.white_agent.examples.basic_adk_usage
uv run python -m src.white_agent.examples.agent_tool_usage
uv run python -m src.white_agent.examples.a2a_compatibility
```

### Benefits of This Architecture

- ✅ **Uses OpenAI**: White agent uses OpenAI (no Google credentials needed)
- ✅ **Simple**: Standard A2A HTTP server
- ✅ **ADK Compatible**: Optional wrapper for ADK users
- ✅ **Flexible**: Mix different LLMs (Gemini for orchestration, OpenAI for vision)
- ✅ **Independent**: White agent doesn't depend on ADK
- ✅ **Backward Compatible**: Existing A2A clients work unchanged

### How It Works

1. **White Agent**: A2A HTTP server using OpenAI via LiteLLM
2. **For A2A Clients**: Call directly via A2A protocol
3. **For ADK Users**: Use client wrapper to call HTTP server
4. **Multi-Agent**: ADK agents (Gemini) delegate to white agent (OpenAI) via HTTP

## For LangChain Users (Recommended Web Interface)

The white agent now includes a **LangChain/LangServe interface** with a professional web UI and REST API, all using OpenAI GPT-4o Vision.

### Quick Start

```bash
# Install dependencies (if not already done)
uv sync

# Start LangServe web interface
uv run python main.py langserve
```

Then open your browser to:
- **Interactive Playground**: http://localhost:8000/agent/playground
- **API Documentation**: http://localhost:8000/docs
- **Simple Mode**: http://localhost:8000/simple/playground

### Features

- ✅ **Professional Web UI** - Interactive chat interface with playground
- ✅ **OpenAI GPT-4o Vision** - Direct OpenAI integration (no Google dependency)
- ✅ **Streaming Responses** - Real-time response streaming
- ✅ **REST API** - Full API for programmatic access
- ✅ **Agent Mode** - Full reasoning and tool use capabilities
- ✅ **Simple Mode** - Direct HTML generation without agent overhead

### Using the Agent in Code

#### Direct Function Call (Simplest)

```python
from src.white_agent import generate_html_from_screenshot_impl
import asyncio

async def generate():
    html = generate_html_from_screenshot_impl(
        screenshot_base64="<your_base64_screenshot>",
        description="Generate a landing page"
    )
    return html

asyncio.run(generate())
```

#### LangChain Agent Executor

```python
from src.white_agent import create_white_agent
import asyncio

async def use_agent():
    agent = create_white_agent()

    result = agent.invoke({
        "input": "Generate HTML from this screenshot: <base64_data>"
    })

    print(result["output"])

asyncio.run(use_agent())
```

#### Simple Chain (Fast, No Agent)

```python
from src.white_agent import create_simple_chain

chain = create_simple_chain()

result = chain({
    "screenshot_base64": "<base64_data>",
    "description": "Generate HTML"
})

html = result["output"]
```

### API Usage

The LangServe API provides multiple endpoints:

```bash
# Agent endpoint (with reasoning)
curl -X POST "http://localhost:8000/agent/invoke" \
  -H "Content-Type: application/json" \
  -d '{"input": {"input": "Generate HTML from: <base64>"}}'

# Simple endpoint (direct generation)
curl -X POST "http://localhost:8000/simple/invoke" \
  -H "Content-Type: application/json" \
  -d '{"input": {"screenshot_base64": "<base64>", "description": "Generate HTML"}}'

# Health check
curl http://localhost:8000/health
```

### Running Examples

```bash
# Start LangServe first
uv run python main.py langserve

# In another terminal, run examples
uv run python -m src.white_agent.examples.langchain_usage
uv run python -m src.white_agent.examples.a2a_compatibility
```

### Architecture Comparison

| Interface | Port | Use Case | LLM |
|-----------|------|----------|-----|
| **A2A Server** | 10002 | Green agent evaluation, A2A clients | OpenAI GPT-4o |
| **LangServe** | 8000 | Web UI, REST API, LangChain agents | OpenAI GPT-4o |
| **ADK Wrapper** | - | ADK agents (calls A2A server) | Gemini (orchestration) + OpenAI (HTML gen) |

**Recommendation**: Use LangServe for:
- Interactive web interface
- Development and testing
- REST API access
- LangChain-based workflows
- When you want the best web experience with OpenAI

Use A2A server for:
- Green agent evaluation
- Production A2A workflows
- When you need A2A protocol compliance

### Benefits of LangServe Interface

- ✅ **All OpenAI** - No Google API keys needed
- ✅ **Better UI** - Professional web interface with playground
- ✅ **Developer Friendly** - Interactive docs, streaming, batch processing
- ✅ **Production Ready** - FastAPI backend, scalable
- ✅ **Flexible** - Agent mode for reasoning, simple mode for speed
- ✅ **Observable** - Compatible with LangSmith for monitoring

**Requirements**: Only `OPENAI_API_KEY` needed (no Google dependencies)
