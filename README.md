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

## Usage

First, configure `.env` with `OPENAI_API_KEY=...`, then

```bash
# Launch complete evaluation
uv run python main.py launch

# Start green agent only
uv run python main.py green

# Start white agent only
uv run python main.py white
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
