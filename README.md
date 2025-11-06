# Agentify Design2Code

Assessment framework for evaluating HTML generation from screenshots using A2A standards.

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
