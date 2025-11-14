# Tests for Agentify Design2Code

This directory contains pytest tests for the agentify-design2code project.

## Running Tests

### Install Dependencies

First, ensure pytest is installed:

```bash
uv add pytest
```

### Run All Tests

```bash
uv run pytest
```

### Run Specific Test File

```bash
uv run pytest tests/test_block_detector.py
```

### Run Tests with Coverage

```bash
uv run pytest --cov=src --cov-report=html
```

### Run Tests with Verbose Output

```bash
uv run pytest -v
```

### Run Specific Test

```bash
uv run pytest tests/test_block_detector.py::TestBlockDetector::test_detect_blocks_with_existing_screenshot
```

## Test Structure

### `test_block_detector.py`

Tests for the `BlockDetector` class, covering:

- **Initialization**: Verifies that BlockDetector instances are created correctly
- **Screenshot Generation**: Tests the workflow when screenshots need to be generated
- **Existing Screenshots**: Tests behavior when screenshots already exist
- **Error Handling**: Ensures proper error handling when screenshot generation fails
- **Block Detection**: Tests the placeholder block detection logic
- **Block Merging**: Tests merging blocks with identical bounding boxes
- **Ground Truth Testing**: Tests with actual data files:
  - `11.html` - Ground truth HTML
  - `11.png` - Ground truth screenshot
  - `11-gpt5-v2.html` - Generated HTML from GPT-5
- **Comparison Testing**: Compares block detection between ground truth and generated HTML

The tests use:
- **Mocking**: `unittest.mock` to avoid dependencies on Playwright/browser automation
- **Fixtures**: pytest fixtures for creating temporary test files and detector instances
- **Temporary Files**: `tmp_path` fixture for isolated test file creation
- **Real Data Files**: Actual HTML and screenshot files from `data/` directory

## Integration Tests

Some tests are marked with `@pytest.mark.skipif` and can be skipped by setting:

```bash
export SKIP_INTEGRATION_TESTS=1
uv run pytest
```

## Notes

- Tests mock the `generate_screenshot_from_html` function to avoid requiring Playwright installation
- Temporary files are automatically cleaned up after each test
- The current block detection implementation returns empty lists (placeholder behavior)
- Tests verify directory creation, file handling, and error conditions

