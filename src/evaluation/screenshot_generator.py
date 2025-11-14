"""Screenshot generation utilities for HTML files."""

import time
import asyncio
import logging
from pathlib import Path
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)


def generate_screenshot_from_html(
    html_path: str,
    output_path: str,
    width: int = 1280,
    height: Optional[int] = None,
    method: str = "playwright"
) -> bool:
    """
    Generate a screenshot from an HTML file.

    Args:
        html_path: Path to HTML file to render
        output_path: Path where screenshot PNG should be saved
        width: Viewport width in pixels (default: 1280)
        height: Requested viewport height. Ignored for Playwright full-page
            capture but still accepted for API compatibility.
        method: Method to use ("playwright" or "selenium", default: "playwright")

    Returns:
        True if screenshot was generated successfully, False otherwise
    """
    # Check if we're inside an asyncio event loop
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context, need to run async version in a thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_generate_screenshot_sync, html_path, output_path, width, height, method)
            return future.result()
    except RuntimeError:
        # No event loop running, we can use sync version directly
        return _generate_screenshot_sync(html_path, output_path, width, height, method)


def _generate_screenshot_sync(
    html_path: str,
    output_path: str,
    width: int = 1280,
    height: Optional[int] = None,
    method: str = "playwright"
) -> bool:
    """
    Synchronous screenshot generation (internal function).

    Args:
        html_path: Path to HTML file to render
        output_path: Path where screenshot PNG should be saved
        width: Viewport width in pixels (default: 1280)
        height: Requested viewport height. Ignored for Playwright full-page
            capture but still accepted for API compatibility.
        method: Method to use ("playwright" or "selenium", default: "playwright")

    Returns:
        True if screenshot was generated successfully, False otherwise
    """
    # Try playwright first (faster and more reliable)
    if method == "playwright":
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                # Launch browser in headless mode
                browser = p.chromium.launch(headless=True)
                # Create new page with specified viewport width. Playwright
                # requires a height value, but full_page=True below will still
                # capture the entire document. Default height falls back to 720
                # if none is supplied.
                viewport = {"width": width, "height": height or 720}
                page = browser.new_page(viewport=viewport)

                # Load HTML file
                html_file_url = f"file://{Path(html_path).resolve()}"
                page.goto(html_file_url, wait_until="networkidle")

                # Wait a bit for rendering to complete
                time.sleep(0.5)

                # Take full-page screenshot to capture entire height
                page.screenshot(path=output_path, full_page=True)

                # Close browser
                browser.close()

                return True

        except ImportError:
            logger.warning("playwright not installed, falling back to selenium")
            method = "selenium"
        except Exception as e:
            logger.warning(f"playwright screenshot failed: {e}")
            return False

    # Fallback to selenium
    if method == "selenium":
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options

            # Configure Chrome options for headless mode
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            # Use a large height for selenium to capture full page
            selenium_height = height if height else 10000
            chrome_options.add_argument(f"--window-size={width},{selenium_height}")

            # Create driver
            driver = webdriver.Chrome(options=chrome_options)

            try:
                # Load HTML file
                html_file_url = f"file://{Path(html_path).resolve()}"
                driver.get(html_file_url)

                # Wait for page to load
                time.sleep(1)

                # Take screenshot (selenium's save_screenshot captures full page by default)
                driver.save_screenshot(output_path)

                return True

            finally:
                # Clean up driver
                driver.quit()

        except ImportError:
            logger.error("Neither playwright nor selenium is installed")
            logger.error("Install with: uv add playwright && uv run playwright install chromium")
            logger.error("Or: uv add selenium")
            return False
        except Exception as e:
            logger.error(f"selenium screenshot failed: {e}")
            return False

    return False
