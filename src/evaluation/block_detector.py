"""Block detection from HTML screenshots using OCR-free approach."""

import os
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
import numpy as np
from .screenshot_generator import generate_screenshot_from_html

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """Represents a detected text block in a screenshot."""
    text: str
    bbox: Tuple[float, float, float, float]  # (x, y, width, height) normalized [0, 1]
    color: Tuple[int, int, int]  # RGB color


class BlockDetector:
    """Detects text blocks in HTML screenshots."""

    def __init__(self):
        """Initialize the block detector."""
        pass

    def detect_blocks(self, html_path: str, screenshot_path: str) -> List[TextBlock]:
        """
        Detect text blocks in a screenshot by analyzing the HTML.

        This is a simplified version. For production, you would implement
        the OCR-free detection from Design2Code or use an OCR library.

        Args:
            html_path: Path to HTML file
            screenshot_path: Path to screenshot PNG file

        Returns:
            List of detected text blocks
        """
        logger.debug(f"Detecting blocks in: {screenshot_path}")
        logger.trace(f"HTML file: {html_path}")

        # Ensure screenshot exists
        if not os.path.exists(screenshot_path):
            logger.debug(f"Screenshot not found, generating: {screenshot_path}")
            self._generate_screenshot(html_path, screenshot_path)

        # For now, return a simplified block detection
        # In production, implement the color-based OCR-free detection
        blocks = self._simple_block_detection(screenshot_path)

        logger.debug(f"Detected {len(blocks)} text blocks")
        for i, block in enumerate(blocks):
            logger.trace(f"  Block {i+1}: text='{block.text[:50]}...', bbox={block.bbox}, color={block.color}")

        return blocks

    def _generate_screenshot(self, html_path: str, output_path: str) -> None:
        """
        Generate screenshot from HTML file.

        Args:
            html_path: Path to HTML file
            output_path: Path to save screenshot
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Generating screenshot for {html_path}")
        success = generate_screenshot_from_html(html_path, output_path)

        if not success or not os.path.exists(output_path):
            logger.error(
                "Failed to generate screenshot from %s to %s",
                html_path,
                output_path,
            )
            raise RuntimeError(
                f"Screenshot generation failed for {html_path}; "
                "ensure Playwright or Selenium dependencies are installed."
            )

    def _simple_block_detection(self, screenshot_path: str) -> List[TextBlock]:
        """
        Simple block detection as a fallback.

        This is a placeholder - implement actual detection logic here.

        Args:
            screenshot_path: Path to screenshot

        Returns:
            List of detected blocks
        """
        # Placeholder implementation
        # In production, use OCR or the color-based detection method

        try:
            image = Image.open(screenshot_path)
            width, height = image.size

            logger.trace(f"Screenshot dimensions: {width}x{height}")
            logger.trace("Using placeholder block detection (OCR-free method not yet implemented)")

            # Return empty list for now - this will be improved
            # with actual OCR or the Design2Code OCR-free method
            return []

        except Exception as e:
            logger.error(f"Error detecting blocks in {screenshot_path}: {e}")
            return []

    def merge_blocks_by_bbox(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Merge blocks with identical bounding boxes.

        Args:
            blocks: List of text blocks

        Returns:
            Merged list of blocks
        """
        logger.trace(f"Merging {len(blocks)} blocks by bounding box")
        merged_dict = {}

        for block in blocks:
            bbox_key = block.bbox
            if bbox_key in merged_dict:
                # Merge text and average color
                existing = merged_dict[bbox_key]
                merged_text = existing.text + " " + block.text
                merged_color = tuple(
                    (c1 + c2) // 2
                    for c1, c2 in zip(existing.color, block.color)
                )
                merged_dict[bbox_key] = TextBlock(
                    text=merged_text,
                    bbox=bbox_key,
                    color=merged_color
                )
                logger.trace(f"  Merged blocks at bbox {bbox_key}")
            else:
                merged_dict[bbox_key] = block

        result = list(merged_dict.values())
        logger.debug(f"Merged {len(blocks)} blocks into {len(result)} unique blocks")
        return result
