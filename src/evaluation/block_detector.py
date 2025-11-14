"""Block detection from HTML screenshots using OCR-free approach."""

import os
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image, ImageColor
import numpy as np
import cv2
from bs4 import BeautifulSoup, NavigableString, Tag, Comment
from .screenshot_generator import generate_screenshot_from_html

# Patch for numpy compatibility
# numpy.asscalar was removed in numpy 1.23+, but some libraries still use it
def _patch_asscalar(a):
    """Compatibility patch for numpy.asscalar removal."""
    return a.item()

if not hasattr(np, "asscalar"):
    setattr(np, "asscalar", _patch_asscalar)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """Represents a detected text block in a screenshot."""
    text: str
    bbox: Tuple[float, float, float, float]  # (x, y, width, height) normalized [0, 1]
    color: Tuple[int, int, int]  # RGB color


# Helper functions ported from Design2Code ocr_free_utils.py

def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert an RGB tuple to hexadecimal format."""
    return '{:02X}{:02X}{:02X}'.format(*rgb)


class ColorPool:
    """Manages a pool of colors for text coloring in HTML."""

    def __init__(self, offset: int = 0):
        color_values = list(range(10, 251, 16))
        color_list = [
            ((r + offset) % 256, (g + offset) % 256, (b + offset) % 256)
            for r in color_values
            for g in color_values
            for b in color_values
        ]
        self.color_pool = [rgb_to_hex(color) for color in color_list]

    def pop_color(self) -> str:
        if self.color_pool:
            return self.color_pool.pop()
        else:
            raise NotImplementedError("Color pool exhausted")


def process_html(input_file_path: str, output_file_path: str, offset: int = 0) -> None:
    """
    Process HTML to add unique color coding to text elements.

    Args:
        input_file_path: Path to input HTML file
        output_file_path: Path to output HTML file with color coding
        offset: Color offset for creating variations
    """
    # Read the input HTML file
    with open(input_file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    def update_style(element, property_name: str, value) -> None:
        """Update element's style attribute with the given property and value."""
        important_value = f"{value} !important"
        styles = element.attrs.get('style', '').split(';')
        updated_styles = [
            s for s in styles
            if not s.strip().startswith(property_name) and len(s.strip()) > 0
        ]
        updated_styles.append(f"{property_name}: {important_value}")
        element['style'] = '; '.join(updated_styles).strip()

    # Set the background color of all elements to transparent white
    for element in soup.find_all(True):
        update_style(element, 'background-color', 'rgba(255, 255, 255, 0.0)')

    color_pool = ColorPool(offset)

    # Assign a unique color to text within each text-containing element
    text_tags = [
        'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'span', 'a', 'b',
        'li', 'table', 'td', 'th', 'button', 'footer', 'header', 'figcaption'
    ]
    for tag in soup.find_all(text_tags):
        color = f"#{color_pool.pop_color()}"
        update_style(tag, 'color', color)
        update_style(tag, 'opacity', 1.0)

    # Write the modified HTML to a new file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(str(soup))


def similar(n1: float, n2: float) -> bool:
    """Check if two numbers are similar within a threshold of 8."""
    return abs(n1 - n2) <= 8


def find_different_pixels(image1_path: str, image2_path: str) -> Optional[np.ndarray]:
    """
    Find pixels that differ between two images based on color offset.

    Args:
        image1_path: Path to first image
        image2_path: Path to second image

    Returns:
        Array of coordinates where pixels differ, or None if images don't match
    """
    # Open the images
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # Ensure both images are of the same size
    if img1.size != img2.size:
        logger.warning(f"Images are not the same size: {image1_path}, {image2_path}")
        return None

    # Convert images to RGB if they are not
    img1 = img1.convert('RGB')
    img2 = img2.convert('RGB')

    # Get pixel data
    pixels1 = img1.load()
    pixels2 = img2.load()

    # List to store coordinates of different pixels
    different_pixels = []

    # Iterate through each pixel
    for x in range(img1.size[0]):
        for y in range(img1.size[1]):
            # Compare pixel colors
            r1, g1, b1 = pixels1[x, y]
            r2, g2, b2 = pixels2[x, y]
            if (similar((r1 + 50) % 256, r2) and
                similar((g1 + 50) % 256, g2) and
                similar((b1 + 50) % 256, b2)):
                different_pixels.append((y, x))

    if len(different_pixels) > 0:
        return np.stack(different_pixels)
    else:
        return None


def extract_text_with_color(html_file: str) -> List:
    """
    Extract text with color information from HTML file.

    Args:
        html_file: Path to HTML file

    Returns:
        Nested list of (text, color) tuples
    """
    def get_color(tag) -> Optional[str]:
        if 'style' in tag.attrs:
            styles = tag['style'].split(';')
            color_style = [s for s in styles if 'color' in s and 'background-color' not in s]
            if color_style:
                color = color_style[-1].split(':')[1].strip().replace(" !important", "")
                if color[0] == "#":
                    return color
                else:
                    try:
                        if color.startswith('rgb'):
                            color = tuple(map(int, color[4:-1].split(',')))
                        else:
                            color = ImageColor.getrgb(color)
                        return '#{:02x}{:02x}{:02x}'.format(*color)
                    except ValueError:
                        logger.warning(f"Unable to identify or convert color in {html_file}: {color}")
                        return None
        return None

    def extract_text_recursive(element, parent_color: str = '#000000'):
        if isinstance(element, Comment):
            return None
        elif isinstance(element, NavigableString):
            text = element.strip()
            return (text, parent_color) if text else None
        elif isinstance(element, Tag):
            current_color = get_color(element) or parent_color
            children_texts = filter(
                None,
                [extract_text_recursive(child, current_color) for child in element.children]
            )
            return list(children_texts)

    with open(html_file, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        body = soup.body
        return extract_text_recursive(body) if body else []


def flatten_tree(tree) -> List:
    """Flatten a nested tree structure into a list."""
    flat_list = []

    def flatten(node):
        if isinstance(node, list):
            for item in node:
                flatten(item)
        else:
            flat_list.append(node)

    flatten(tree)
    return flat_list


def average_color(image_path: str, coordinates: np.ndarray) -> Tuple[int, int, int]:
    """
    Calculate the average color of the specified coordinates in the given image.

    Args:
        image_path: Path to image file
        coordinates: 2D numpy array of coordinates, where each row represents [x, y]

    Returns:
        Tuple representing the average color (R, G, B)
    """
    # Convert image to numpy array
    image_array = np.array(Image.open(image_path).convert('RGB'))

    # Extract colors at the specified coordinates
    colors = [image_array[x, y] for x, y in coordinates]

    # Calculate the average color
    avg_color = np.mean(colors, axis=0)

    return tuple(avg_color.astype(int))


def get_blocks_from_image_diff_pixels(
    image_path: str,
    original_image_path: str,
    html_text_color_tree: List,
    different_pixels: np.ndarray
) -> List[Dict]:
    """
    Extract text blocks from image using different pixels.

    Args:
        image_path: Path to color-coded screenshot
        original_image_path: Path to original screenshot (for color extraction)
        html_text_color_tree: List of (text, color) tuples
        different_pixels: Array of pixel coordinates that differ

    Returns:
        List of block dictionaries with text, bbox, and color
    """
    image = cv2.imread(image_path)
    x_w = image.shape[0]
    y_w = image.shape[1]

    def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
        """Convert a hex color string to a BGR color tuple."""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return rgb[::-1]

    def get_intersect(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """Find intersection of two arrays."""
        # Reshape the arrays to 1D
        arr1_reshaped = arr1.view([('', arr1.dtype)] * arr1.shape[1])
        arr2_reshaped = arr2.view([('', arr2.dtype)] * arr2.shape[1])

        # Find the intersection
        common_rows = np.intersect1d(arr1_reshaped, arr2_reshaped)

        # Reshape the result back to 2D, if needed
        common_rows = common_rows.view(arr1.dtype).reshape(-1, arr1.shape[1])
        return common_rows

    blocks = []
    for item in html_text_color_tree:
        try:
            color = np.array(hex_to_bgr(item[1]), dtype="uint8")
        except:
            continue

        lower = color - 4
        upper = color + 4

        mask = cv2.inRange(image, lower, upper)

        coords = np.column_stack(np.where(mask > 0))

        coords = get_intersect(coords, different_pixels)

        if coords.size == 0:
            continue

        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)
        color = average_color(original_image_path, coords)

        blocks.append({
            'text': item[0].lower(),
            'bbox': (
                y_min / y_w,
                x_min / x_w,
                (y_max - y_min + 1) / y_w,
                (x_max - x_min + 1) / x_w
            ),
            'color': color
        })

    return blocks


class BlockDetector:
    """Detects text blocks in HTML screenshots."""

    def __init__(self):
        """Initialize the block detector."""
        pass

    def detect_blocks(self, html_path: str, screenshot_path: str) -> List[TextBlock]:
        """
        Detect text blocks in a screenshot by analyzing the HTML.

        Uses the OCR-free detection method from Design2Code, which:
        1. Creates color-coded versions of the HTML
        2. Takes screenshots and compares pixel differences
        3. Extracts text blocks with position and color information

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

        # Use OCR-free block detection
        blocks = self._ocr_free_block_detection(html_path, screenshot_path)

        logger.debug(f"Detected {len(blocks)} text blocks")
        for i, block in enumerate(blocks):
            text_preview = block.text[:50] if len(block.text) > 50 else block.text
            logger.trace(
                f"  Block {i+1}: text='{text_preview}...', "
                f"bbox={block.bbox}, color={block.color}"
            )

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

    def _ocr_free_block_detection(
        self, html_path: str, screenshot_path: str
    ) -> List[TextBlock]:
        """
        OCR-free block detection using color-coded HTML analysis.

        This implements the Design2Code OCR-free detection method by:
        1. Creating two color-coded versions of the HTML (offset 0 and 50)
        2. Taking screenshots of both versions
        3. Finding pixels that differ between the two screenshots
        4. Extracting text blocks based on color regions

        Args:
            html_path: Path to HTML file
            screenshot_path: Path to the original screenshot

        Returns:
            List of detected TextBlock objects
        """
        logger.debug(f"Running OCR-free block detection on {html_path}")

        # Generate temporary file paths in the same directory as screenshot
        screenshot_dir = Path(screenshot_path).parent
        screenshot_stem = Path(screenshot_path).stem

        # Temporary files for color-coded HTML and screenshots
        p_html = screenshot_dir / f"{screenshot_stem}_p.html"
        p_html_1 = screenshot_dir / f"{screenshot_stem}_p_1.html"
        p_png = screenshot_dir / f"{screenshot_stem}_p.png"
        p_png_1 = screenshot_dir / f"{screenshot_stem}_p_1.png"

        try:
            # Process HTML with two different color offsets
            logger.trace("Processing HTML with color offsets 0 and 50")
            process_html(html_path, str(p_html), offset=0)
            process_html(html_path, str(p_html_1), offset=50)

            # Take screenshots of both color-coded versions
            logger.trace(f"Generating screenshot for {p_html}")
            success1 = generate_screenshot_from_html(str(p_html), str(p_png))
            if not success1:
                logger.warning(f"Failed to generate screenshot for {p_html}")
                return []

            logger.trace(f"Generating screenshot for {p_html_1}")
            success2 = generate_screenshot_from_html(str(p_html_1), str(p_png_1))
            if not success2:
                logger.warning(f"Failed to generate screenshot for {p_html_1}")
                return []

            # Find different pixels between the two screenshots
            logger.trace("Finding different pixels between color-coded screenshots")
            different_pixels = find_different_pixels(str(p_png), str(p_png_1))

            if different_pixels is None:
                logger.warning(
                    f"Unable to get pixels with different colors from {p_png}, {p_png_1}"
                )
                return []

            logger.trace(f"Found {len(different_pixels)} different pixels")

            # Extract text with color information from HTML
            logger.trace("Extracting text with color information from HTML")
            html_text_color_tree = flatten_tree(extract_text_with_color(str(p_html)))

            logger.trace(f"Extracted {len(html_text_color_tree)} text elements")

            # Get blocks from image difference pixels
            logger.trace("Extracting blocks from image difference pixels")
            blocks_dict = get_blocks_from_image_diff_pixels(
                str(p_png),
                screenshot_path,
                html_text_color_tree,
                different_pixels
            )

            # Convert dict format to TextBlock objects
            blocks = [
                TextBlock(
                    text=block['text'],
                    bbox=block['bbox'],
                    color=block['color']
                )
                for block in blocks_dict
            ]

            logger.debug(f"Detected {len(blocks)} text blocks using OCR-free method")

            return blocks

        except Exception as e:
            logger.error(f"Error in OCR-free block detection: {e}", exc_info=True)
            return []

        finally:
            # Clean up temporary files
            logger.trace("Cleaning up temporary files")
            for tmp_file in [p_html, p_html_1, p_png, p_png_1]:
                try:
                    if tmp_file.exists():
                        tmp_file.unlink()
                        logger.trace(f"Removed temporary file: {tmp_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {tmp_file}: {e}")

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
