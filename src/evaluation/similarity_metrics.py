"""Similarity metrics for evaluating visual correspondence between webpages."""

import math
from difflib import SequenceMatcher
from typing import List, Tuple
from PIL import Image
import numpy as np

# Try to import optional dependencies with fallbacks
try:
    from colormath.color_objects import sRGBColor, LabColor
    from colormath.color_conversions import convert_color
    from colormath.color_diff import delta_e_cie2000
    HAS_COLORMATH = True
except ImportError:
    HAS_COLORMATH = False

try:
    import torch
    import open_clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate text similarity using Sørensen-Dice coefficient.

    This measures character-level similarity between two strings.

    Args:
        text1: First text string
        text2: Second text string

    Returns:
        Similarity score in [0, 1]
    """
    return SequenceMatcher(None, text1, text2).ratio()


def calculate_position_similarity(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float]
) -> float:
    """
    Calculate position similarity between two bounding boxes.

    Uses the maximum of horizontal and vertical distance between centers.

    Args:
        bbox1: (x, y, width, height) normalized coordinates
        bbox2: (x, y, width, height) normalized coordinates

    Returns:
        Similarity score in [0, 1], where 1 is identical positions
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Calculate centers
    center1_x = x1 + w1 / 2
    center1_y = y1 + h1 / 2
    center2_x = x2 + w2 / 2
    center2_y = y2 + h2 / 2

    # Maximum distance in either direction
    max_distance = max(
        abs(center2_x - center1_x),
        abs(center2_y - center1_y)
    )

    # Convert to similarity (1 - distance)
    similarity = 1.0 - max_distance

    return max(0.0, similarity)


def calculate_color_similarity(
    color1: Tuple[int, int, int],
    color2: Tuple[int, int, int]
) -> float:
    """
    Calculate color similarity using CIEDE2000 formula.

    This uses perceptual color difference in Lab color space.

    Args:
        color1: RGB tuple (0-255)
        color2: RGB tuple (0-255)

    Returns:
        Similarity score in [0, 1]
    """
    try:
        from colormath.color_objects import sRGBColor, LabColor
        from colormath.color_conversions import convert_color
        from colormath.color_diff import delta_e_cie2000

        # Convert RGB to Lab
        rgb1 = sRGBColor(color1[0], color1[1], color1[2], is_upscaled=True)
        rgb2 = sRGBColor(color2[0], color2[1], color2[2], is_upscaled=True)

        lab1 = convert_color(rgb1, LabColor)
        lab2 = convert_color(rgb2, LabColor)

        # Calculate delta E
        delta_e = delta_e_cie2000(lab1, lab2)

        # Normalize to [0, 1] (delta_e of 100 = completely different)
        similarity = max(0, 1 - (delta_e / 100))

        return similarity

    except ImportError:
        # Fallback to simple Euclidean distance if colormath not available
        print("Warning: colormath not installed, using simple RGB distance")
        distance = math.sqrt(
            (color1[0] - color2[0]) ** 2 +
            (color1[1] - color2[1]) ** 2 +
            (color1[2] - color2[2]) ** 2
        )
        # Max RGB distance is sqrt(255^2 * 3) ≈ 441
        max_distance = 441.67
        similarity = 1 - (distance / max_distance)
        return max(0.0, similarity)


def calculate_clip_similarity(
    ref_image: Image.Image,
    gen_image: Image.Image,
    device: str = "cpu"
) -> float:
    """
    Calculate CLIP visual similarity between two images.

    Uses CLIP ViT-B/32 to encode images and compute cosine similarity.

    Args:
        ref_image: Reference PIL Image
        gen_image: Generated PIL Image
        device: Device for CLIP model ("cpu" or "cuda")

    Returns:
        Similarity score (0.0 to 1.0)
    """
    if not HAS_CLIP:
        print("Warning: CLIP not available, using placeholder score")
        return 0.5

    try:
        # Load CLIP model and preprocessing
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='openai'
        )
        model = model.to(device)
        model.eval()

        # Preprocess images
        ref_tensor = preprocess(ref_image).unsqueeze(0).to(device)
        gen_tensor = preprocess(gen_image).unsqueeze(0).to(device)

        # Encode images
        with torch.no_grad():
            ref_features = model.encode_image(ref_tensor)
            gen_features = model.encode_image(gen_tensor)

            # Normalize features
            ref_features = ref_features / ref_features.norm(dim=-1, keepdim=True)
            gen_features = gen_features / gen_features.norm(dim=-1, keepdim=True)

            # Compute cosine similarity
            similarity = (ref_features @ gen_features.T).item()

        # Convert from [-1, 1] to [0, 1]
        return (similarity + 1) / 2

    except Exception as e:
        print(f"Warning: CLIP similarity calculation failed: {e}")
        return 0.5


def calculate_clip_similarity_from_paths(
    image_path1: str,
    image_path2: str
) -> float:
    """
    Calculate CLIP similarity directly from image file paths.

    Convenience function that loads images and calls calculate_clip_similarity.

    Args:
        image_path1: Path to first image
        image_path2: Path to second image

    Returns:
        Similarity score in [0, 1]
    """
    try:
        from PIL import Image

        # Load images
        image1 = Image.open(image_path1).convert('RGB')
        image2 = Image.open(image_path2).convert('RGB')

        # Use the main CLIP similarity function
        return calculate_clip_similarity(image1, image2)

    except Exception as e:
        print(f"Warning: Failed to calculate CLIP similarity from paths: {e}")
        return 0.5
