"""Evaluation module for Design2Code visual similarity assessment."""

from .visual_evaluator import VisualEvaluator, EvaluationResult
from .block_detector import BlockDetector, TextBlock
from .block_matcher import BlockMatcher
from .similarity_metrics import (
    calculate_text_similarity,
    calculate_position_similarity,
    calculate_color_similarity,
    calculate_clip_similarity,
)
from .screenshot_generator import generate_screenshot_from_html

__all__ = [
    "VisualEvaluator",
    "EvaluationResult",
    "BlockDetector",
    "TextBlock",
    "BlockMatcher",
    "calculate_text_similarity",
    "calculate_position_similarity",
    "calculate_color_similarity",
    "calculate_clip_similarity",
    "generate_screenshot_from_html",
]
