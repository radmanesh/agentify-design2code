"""Evaluation module for Design2Code visual metrics."""

from .visual_evaluator import VisualEvaluator
from .block_detector import BlockDetector
from .block_matcher import BlockMatcher
from .similarity_metrics import (
    calculate_text_similarity,
    calculate_position_similarity,
    calculate_color_similarity,
    calculate_clip_similarity,
)

__all__ = [
    "VisualEvaluator",
    "BlockDetector",
    "BlockMatcher",
    "calculate_text_similarity",
    "calculate_position_similarity",
    "calculate_color_similarity",
    "calculate_clip_similarity",
]
