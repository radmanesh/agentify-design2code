"""Main visual evaluation module for Design2Code."""

import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from .block_detector import BlockDetector, TextBlock
from .block_matcher import BlockMatcher
from .similarity_metrics import (
    calculate_text_similarity,
    calculate_position_similarity,
    calculate_color_similarity,
    calculate_clip_similarity_from_paths,
)
from .screenshot_generator import generate_screenshot_from_html


@dataclass
class EvaluationResult:
    """Results from visual evaluation."""
    block_match_score: float
    text_score: float
    position_score: float
    color_score: float
    clip_score: float
    overall_score: float
    matched_pairs: int
    total_ref_blocks: int
    total_gen_blocks: int
    details: Dict


class VisualEvaluator:
    """
    Evaluates visual similarity between reference and generated HTML screenshots.

    Implements the Design2Code evaluation metrics:
    - Block-Match: Coverage of matched blocks
    - Text: Text content similarity
    - Position: Layout positioning similarity
    - Color: Text color similarity (CIEDE2000)
    - CLIP: Visual similarity using CLIP embeddings
    """

    def __init__(
        self,
        min_similarity_threshold: float = 0.5,
        consecutive_bonus: float = 0.1,
        window_size: int = 1
    ):
        """
        Initialize the visual evaluator.

        Args:
            min_similarity_threshold: Minimum text similarity for matching
            consecutive_bonus: Bonus for consecutive context matches
            window_size: Window size for context matching
        """
        self.block_detector = BlockDetector()
        self.block_matcher = BlockMatcher(min_similarity_threshold)
        self.consecutive_bonus = consecutive_bonus
        self.window_size = window_size

    def evaluate(
        self,
        ref_html_path: str,
        gen_html_path: str,
        ref_screenshot_path: Optional[str] = None,
        gen_screenshot_path: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate visual similarity between reference and generated HTML.

        Args:
            ref_html_path: Path to reference HTML file
            gen_html_path: Path to generated HTML file
            ref_screenshot_path: Path to reference screenshot (auto-generated if None)
            gen_screenshot_path: Path to generated screenshot (auto-generated if None)

        Returns:
            EvaluationResult with all metrics
        """
        # Set screenshot paths
        if ref_screenshot_path is None:
            ref_screenshot_path = ref_html_path.replace(".html", ".png")
        if gen_screenshot_path is None:
            gen_screenshot_path = gen_html_path.replace(".html", ".png")

        # Generate screenshots if they don't exist
        if not os.path.exists(ref_screenshot_path):
            print(f"Generating reference screenshot: {ref_screenshot_path}")
            if not generate_screenshot_from_html(ref_html_path, ref_screenshot_path):
                print(f"Warning: Failed to generate reference screenshot")

        if not os.path.exists(gen_screenshot_path):
            print(f"Generating generated screenshot: {gen_screenshot_path}")
            if not generate_screenshot_from_html(gen_html_path, gen_screenshot_path):
                print(f"Warning: Failed to generate generated screenshot")

        # Detect blocks
        ref_blocks = self.block_detector.detect_blocks(ref_html_path, ref_screenshot_path)
        gen_blocks = self.block_detector.detect_blocks(gen_html_path, gen_screenshot_path)

        # Merge blocks with same bbox
        ref_blocks = self.block_detector.merge_blocks_by_bbox(ref_blocks)
        gen_blocks = self.block_detector.merge_blocks_by_bbox(gen_blocks)

        # Handle empty blocks
        if len(ref_blocks) == 0 or len(gen_blocks) == 0:
            print(f"Warning: Empty blocks detected (ref: {len(ref_blocks)}, gen: {len(gen_blocks)})")
            clip_score = self._safe_clip_score(ref_screenshot_path, gen_screenshot_path)
            return EvaluationResult(
                block_match_score=0.0,
                text_score=0.0,
                position_score=0.0,
                color_score=0.0,
                clip_score=clip_score,
                overall_score=0.2 * clip_score,
                matched_pairs=0,
                total_ref_blocks=len(ref_blocks),
                total_gen_blocks=len(gen_blocks),
                details={}
            )

        # Match blocks
        matched_pairs, cost_matrix = self.block_matcher.match_blocks(
            ref_blocks, gen_blocks, self.consecutive_bonus, self.window_size
        )

        # Calculate metrics
        scores = self._calculate_scores(
            ref_blocks, gen_blocks, matched_pairs,
            ref_screenshot_path, gen_screenshot_path
        )

        return scores

    def _calculate_scores(
        self,
        ref_blocks: List[TextBlock],
        gen_blocks: List[TextBlock],
        matched_pairs: List[Tuple[int, int]],
        ref_screenshot_path: str,
        gen_screenshot_path: str
    ) -> EvaluationResult:
        """
        Calculate all evaluation scores.

        Args:
            ref_blocks: Reference text blocks
            gen_blocks: Generated text blocks
            matched_pairs: List of (ref_idx, gen_idx) matched pairs
            ref_screenshot_path: Path to reference screenshot
            gen_screenshot_path: Path to generated screenshot

        Returns:
            EvaluationResult with all scores
        """
        if len(matched_pairs) == 0:
            print("Warning: No matched blocks")
            clip_score = self._safe_clip_score(ref_screenshot_path, gen_screenshot_path)
            return EvaluationResult(
                block_match_score=0.0,
                text_score=0.0,
                position_score=0.0,
                color_score=0.0,
                clip_score=clip_score,
                overall_score=0.2 * clip_score,
                matched_pairs=0,
                total_ref_blocks=len(ref_blocks),
                total_gen_blocks=len(gen_blocks),
                details={}
            )

        # Calculate block areas
        matched_areas = []
        total_ref_area = 0.0
        total_gen_area = 0.0

        ref_matched_indices = set(i for i, _ in matched_pairs)
        gen_matched_indices = set(j for _, j in matched_pairs)

        # Calculate total areas
        for i, block in enumerate(ref_blocks):
            area = block.bbox[2] * block.bbox[3]
            total_ref_area += area
            if i in ref_matched_indices:
                matched_areas.append(area)

        for j, block in enumerate(gen_blocks):
            area = block.bbox[2] * block.bbox[3]
            total_gen_area += area
            if j in gen_matched_indices:
                matched_areas.append(area)

        # Block-Match score
        total_area = total_ref_area + total_gen_area
        matched_area = sum(matched_areas)
        block_match_score = matched_area / total_area if total_area > 0 else 0.0

        # Calculate per-pair scores
        text_scores = []
        position_scores = []
        color_scores = []

        for ref_idx, gen_idx in matched_pairs:
            ref_block = ref_blocks[ref_idx]
            gen_block = gen_blocks[gen_idx]

            # Text similarity
            text_sim = calculate_text_similarity(ref_block.text, gen_block.text)
            text_scores.append(text_sim)

            # Position similarity
            pos_sim = calculate_position_similarity(ref_block.bbox, gen_block.bbox)
            position_scores.append(pos_sim)

            # Color similarity
            color_sim = calculate_color_similarity(ref_block.color, gen_block.color)
            color_scores.append(color_sim)

        # Average scores
        text_score = sum(text_scores) / len(text_scores) if text_scores else 0.0
        position_score = sum(position_scores) / len(position_scores) if position_scores else 0.0
        color_score = sum(color_scores) / len(color_scores) if color_scores else 0.0

        # CLIP score
        clip_score = self._safe_clip_score(ref_screenshot_path, gen_screenshot_path)

        # Overall score (average of 5 components)
        overall_score = 0.2 * (
            block_match_score + text_score + position_score + color_score + clip_score
        )

        return EvaluationResult(
            block_match_score=block_match_score,
            text_score=text_score,
            position_score=position_score,
            color_score=color_score,
            clip_score=clip_score,
            overall_score=overall_score,
            matched_pairs=len(matched_pairs),
            total_ref_blocks=len(ref_blocks),
            total_gen_blocks=len(gen_blocks),
            details={
                "matched_area": matched_area,
                "total_area": total_area,
                "text_scores": text_scores,
                "position_scores": position_scores,
                "color_scores": color_scores,
            }
        )

    def _safe_clip_score(self, path1: str, path2: str) -> float:
        """
        Safely calculate CLIP score with fallback.

        Args:
            path1: First image path
            path2: Second image path

        Returns:
            CLIP similarity score
        """
        try:
            if os.path.exists(path1) and os.path.exists(path2):
                return calculate_clip_similarity_from_paths(path1, path2)
            else:
                print(f"Warning: Screenshot missing (ref: {os.path.exists(path1)}, gen: {os.path.exists(path2)})")
                return 0.5
        except Exception as e:
            print(f"Warning: CLIP similarity calculation failed: {e}")
            return 0.5
