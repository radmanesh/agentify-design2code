"""Block matching using Hungarian algorithm for optimal assignment."""

import logging
import numpy as np
from typing import List, Tuple, Dict
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
from collections import Counter

from .block_detector import TextBlock
from .similarity_metrics import calculate_text_similarity

# Configure logging
logger = logging.getLogger(__name__)


class BlockMatcher:
    """Matches text blocks between reference and generated screenshots."""

    def __init__(self, min_similarity_threshold: float = 0.5):
        """
        Initialize the block matcher.

        Args:
            min_similarity_threshold: Minimum text similarity to consider a match
        """
        self.min_similarity_threshold = min_similarity_threshold

    def match_blocks(
        self,
        blocks_ref: List[TextBlock],
        blocks_gen: List[TextBlock],
        consecutive_bonus: float = 0.1,
        window_size: int = 1
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Find optimal matching between reference and generated blocks.

        Uses Hungarian algorithm (Jonker-Volgenant) for optimal assignment.

        Args:
            blocks_ref: Reference blocks
            blocks_gen: Generated blocks
            consecutive_bonus: Bonus for consecutive matches
            window_size: Window size for context similarity

        Returns:
            Tuple of (matched_pairs, cost_matrix) where matched_pairs is
            a list of (ref_idx, gen_idx) tuples
        """
        if len(blocks_ref) == 0 or len(blocks_gen) == 0:
            logger.warning("Empty block list, no matching possible")
            return [], np.array([])

        # Create cost matrix based on negative text similarity
        cost_matrix = self._create_cost_matrix(blocks_ref, blocks_gen)

        # Adjust costs for context similarity
        if window_size > 0:
            cost_matrix = self._adjust_cost_for_context(
                cost_matrix, consecutive_bonus, window_size
            )

        # Use Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Filter matches by minimum similarity threshold
        matched_pairs = []
        for i, j in zip(row_ind, col_ind):
            text_sim = calculate_text_similarity(
                blocks_ref[i].text, blocks_gen[j].text
            )
            if text_sim >= self.min_similarity_threshold:
                matched_pairs.append((i, j))

        return matched_pairs, cost_matrix

    def _create_cost_matrix(
        self,
        blocks_ref: List[TextBlock],
        blocks_gen: List[TextBlock]
    ) -> np.ndarray:
        """
        Create cost matrix for matching.

        Args:
            blocks_ref: Reference blocks
            blocks_gen: Generated blocks

        Returns:
            Cost matrix (negative similarity)
        """
        n = len(blocks_ref)
        m = len(blocks_gen)
        cost_matrix = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                similarity = calculate_text_similarity(
                    blocks_ref[i].text, blocks_gen[j].text
                )
                cost_matrix[i, j] = -similarity

        return cost_matrix

    def _adjust_cost_for_context(
        self,
        cost_matrix: np.ndarray,
        consecutive_bonus: float,
        window_size: int
    ) -> np.ndarray:
        """
        Adjust cost matrix to favor consecutive matches.

        Args:
            cost_matrix: Original cost matrix
            consecutive_bonus: Bonus factor for consecutive matches
            window_size: Window size for context

        Returns:
            Adjusted cost matrix
        """
        if window_size <= 0:
            return cost_matrix

        n, m = cost_matrix.shape
        adjusted = np.copy(cost_matrix)
        adjustments_made = 0

        for i in range(n):
            for j in range(m):
                # Skip if already poor match
                if adjusted[i][j] >= -0.5:
                    continue

                # Get nearby costs
                i_start = max(0, i - window_size)
                i_end = min(n, i + window_size + 1)
                j_start = max(0, j - window_size)
                j_end = min(m, j + window_size + 1)

                nearby = cost_matrix[i_start:i_end, j_start:j_end]
                flattened = nearby.flatten()
                sorted_arr = np.sort(flattened)[::-1]

                # Remove current element
                current_val = cost_matrix[i, j]
                sorted_arr = np.delete(
                    sorted_arr,
                    np.where(sorted_arr == current_val)[0][0]
                )

                # Calculate bonus from top-k neighbors
                top_k = sorted_arr[-(window_size * 2):]
                bonus = consecutive_bonus * np.sum(top_k)
                adjusted[i, j] += bonus
                adjustments_made += 1

        return adjusted

    def merge_blocks(
        self,
        blocks: List[TextBlock],
        merge_threshold: float = 0.05
    ) -> List[TextBlock]:
        """
        Merge consecutive blocks to improve matching.

        Args:
            blocks: List of blocks to potentially merge
            merge_threshold: Threshold for deciding whether to merge

        Returns:
            Merged list of blocks
        """
        if len(blocks) < 2:
            return blocks

        merged = deepcopy(blocks)
        changed = True

        while changed:
            changed = False

            # Try merging consecutive blocks
            for i in range(len(merged) - 1):
                # Create merged version
                test_merged = self._merge_two_blocks(merged[i], merged[i + 1])

                # Simple heuristic: merge if it seems beneficial
                # (In production, test against reference blocks)
                if self._should_merge(merged[i], merged[i + 1]):
                    merged[i] = test_merged
                    merged.pop(i + 1)
                    changed = True
                    break

        return merged

    def _merge_two_blocks(
        self,
        block1: TextBlock,
        block2: TextBlock
    ) -> TextBlock:
        """
        Merge two text blocks.

        Args:
            block1: First block
            block2: Second block

        Returns:
            Merged block
        """
        # Merge text
        merged_text = block1.text + " " + block2.text

        # Calculate bounding box
        x_min = min(block1.bbox[0], block2.bbox[0])
        y_min = min(block1.bbox[1], block2.bbox[1])
        x_max = max(
            block1.bbox[0] + block1.bbox[2],
            block2.bbox[0] + block2.bbox[2]
        )
        y_max = max(
            block1.bbox[1] + block1.bbox[3],
            block2.bbox[1] + block2.bbox[3]
        )
        merged_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        # Average color
        merged_color = tuple(
            (c1 + c2) // 2
            for c1, c2 in zip(block1.color, block2.color)
        )

        return TextBlock(
            text=merged_text,
            bbox=merged_bbox,
            color=merged_color
        )

    def _should_merge(self, block1: TextBlock, block2: TextBlock) -> bool:
        """
        Heuristic to decide if two blocks should be merged.

        Args:
            block1: First block
            block2: Second block

        Returns:
            True if blocks should be merged
        """
        # Simple heuristic: merge if vertically aligned and close
        x1_center = block1.bbox[0] + block1.bbox[2] / 2
        x2_center = block2.bbox[0] + block2.bbox[2] / 2

        y1_bottom = block1.bbox[1] + block1.bbox[3]
        y2_top = block2.bbox[1]

        # Check horizontal alignment
        horizontal_overlap = abs(x1_center - x2_center) < 0.1

        # Check vertical proximity
        vertical_gap = abs(y2_top - y1_bottom)
        close_vertically = vertical_gap < 0.05

        return horizontal_overlap and close_vertically
