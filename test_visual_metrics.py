#!/usr/bin/env python3
"""
Test script to evaluate generated HTML against ground truth.
Compares 11gen.html with 11.html and displays all metrics.

Usage with uv:
    uv run python test_visual_metrics.py

Or directly:
    uv run test_visual_metrics.py
"""

import os
import sys
from pathlib import Path

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import from evaluation module
from evaluation import VisualEvaluator


def main():
    """Run visual evaluation and display all metrics."""

    # Paths
    ref_html = "data/11.html"
    gen_html = "data/11gen.html"

    # Check if files exist
    if not os.path.exists(ref_html):
        print(f"Error: Reference HTML not found: {ref_html}")
        return 1

    if not os.path.exists(gen_html):
        print(f"Error: Generated HTML not found: {gen_html}")
        return 1

    print("=" * 80)
    print("VISUAL EVALUATION TEST")
    print("=" * 80)
    print(f"Reference HTML: {ref_html}")
    print(f"Generated HTML: {gen_html}")
    print("=" * 80)
    print()

    # Create evaluator with default Design2Code parameters
    evaluator = VisualEvaluator(
        min_similarity_threshold=0.5,
        consecutive_bonus=0.1,
        window_size=1
    )

    # Run evaluation
    try:
        result = evaluator.evaluate(
            ref_html_path=ref_html,
            gen_html_path=gen_html
        )

        # Display detailed results
        print()
        print("=" * 80)
        print("DETAILED METRICS BREAKDOWN")
        print("=" * 80)
        print()
        print(f"Block Statistics:")
        print(f"  - Reference blocks detected: {result.total_ref_blocks}")
        print(f"  - Generated blocks detected: {result.total_gen_blocks}")
        print(f"  - Matched pairs: {result.matched_pairs}")
        print(f"  - Match rate (ref): {result.matched_pairs / result.total_ref_blocks * 100:.1f}%" if result.total_ref_blocks > 0 else "  - Match rate (ref): N/A")
        print(f"  - Match rate (gen): {result.matched_pairs / result.total_gen_blocks * 100:.1f}%" if result.total_gen_blocks > 0 else "  - Match rate (gen): N/A")
        print()

        print(f"Individual Metric Scores:")
        print(f"  1. Block Match:     {result.block_match_score:.4f}  (area-based coverage)")
        print(f"  2. Text:            {result.text_score:.4f}  (content similarity)")
        print(f"  3. Position:        {result.position_score:.4f}  (layout similarity)")
        print(f"  4. Color:           {result.color_score:.4f}  (color similarity)")
        print(f"  5. CLIP:            {result.clip_score:.4f}  (visual similarity)")
        print()

        print(f"Overall Score:        {result.overall_score:.4f}")
        print(f"  Formula: 0.2 × (Block Match + Text + Position + Color + CLIP)")
        print(f"  Calculation: 0.2 × ({result.block_match_score:.4f} + {result.text_score:.4f} + {result.position_score:.4f} + {result.color_score:.4f} + {result.clip_score:.4f})")
        print(f"  Result: {result.overall_score:.4f}")
        print()

        # Additional details if available
        if result.details:
            print("Additional Details:")
            if "matched_area" in result.details and "total_area" in result.details:
                print(f"  - Matched area: {result.details['matched_area']:.4f}")
                print(f"  - Total area: {result.details['total_area']:.4f}")

            if "text_scores" in result.details:
                text_scores = result.details["text_scores"]
                if text_scores:
                    print(f"  - Text scores: min={min(text_scores):.4f}, max={max(text_scores):.4f}, avg={sum(text_scores)/len(text_scores):.4f}")

            if "position_scores" in result.details:
                pos_scores = result.details["position_scores"]
                if pos_scores:
                    print(f"  - Position scores: min={min(pos_scores):.4f}, max={max(pos_scores):.4f}, avg={sum(pos_scores)/len(pos_scores):.4f}")

            if "color_scores" in result.details:
                color_scores = result.details["color_scores"]
                if color_scores:
                    print(f"  - Color scores: min={min(color_scores):.4f}, max={max(color_scores):.4f}, avg={sum(color_scores)/len(color_scores):.4f}")
            print()

        print("=" * 80)
        print("COMPARISON FORMAT (for Design2Code)")
        print("=" * 80)
        print()
        print("Metric Results:")
        print(f"block_match: {result.block_match_score:.4f}")
        print(f"text: {result.text_score:.4f}")
        print(f"position: {result.position_score:.4f}")
        print(f"color: {result.color_score:.4f}")
        print(f"clip: {result.clip_score:.4f}")
        print(f"overall: {result.overall_score:.4f}")
        print()
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

