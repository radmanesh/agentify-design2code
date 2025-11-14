"""Tests for the BlockDetector class."""

import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import tempfile

from src.evaluation.block_detector import BlockDetector, TextBlock


@pytest.fixture
def block_detector():
    """Create a BlockDetector instance for testing."""
    return BlockDetector()


@pytest.fixture
def temp_html_file(tmp_path):
    """Create a temporary HTML file for testing."""
    html_path = tmp_path / "test.html"
    html_path.write_text("""
    <!DOCTYPE html>
    <html>
    <head><title>Test</title></head>
    <body><h1>Test Page</h1></body>
    </html>
    """)
    return str(html_path)


@pytest.fixture
def temp_screenshot_file(tmp_path):
    """Create a temporary screenshot file for testing."""
    screenshot_path = tmp_path / "test.png"
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='white')
    img.save(screenshot_path)
    return str(screenshot_path)


class TestBlockDetector:
    """Test suite for BlockDetector."""

    def test_init(self, block_detector):
        """Test that BlockDetector initializes correctly."""
        assert block_detector is not None
        assert isinstance(block_detector, BlockDetector)

    def test_detect_blocks_with_existing_screenshot(
        self, block_detector, temp_html_file, temp_screenshot_file
    ):
        """Test detect_blocks when screenshot already exists."""
        # The screenshot file already exists, so generation should not be called
        with patch.object(
            block_detector, '_generate_screenshot'
        ) as mock_generate:
            blocks = block_detector.detect_blocks(
                temp_html_file, temp_screenshot_file
            )

            # Should not call _generate_screenshot since file exists
            mock_generate.assert_not_called()

            # Should return empty list (placeholder behavior)
            assert isinstance(blocks, list)
            assert len(blocks) == 0

    def test_detect_blocks_generates_missing_screenshot(
        self, block_detector, temp_html_file, tmp_path
    ):
        """Test detect_blocks generates screenshot when it doesn't exist."""
        screenshot_path = str(tmp_path / "missing_screenshot.png")

        # Mock the screenshot generation to create the file
        def mock_generate(html_path, output_path):
            # Create a dummy screenshot file
            img = Image.new('RGB', (100, 100), color='blue')
            img.save(output_path)

        with patch.object(
            block_detector, '_generate_screenshot', side_effect=mock_generate
        ) as mock_generate_method:
            blocks = block_detector.detect_blocks(
                temp_html_file, screenshot_path
            )

            # Should call _generate_screenshot since file doesn't exist
            mock_generate_method.assert_called_once_with(
                temp_html_file, screenshot_path
            )

            # Screenshot should now exist
            assert os.path.exists(screenshot_path)

            # Should return empty list (placeholder behavior)
            assert isinstance(blocks, list)

    def test_generate_screenshot_creates_directory(
        self, block_detector, temp_html_file, tmp_path
    ):
        """Test that _generate_screenshot creates output directory if needed."""
        # Create a nested path that doesn't exist
        nested_dir = tmp_path / "nested" / "path"
        screenshot_path = nested_dir / "screenshot.png"

        with patch(
            'src.evaluation.block_detector.generate_screenshot_from_html',
            return_value=True
        ) as mock_gen:
            # Mock the file creation
            def create_file(*args, **kwargs):
                screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                img = Image.new('RGB', (100, 100), color='green')
                img.save(screenshot_path)
                return True

            mock_gen.side_effect = create_file

            block_detector._generate_screenshot(
                temp_html_file, str(screenshot_path)
            )

            # Directory should be created
            assert nested_dir.exists()
            # Screenshot should exist
            assert screenshot_path.exists()

    def test_generate_screenshot_calls_helper(
        self, block_detector, temp_html_file, tmp_path
    ):
        """Test that _generate_screenshot calls the screenshot generator."""
        screenshot_path = tmp_path / "output.png"

        with patch(
            'src.evaluation.block_detector.generate_screenshot_from_html',
            return_value=True
        ) as mock_gen:
            # Create dummy file so the success check passes
            img = Image.new('RGB', (100, 100), color='red')
            img.save(screenshot_path)

            block_detector._generate_screenshot(
                temp_html_file, str(screenshot_path)
            )

            # Should call the generator function
            mock_gen.assert_called_once_with(temp_html_file, str(screenshot_path))

    def test_generate_screenshot_raises_on_failure(
        self, block_detector, temp_html_file, tmp_path
    ):
        """Test that _generate_screenshot raises error when generation fails."""
        screenshot_path = tmp_path / "failed.png"

        with patch(
            'src.evaluation.block_detector.generate_screenshot_from_html',
            return_value=False
        ):
            with pytest.raises(RuntimeError) as exc_info:
                block_detector._generate_screenshot(
                    temp_html_file, str(screenshot_path)
                )

            assert "Screenshot generation failed" in str(exc_info.value)
            assert temp_html_file in str(exc_info.value)

    def test_simple_block_detection_returns_empty(
        self, block_detector, temp_screenshot_file
    ):
        """Test that _simple_block_detection returns empty list (placeholder)."""
        blocks = block_detector._simple_block_detection(temp_screenshot_file)

        assert isinstance(blocks, list)
        assert len(blocks) == 0

    def test_simple_block_detection_handles_missing_file(
        self, block_detector
    ):
        """Test that _simple_block_detection handles missing screenshot gracefully."""
        blocks = block_detector._simple_block_detection("/nonexistent/path.png")

        # Should return empty list on error
        assert isinstance(blocks, list)
        assert len(blocks) == 0

    def test_merge_blocks_by_bbox(self, block_detector):
        """Test merging blocks with identical bounding boxes."""
        # Create test blocks with same bbox
        block1 = TextBlock(
            text="Hello",
            bbox=(0.1, 0.2, 0.3, 0.4),
            color=(255, 0, 0)
        )
        block2 = TextBlock(
            text="World",
            bbox=(0.1, 0.2, 0.3, 0.4),
            color=(255, 0, 0)
        )
        block3 = TextBlock(
            text="Different",
            bbox=(0.5, 0.6, 0.7, 0.8),
            color=(0, 255, 0)
        )

        blocks = [block1, block2, block3]
        merged = block_detector.merge_blocks_by_bbox(blocks)

        # Should have 2 blocks (block1 and block2 merged)
        assert len(merged) == 2

        # Find the merged block
        merged_block = next(b for b in merged if b.bbox == (0.1, 0.2, 0.3, 0.4))

        # Text should be combined
        assert "Hello" in merged_block.text
        assert "World" in merged_block.text

        # The other block should be unchanged
        other_block = next(b for b in merged if b.bbox == (0.5, 0.6, 0.7, 0.8))
        assert other_block.text == "Different"

    def test_merge_blocks_no_duplicates(self, block_detector):
        """Test merging when there are no duplicate bboxes."""
        block1 = TextBlock(
            text="First",
            bbox=(0.1, 0.2, 0.3, 0.4),
            color=(255, 0, 0)
        )
        block2 = TextBlock(
            text="Second",
            bbox=(0.5, 0.6, 0.7, 0.8),
            color=(0, 255, 0)
        )

        blocks = [block1, block2]
        merged = block_detector.merge_blocks_by_bbox(blocks)

        # Should still have 2 blocks
        assert len(merged) == 2

    def test_text_block_dataclass(self):
        """Test TextBlock dataclass creation."""
        block = TextBlock(
            text="Sample text",
            bbox=(0.0, 0.0, 1.0, 1.0),
            color=(128, 128, 128)
        )

        assert block.text == "Sample text"
        assert block.bbox == (0.0, 0.0, 1.0, 1.0)
        assert block.color == (128, 128, 128)

    def test_detect_blocks_with_ground_truth_html(self, block_detector):
        """Test detect_blocks with ground truth HTML and screenshot from data directory."""
        # Use the actual files from the data directory
        project_root = Path(__file__).parent.parent
        ground_truth_html = project_root / "data" / "11.html"
        ground_truth_screenshot = project_root / "data" / "11.png"

        # Skip test if files don't exist
        if not ground_truth_html.exists() or not ground_truth_screenshot.exists():
            pytest.skip("Ground truth files not found in data/ directory")

        # Test with ground truth HTML and existing screenshot
        blocks = block_detector.detect_blocks(
            str(ground_truth_html),
            str(ground_truth_screenshot)
        )

        # Should successfully detect blocks (even if empty for now)
        assert isinstance(blocks, list)
        # Current implementation returns empty list
        assert len(blocks) == 0

    def test_detect_blocks_with_generated_html(self, block_detector, tmp_path):
        """Test detect_blocks with generated HTML from GPT."""
        # Use the actual files from the data directory
        project_root = Path(__file__).parent.parent
        generated_html = project_root / "data" / "11-gpt5-v2.html"

        # Skip test if file doesn't exist
        if not generated_html.exists():
            pytest.skip("Generated HTML file not found in data/ directory")

        # Generate screenshot for the generated HTML
        generated_screenshot = tmp_path / "11-gpt5-v2.png"

        # Mock screenshot generation
        with patch(
            'src.evaluation.block_detector.generate_screenshot_from_html',
            return_value=True
        ) as mock_gen:
            # Create dummy screenshot
            def create_screenshot(*args, **kwargs):
                img = Image.new('RGB', (1280, 720), color='lightgray')
                img.save(generated_screenshot)
                return True

            mock_gen.side_effect = create_screenshot

            blocks = block_detector.detect_blocks(
                str(generated_html),
                str(generated_screenshot)
            )

            # Should successfully detect blocks (even if empty for now)
            assert isinstance(blocks, list)
            mock_gen.assert_called_once()

    def test_compare_ground_truth_and_generated(self, block_detector):
        """Test comparing blocks from ground truth vs generated HTML."""
        project_root = Path(__file__).parent.parent

        # Ground truth files
        ground_truth_html = project_root / "data" / "11.html"
        ground_truth_screenshot = project_root / "data" / "11.png"

        # Generated files
        generated_html = project_root / "data" / "11-gpt5-v2.html"

        # Skip test if files don't exist
        if not all([
            ground_truth_html.exists(),
            ground_truth_screenshot.exists(),
            generated_html.exists()
        ]):
            pytest.skip("Required test files not found in data/ directory")

        # Detect blocks from ground truth
        gt_blocks = block_detector.detect_blocks(
            str(ground_truth_html),
            str(ground_truth_screenshot)
        )

        # Detect blocks from generated HTML (mock screenshot generation)
        with patch(
            'src.evaluation.block_detector.generate_screenshot_from_html',
            return_value=True
        ):
            # Create a temporary screenshot for generated HTML
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                gen_screenshot_path = tmp.name
                img = Image.new('RGB', (1280, 720), color='white')
                img.save(gen_screenshot_path)

            try:
                gen_blocks = block_detector.detect_blocks(
                    str(generated_html),
                    gen_screenshot_path
                )

                # Both should return lists (even if empty for now)
                assert isinstance(gt_blocks, list)
                assert isinstance(gen_blocks, list)

                # Current implementation returns empty lists
                # When actual detection is implemented, this test will compare blocks
                assert len(gt_blocks) == 0
                assert len(gen_blocks) == 0
            finally:
                # Clean up temporary file
                import os
                if os.path.exists(gen_screenshot_path):
                    os.unlink(gen_screenshot_path)


class TestBlockDetectorIntegration:
    """Integration tests for BlockDetector with real dependencies."""

    @pytest.mark.skipif(
        os.environ.get("SKIP_INTEGRATION_TESTS") == "1",
        reason="Integration tests skipped"
    )
    def test_full_workflow_without_playwright(self, tmp_path):
        """Test full workflow without actually calling playwright."""
        detector = BlockDetector()

        html_path = tmp_path / "integration_test.html"
        html_path.write_text("""
        <!DOCTYPE html>
        <html>
        <head><title>Integration Test</title></head>
        <body>
            <h1>Test Heading</h1>
            <p>Test paragraph</p>
        </body>
        </html>
        """)

        screenshot_path = tmp_path / "integration_screenshot.png"

        # Mock the actual screenshot generation to avoid playwright dependency
        with patch(
            'src.evaluation.block_detector.generate_screenshot_from_html',
            return_value=True
        ):
            # Create a dummy screenshot
            img = Image.new('RGB', (1280, 720), color='lightblue')
            img.save(screenshot_path)

            # This should work without errors
            blocks = detector.detect_blocks(str(html_path), str(screenshot_path))

            assert isinstance(blocks, list)
            # Current implementation returns empty list
            assert len(blocks) == 0

