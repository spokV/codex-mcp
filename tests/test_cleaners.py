"""
Tests for output cleaning functions.
"""

import pytest
from owlex.engine import clean_codex_output, clean_gemini_output


class TestCleanCodexOutput:
    """Tests for clean_codex_output function."""

    def test_removes_echoed_prompt(self):
        """Should remove the echoed prompt from the beginning."""
        prompt = "What is 2+2?"
        raw = "What is 2+2?\n\nThe answer is 4."
        result = clean_codex_output(raw, prompt)
        assert result == "The answer is 4."

    def test_preserves_output_without_prompt(self):
        """Should preserve output when prompt isn't echoed."""
        raw = "The answer is 4."
        result = clean_codex_output(raw, "What is 2+2?")
        assert result == "The answer is 4."

    def test_collapses_multiple_newlines(self):
        """Should collapse 3+ newlines to 2."""
        raw = "Line 1\n\n\n\nLine 2"
        result = clean_codex_output(raw, "")
        assert result == "Line 1\n\nLine 2"

    def test_strips_whitespace(self):
        """Should strip leading/trailing whitespace."""
        raw = "  \n  Output  \n  "
        result = clean_codex_output(raw, "")
        assert result == "Output"

    def test_empty_input(self):
        """Should handle empty input."""
        result = clean_codex_output("", "")
        assert result == ""

    def test_prompt_only(self):
        """Should return empty if output is just the prompt."""
        prompt = "Hello"
        result = clean_codex_output(prompt, prompt)
        assert result == ""


class TestCleanGeminiOutput:
    """Tests for clean_gemini_output function."""

    def test_removes_yolo_warning(self):
        """Should remove YOLO mode warning."""
        raw = "YOLO mode is enabled.\nSome warning\nActual output"
        result = clean_gemini_output(raw, "")
        assert result == "Actual output"

    def test_removes_cached_credentials(self):
        """Should remove cached credentials line."""
        raw = "Loaded cached credentials.\nActual output"
        result = clean_gemini_output(raw, "")
        assert result == "Actual output"

    def test_collapses_multiple_newlines(self):
        """Should collapse 3+ newlines to 2."""
        raw = "Line 1\n\n\n\nLine 2"
        result = clean_gemini_output(raw, "")
        assert result == "Line 1\n\nLine 2"

    def test_strips_whitespace(self):
        """Should strip leading/trailing whitespace."""
        raw = "  \n  Output  \n  "
        result = clean_gemini_output(raw, "")
        assert result == "Output"

    def test_empty_input(self):
        """Should handle empty input."""
        result = clean_gemini_output("", "")
        assert result == ""

    def test_preserves_normal_output(self):
        """Should preserve output without noise."""
        raw = "This is the actual response from Gemini."
        result = clean_gemini_output(raw, "")
        assert result == "This is the actual response from Gemini."

    def test_removes_multiple_noise_lines(self):
        """Should remove both YOLO warning and cached credentials."""
        raw = "YOLO mode is enabled.\nWarning line\nLoaded cached credentials.\nActual output"
        result = clean_gemini_output(raw, "")
        # YOLO removes first 2 lines, then cached credentials is removed
        assert "Actual output" in result
        assert "YOLO" not in result
