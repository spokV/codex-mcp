"""
Tests for configuration loading and validation.
"""

import os
from unittest.mock import patch

import pytest


class TestConfigParsing:
    """Tests for config environment variable parsing."""

    def test_default_timeout_valid(self):
        """Should parse valid timeout value."""
        with patch.dict(os.environ, {"OWLEX_DEFAULT_TIMEOUT": "600"}):
            # Need to reload config module to pick up new env var
            import importlib
            from owlex import config as config_module
            importlib.reload(config_module)

            assert config_module.config.default_timeout == 600

    def test_default_timeout_invalid_string(self, capsys):
        """Should fallback to 300 for invalid string."""
        with patch.dict(os.environ, {"OWLEX_DEFAULT_TIMEOUT": "not_a_number"}):
            import importlib
            from owlex import config as config_module
            importlib.reload(config_module)

            assert config_module.config.default_timeout == 300
            captured = capsys.readouterr()
            assert "Invalid OWLEX_DEFAULT_TIMEOUT" in captured.err

    def test_default_timeout_negative(self, capsys):
        """Should fallback to 300 for negative values."""
        with patch.dict(os.environ, {"OWLEX_DEFAULT_TIMEOUT": "-100"}):
            import importlib
            from owlex import config as config_module
            importlib.reload(config_module)

            assert config_module.config.default_timeout == 300
            captured = capsys.readouterr()
            assert "must be positive" in captured.err

    def test_default_timeout_zero(self, capsys):
        """Should fallback to 300 for zero."""
        with patch.dict(os.environ, {"OWLEX_DEFAULT_TIMEOUT": "0"}):
            import importlib
            from owlex import config as config_module
            importlib.reload(config_module)

            assert config_module.config.default_timeout == 300
            captured = capsys.readouterr()
            assert "must be positive" in captured.err

    def test_codex_bypass_approvals_true(self):
        """Should parse CODEX_BYPASS_APPROVALS=true."""
        with patch.dict(os.environ, {"CODEX_BYPASS_APPROVALS": "true"}):
            import importlib
            from owlex import config as config_module
            importlib.reload(config_module)

            assert config_module.config.codex.bypass_approvals is True

    def test_codex_bypass_approvals_false(self):
        """Should default to false for CODEX_BYPASS_APPROVALS."""
        with patch.dict(os.environ, {"CODEX_BYPASS_APPROVALS": "false"}, clear=False):
            import importlib
            from owlex import config as config_module
            importlib.reload(config_module)

            assert config_module.config.codex.bypass_approvals is False

    def test_codex_enable_search_default(self):
        """Should default to True for CODEX_ENABLE_SEARCH."""
        # Remove the env var if it exists
        env = os.environ.copy()
        env.pop("CODEX_ENABLE_SEARCH", None)
        with patch.dict(os.environ, env, clear=True):
            import importlib
            from owlex import config as config_module
            importlib.reload(config_module)

            assert config_module.config.codex.enable_search is True

    def test_gemini_yolo_mode_default(self):
        """Should default to False for GEMINI_YOLO_MODE."""
        env = os.environ.copy()
        env.pop("GEMINI_YOLO_MODE", None)
        with patch.dict(os.environ, env, clear=True):
            import importlib
            from owlex import config as config_module
            importlib.reload(config_module)

            assert config_module.config.gemini.yolo_mode is False
