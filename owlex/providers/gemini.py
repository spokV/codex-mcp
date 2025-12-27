"""Gemini CLI provider for owlex."""

import asyncio
import os
import re
import shutil
from datetime import datetime

from .protocol import Provider, ProviderConfig, ProviderResult


class GeminiProvider:
    """Provider for Gemini CLI integration."""

    def __init__(self, config: ProviderConfig | None = None):
        self._config = config or ProviderConfig(
            name="gemini",
            type="cli",
            default_model="gemini-3-auto",
        )
        self._yolo_mode = os.environ.get("GEMINI_YOLO_MODE", "true").lower() == "true"
        self._clean_output_enabled = os.environ.get("GEMINI_CLEAN_OUTPUT", "true").lower() == "true"

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def provider_type(self) -> str:
        return "cli"

    @property
    def is_available(self) -> bool:
        """Check if gemini CLI is available."""
        return shutil.which("gemini") is not None

    @property
    def config(self) -> ProviderConfig:
        return self._config

    def _clean_output(self, raw_output: str, original_prompt: str = "") -> str:
        """Clean Gemini CLI output by removing noise."""
        if not self._clean_output_enabled:
            return raw_output
        cleaned = raw_output
        # Remove YOLO mode warning if present
        if cleaned.startswith("YOLO mode is enabled."):
            lines = cleaned.split("\n", 2)
            if len(lines) > 2:
                cleaned = lines[2]
            elif len(lines) > 1:
                cleaned = lines[1]
        # Remove "Loaded cached credentials." line
        cleaned = re.sub(r"^Loaded cached credentials\.\n?", "", cleaned, flags=re.MULTILINE)
        # Collapse triple+ newlines
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    async def call(
        self,
        prompt: str,
        model: str | None = None,
        working_directory: str | None = None,
        **kwargs,
    ) -> ProviderResult:
        """Execute a new Gemini session."""
        start_time = datetime.now()

        # Build command: gemini [options] "prompt"
        full_command = ["gemini"]

        if self._yolo_mode:
            full_command.extend(["--approval-mode", "yolo"])

        if working_directory:
            working_directory = os.path.expanduser(working_directory)
            full_command.extend(["--include-directories", working_directory])

        full_command.append(prompt)

        process = await asyncio.create_subprocess_exec(
            *full_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_directory,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self._config.timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError(f"Gemini exec timed out after {self._config.timeout} seconds")

        stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""

        duration = (datetime.now() - start_time).total_seconds()

        if process.returncode != 0:
            error_output = []
            if stdout_text.strip():
                error_output.append(f"stdout:\n{stdout_text}")
            if stderr_text.strip():
                error_output.append(f"stderr:\n{stderr_text}")
            raise RuntimeError(
                f"Gemini exec failed (exit code {process.returncode}):\n\n"
                + ("\n\n".join(error_output) or "No output")
            )

        cleaned_output = self._clean_output(stdout_text, prompt)

        return ProviderResult(
            content=cleaned_output,
            provider_name=self.name,
            model_name=model or self._config.default_model or "gemini",
            duration_seconds=duration,
        )

    async def resume(
        self,
        prompt: str,
        session_ref: str = "latest",
        model: str | None = None,
        working_directory: str | None = None,
        **kwargs,
    ) -> ProviderResult:
        """Resume an existing Gemini session."""
        start_time = datetime.now()

        # Build command: gemini [options] -r <session> "prompt"
        full_command = ["gemini"]

        if self._yolo_mode:
            full_command.extend(["--approval-mode", "yolo"])

        if working_directory:
            working_directory = os.path.expanduser(working_directory)
            full_command.extend(["--include-directories", working_directory])

        # Add resume flag
        full_command.extend(["-r", session_ref])
        full_command.append(prompt)

        process = await asyncio.create_subprocess_exec(
            *full_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_directory,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self._config.timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError(f"Gemini resume timed out after {self._config.timeout} seconds")

        stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""

        duration = (datetime.now() - start_time).total_seconds()

        if process.returncode != 0:
            error_output = []
            if stdout_text.strip():
                error_output.append(f"stdout:\n{stdout_text}")
            if stderr_text.strip():
                error_output.append(f"stderr:\n{stderr_text}")
            raise RuntimeError(
                f"Gemini resume failed (exit code {process.returncode}):\n\n"
                + ("\n\n".join(error_output) or "No output")
            )

        cleaned_output = self._clean_output(stdout_text, prompt)

        return ProviderResult(
            content=cleaned_output,
            provider_name=self.name,
            model_name=model or self._config.default_model or "gemini",
            duration_seconds=duration,
        )
