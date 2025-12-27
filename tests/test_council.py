"""
Tests for council orchestration logic with mocked engine.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from owlex.council import Council
from owlex.models import Task, TaskStatus
from owlex.engine import codex_runner, gemini_runner


@pytest.fixture
def mock_engine():
    """Create a mock engine for testing council logic."""
    with patch("owlex.council.engine") as mock:
        # Mock create_task to return proper Task objects
        task_counter = [0]

        def create_task(command, args, context=None):
            task_counter[0] += 1
            return Task(
                task_id=f"test-task-{task_counter[0]}",
                status=TaskStatus.PENDING.value,
                command=command,
                args=args,
                start_time=datetime.now(),
                context=context,
            )

        mock.create_task = MagicMock(side_effect=create_task)

        # Mock kill_task_subprocess
        mock.kill_task_subprocess = AsyncMock()

        yield mock


@pytest.fixture
def mock_config():
    """Mock config to control enable_search setting."""
    with patch("owlex.council.config") as mock:
        mock.codex.enable_search = True
        mock.default_timeout = 300
        yield mock


class TestCouncilRound1:
    """Tests for round 1 parallel execution."""

    async def test_runs_both_agents_in_parallel(self, mock_engine, mock_config):
        """Should run Codex and Gemini in parallel."""
        call_order = []

        async def mock_run_agent(task, runner, mode="exec", **kwargs):
            agent_name = runner.name
            call_order.append(f"{agent_name}_start")
            await asyncio.sleep(0.05)
            task.status = TaskStatus.COMPLETED.value
            task.result = f"{agent_name.title()} Output:\n\n{agent_name.title()} response"
            task.completion_time = datetime.now()
            call_order.append(f"{agent_name}_end")

        mock_engine.run_agent = mock_run_agent

        council = Council()
        response = await council.deliberate(
            prompt="Test question",
            deliberate=False,
            timeout=10,
        )

        # Both should start before either ends (parallel execution)
        assert "codex_start" in call_order[:2]
        assert "gemini_start" in call_order[:2]
        assert response.round_1 is not None
        assert response.round_1.codex.status == "completed"
        assert response.round_1.gemini.status == "completed"

    async def test_handles_timeout(self, mock_engine, mock_config):
        """Should handle agent timeout gracefully."""
        async def mock_run_agent(task, runner, mode="exec", **kwargs):
            if runner.name == "codex":
                await asyncio.sleep(10)  # Longer than timeout
                task.status = TaskStatus.COMPLETED.value
            else:
                task.status = TaskStatus.COMPLETED.value
                task.result = "Gemini Output:\n\nGemini response"
                task.completion_time = datetime.now()

        mock_engine.run_agent = mock_run_agent

        council = Council()
        response = await council.deliberate(
            prompt="Test question",
            deliberate=False,
            timeout=1,
        )

        # Codex should have timed out
        assert response.round_1 is not None
        assert response.round_1.codex.status == "failed"
        # Gemini should succeed
        assert response.round_1.gemini.status == "completed"


class TestCouncilRound2:
    """Tests for round 2 deliberation."""

    async def test_deliberation_includes_round1_answers(self, mock_engine, mock_config):
        """Round 2 prompt should include round 1 answers."""
        round2_prompt = None

        async def mock_run_agent(task, runner, mode="exec", prompt="", **kwargs):
            nonlocal round2_prompt
            if "_delib" in task.command and runner.name == "codex":
                round2_prompt = prompt
            task.status = TaskStatus.COMPLETED.value
            task.result = f"{runner.name.title()} Output:\n\n{runner.name.title()} response"
            task.completion_time = datetime.now()

        mock_engine.run_agent = mock_run_agent

        council = Council()
        response = await council.deliberate(
            prompt="Test question",
            deliberate=True,
            critique=False,
            timeout=10,
        )

        assert response.round_2 is not None
        assert round2_prompt is not None
        assert "CODEX'S ANSWER:" in round2_prompt
        assert "GEMINI'S ANSWER:" in round2_prompt

    async def test_critique_mode_prompt(self, mock_engine, mock_config):
        """Critique mode should use critical analysis prompt."""
        round2_prompt = None

        async def mock_run_agent(task, runner, mode="exec", prompt="", **kwargs):
            nonlocal round2_prompt
            if "_delib" in task.command and runner.name == "codex":
                round2_prompt = prompt
            task.status = TaskStatus.COMPLETED.value
            task.result = f"{runner.name.title()} Output:\n\n{runner.name.title()} response"
            task.completion_time = datetime.now()

        mock_engine.run_agent = mock_run_agent

        council = Council()
        await council.deliberate(
            prompt="Test question",
            deliberate=True,
            critique=True,
            timeout=10,
        )

        assert round2_prompt is not None
        assert "senior reviewer" in round2_prompt.lower() or "critical" in round2_prompt.lower()
        assert "bugs" in round2_prompt.lower() or "flaws" in round2_prompt.lower()

    async def test_claude_opinion_included(self, mock_engine, mock_config):
        """Claude's opinion should be included in round 2."""
        round2_prompt = None

        async def mock_run_agent(task, runner, mode="exec", prompt="", **kwargs):
            nonlocal round2_prompt
            if "_delib" in task.command and runner.name == "codex":
                round2_prompt = prompt
            task.status = TaskStatus.COMPLETED.value
            task.result = f"{runner.name.title()} Output:\n\n{runner.name.title()} response"
            task.completion_time = datetime.now()

        mock_engine.run_agent = mock_run_agent

        council = Council()
        response = await council.deliberate(
            prompt="Test question",
            claude_opinion="Claude's expert analysis here",
            deliberate=True,
            timeout=10,
        )

        assert round2_prompt is not None
        assert "CLAUDE'S ANSWER:" in round2_prompt
        assert "Claude's expert analysis here" in round2_prompt
        assert response.claude_opinion is not None
        assert response.claude_opinion.content == "Claude's expert analysis here"


class TestCouncilMetadata:
    """Tests for council response metadata."""

    async def test_metadata_includes_timing(self, mock_engine, mock_config):
        """Metadata should include timing information."""
        async def mock_run_agent(task, runner, mode="exec", **kwargs):
            task.status = TaskStatus.COMPLETED.value
            task.result = f"{runner.name.title()} Output:\n\n{runner.name.title()} response"
            task.completion_time = datetime.now()

        mock_engine.run_agent = mock_run_agent

        council = Council()
        response = await council.deliberate(
            prompt="Test question",
            deliberate=True,
            timeout=10,
        )

        assert response.metadata is not None
        assert response.metadata.total_duration_seconds > 0
        assert response.metadata.rounds == 2
        assert len(response.metadata.log) > 0

    async def test_single_round_metadata(self, mock_engine, mock_config):
        """Single round should report rounds=1."""
        async def mock_run_agent(task, runner, mode="exec", **kwargs):
            task.status = TaskStatus.COMPLETED.value
            task.result = f"{runner.name.title()} Output:\n\n{runner.name.title()} response"
            task.completion_time = datetime.now()

        mock_engine.run_agent = mock_run_agent

        council = Council()
        response = await council.deliberate(
            prompt="Test question",
            deliberate=False,
            timeout=10,
        )

        assert response.metadata.rounds == 1


class TestCouncilConfig:
    """Tests for config integration."""

    async def test_uses_config_enable_search(self, mock_engine):
        """Should use config.codex.enable_search setting."""
        search_setting = None

        with patch("owlex.council.config") as mock_config:
            mock_config.codex.enable_search = False
            mock_config.default_timeout = 300

            async def mock_run_agent(task, runner, mode="exec", enable_search=False, **kwargs):
                nonlocal search_setting
                if runner.name == "codex":
                    search_setting = enable_search
                task.status = TaskStatus.COMPLETED.value
                task.result = f"{runner.name.title()} Output:\n\n{runner.name.title()} response"
                task.completion_time = datetime.now()

            mock_engine.run_agent = mock_run_agent

            council = Council()
            await council.deliberate(
                prompt="Test question",
                deliberate=False,
                timeout=10,
            )

            assert search_setting is False
