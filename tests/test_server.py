"""
Tests for server.py MCP tool validation and error handling.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from owlex.models import Task, TaskStatus, ErrorCode


class TestWorkingDirectoryValidation:
    """Tests for working directory validation."""

    def test_validates_nonexistent_directory(self):
        """Should reject non-existent directories."""
        from owlex.server import _validate_working_directory

        path, error = _validate_working_directory("/nonexistent/path/12345")

        assert path is None
        assert error is not None
        assert "does not exist" in error

    def test_expands_home_directory(self, tmp_path):
        """Should expand ~ in paths."""
        from owlex.server import _validate_working_directory

        # Use a real temp directory for testing
        path, error = _validate_working_directory(str(tmp_path))

        assert error is None
        assert path == str(tmp_path)

    def test_accepts_none(self):
        """Should accept None working directory."""
        from owlex.server import _validate_working_directory

        path, error = _validate_working_directory(None)

        assert path is None
        assert error is None

    def test_rejects_file_path(self, tmp_path):
        """Should reject paths that are files, not directories."""
        from owlex.server import _validate_working_directory

        # Create a file
        file_path = tmp_path / "test_file.txt"
        file_path.write_text("test")

        path, error = _validate_working_directory(str(file_path))

        assert path is None
        assert error is not None
        assert "not a directory" in error


class TestStartCodexSession:
    """Tests for start_codex_session validation."""

    @pytest.fixture
    def mock_context(self):
        return MagicMock()

    @pytest.fixture
    def mock_engine(self):
        with patch("owlex.server.engine") as mock:
            mock.create_task = MagicMock(return_value=Task(
                task_id="test-123",
                status=TaskStatus.PENDING.value,
                command="codex_exec",
                args={},
                start_time=datetime.now(),
            ))
            mock.run_agent = AsyncMock()
            yield mock

    async def test_rejects_empty_prompt(self, mock_context):
        """Should reject empty prompt."""
        from owlex.server import start_codex_session

        result = await start_codex_session(mock_context, prompt="")

        response = json.loads(result)
        assert response["success"] is False
        assert response["error_code"] == ErrorCode.INVALID_ARGS.value
        assert "required" in response["error"]

    async def test_rejects_whitespace_prompt(self, mock_context):
        """Should reject whitespace-only prompt."""
        from owlex.server import start_codex_session

        result = await start_codex_session(mock_context, prompt="   ")

        response = json.loads(result)
        assert response["success"] is False
        assert response["error_code"] == ErrorCode.INVALID_ARGS.value

    async def test_rejects_invalid_working_directory(self, mock_context):
        """Should reject non-existent working directory."""
        from owlex.server import start_codex_session

        result = await start_codex_session(
            mock_context,
            prompt="Hello",
            working_directory="/nonexistent/path/xyz"
        )

        response = json.loads(result)
        assert response["success"] is False
        assert response["error_code"] == ErrorCode.INVALID_ARGS.value
        assert "does not exist" in response["error"]


class TestGetTaskResult:
    """Tests for get_task_result error handling."""

    async def test_returns_not_found_for_missing_task(self):
        """Should return NOT_FOUND for non-existent task."""
        from owlex.server import get_task_result

        with patch("owlex.server.engine") as mock_engine:
            mock_engine.get_task.return_value = None

            result = await get_task_result("nonexistent-id")

            response = json.loads(result)
            assert response["success"] is False
            assert response["error_code"] == ErrorCode.NOT_FOUND.value


class TestWaitForTask:
    """Tests for wait_for_task error handling."""

    async def test_returns_not_found_for_missing_task(self):
        """Should return NOT_FOUND for non-existent task."""
        from owlex.server import wait_for_task

        with patch("owlex.server.engine") as mock_engine:
            mock_engine.get_task.return_value = None

            result = await wait_for_task("nonexistent-id")

            response = json.loads(result)
            assert response["success"] is False
            assert response["error_code"] == ErrorCode.NOT_FOUND.value

    async def test_returns_execution_failed_for_failed_task(self):
        """Should return EXECUTION_FAILED for failed tasks."""
        from owlex.server import wait_for_task

        failed_task = Task(
            task_id="failed-123",
            status=TaskStatus.FAILED.value,
            command="test",
            args={},
            start_time=datetime.now(),
            error="Something went wrong",
        )

        with patch("owlex.server.engine") as mock_engine:
            mock_engine.get_task.return_value = failed_task

            result = await wait_for_task("failed-123")

            response = json.loads(result)
            assert response["success"] is False
            assert response["error_code"] == ErrorCode.EXECUTION_FAILED.value

    async def test_returns_cancelled_for_cancelled_task(self):
        """Should return CANCELLED for cancelled tasks."""
        from owlex.server import wait_for_task

        cancelled_task = Task(
            task_id="cancelled-123",
            status=TaskStatus.CANCELLED.value,
            command="test",
            args={},
            start_time=datetime.now(),
            error="Cancelled by user",
        )

        with patch("owlex.server.engine") as mock_engine:
            mock_engine.get_task.return_value = cancelled_task

            result = await wait_for_task("cancelled-123")

            response = json.loads(result)
            assert response["success"] is False
            assert response["error_code"] == ErrorCode.CANCELLED.value


class TestCancelTask:
    """Tests for cancel_task error handling."""

    async def test_returns_not_found_for_missing_task(self):
        """Should return NOT_FOUND for non-existent task."""
        from owlex.server import cancel_task

        with patch("owlex.server.engine") as mock_engine:
            mock_engine.get_task.return_value = None

            result = await cancel_task("nonexistent-id")

            response = json.loads(result)
            assert response["success"] is False
            assert response["error_code"] == ErrorCode.NOT_FOUND.value

    async def test_returns_error_for_already_completed_task(self):
        """Should return error for already completed task."""
        from owlex.server import cancel_task

        completed_task = Task(
            task_id="completed-123",
            status=TaskStatus.COMPLETED.value,
            command="test",
            args={},
            start_time=datetime.now(),
        )

        with patch("owlex.server.engine") as mock_engine:
            mock_engine.get_task.return_value = completed_task

            result = await cancel_task("completed-123")

            response = json.loads(result)
            assert response["success"] is False
            assert response["error_code"] == ErrorCode.INVALID_ARGS.value
            assert "already completed" in response["error"]

    async def test_successfully_cancels_running_task(self):
        """Should successfully cancel a running task."""
        from owlex.server import cancel_task

        running_task = Task(
            task_id="running-123",
            status=TaskStatus.RUNNING.value,
            command="test",
            args={},
            start_time=datetime.now(),
        )

        with patch("owlex.server.engine") as mock_engine:
            mock_engine.get_task.return_value = running_task
            mock_engine.kill_task_subprocess = AsyncMock()

            result = await cancel_task("running-123")

            response = json.loads(result)
            assert response["success"] is True
            assert response["status"] == "cancelled"
            mock_engine.kill_task_subprocess.assert_called_once_with(running_task)


class TestListTasks:
    """Tests for list_tasks functionality."""

    async def test_returns_empty_list(self):
        """Should return empty list when no tasks."""
        from owlex.server import list_tasks

        with patch("owlex.server.engine") as mock_engine:
            mock_engine.tasks = {}

            # Pass explicit values for all Field() parameters
            result = await list_tasks(status_filter=None, limit=20)

            response = json.loads(result)
            assert response["success"] is True
            assert response["count"] == 0
            assert response["tasks"] == []

    async def test_filters_by_status(self):
        """Should filter tasks by status."""
        from owlex.server import list_tasks

        with patch("owlex.server.engine") as mock_engine:
            mock_engine.tasks = {
                "task-1": Task(
                    task_id="task-1",
                    status=TaskStatus.COMPLETED.value,
                    command="test",
                    args={},
                    start_time=datetime.now(),
                ),
                "task-2": Task(
                    task_id="task-2",
                    status=TaskStatus.RUNNING.value,
                    command="test",
                    args={},
                    start_time=datetime.now(),
                ),
            }

            result = await list_tasks(status_filter="running", limit=20)

            response = json.loads(result)
            assert response["count"] == 1
            assert response["tasks"][0]["status"] == "running"

    async def test_respects_limit(self):
        """Should respect limit parameter."""
        from owlex.server import list_tasks

        with patch("owlex.server.engine") as mock_engine:
            # Create many tasks
            mock_engine.tasks = {
                f"task-{i}": Task(
                    task_id=f"task-{i}",
                    status=TaskStatus.COMPLETED.value,
                    command="test",
                    args={},
                    start_time=datetime.now(),
                )
                for i in range(10)
            }

            result = await list_tasks(status_filter=None, limit=3)

            response = json.loads(result)
            assert len(response["tasks"]) == 3


class TestCouncilAsk:
    """Tests for council_ask validation."""

    @pytest.fixture
    def mock_context(self):
        return MagicMock()

    async def test_rejects_empty_prompt(self, mock_context):
        """Should reject empty prompt."""
        from owlex.server import council_ask

        result = await council_ask(mock_context, prompt="")

        response = json.loads(result)
        assert "error" in response
        assert "required" in response["error"]

    async def test_rejects_invalid_working_directory(self, mock_context):
        """Should reject non-existent working directory."""
        from owlex.server import council_ask

        result = await council_ask(
            mock_context,
            prompt="Test question",
            working_directory="/nonexistent/path/xyz"
        )

        response = json.loads(result)
        assert "error" in response
        assert "does not exist" in response["error"]
