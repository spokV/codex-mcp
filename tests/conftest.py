"""
Pytest fixtures for owlex tests.
"""

import asyncio
import sys
from pathlib import Path

import pytest

# Add owlex to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from owlex.engine import TaskEngine


@pytest.fixture
def engine():
    """Create a fresh TaskEngine for each test."""
    return TaskEngine()


@pytest.fixture
def helper_script_dir(tmp_path):
    """Create directory with helper scripts for subprocess testing."""
    # Script that echoes stdin to stdout
    echo_script = tmp_path / "echo_stdin.py"
    echo_script.write_text('''
import sys
data = sys.stdin.read()
print(data, end="")
''')

    # Script that sleeps forever (for timeout tests)
    sleep_script = tmp_path / "sleep_forever.py"
    sleep_script.write_text('''
import time
time.sleep(3600)
''')

    # Script that exits with error
    error_script = tmp_path / "exit_error.py"
    error_script.write_text('''
import sys
print("error output", file=sys.stderr)
sys.exit(1)
''')

    # Script that outputs to both stdout and stderr
    mixed_output_script = tmp_path / "mixed_output.py"
    mixed_output_script.write_text('''
import sys
print("stdout line 1")
print("stderr line 1", file=sys.stderr)
print("stdout line 2")
''')

    # Script that sleeps briefly then outputs
    delayed_script = tmp_path / "delayed_output.py"
    delayed_script.write_text('''
import time
import sys
time.sleep(0.1)
print("delayed output")
''')

    return tmp_path
