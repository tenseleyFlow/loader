"""Pytest configuration and fixtures."""

import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_file(temp_dir):
    """Create a sample file for testing."""
    file_path = temp_dir / "sample.txt"
    file_path.write_text("Line 1\nLine 2\nLine 3\n")
    return file_path


@pytest.fixture
def sample_python_file(temp_dir):
    """Create a sample Python file for testing."""
    file_path = temp_dir / "sample.py"
    file_path.write_text(
        '''"""Sample module."""

def hello():
    """Say hello."""
    return "Hello, world!"

def add(a, b):
    """Add two numbers."""
    return a + b
'''
    )
    return file_path
