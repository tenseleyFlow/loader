"""Tests for tool implementations."""

import asyncio

import pytest

from loader.tools import (
    BashTool,
    ConfirmationRequired,
    EditTool,
    GlobTool,
    GrepTool,
    ReadTool,
    WriteTool,
)
from loader.tools.base import create_default_registry


class TestReadTool:
    """Tests for ReadTool."""

    @pytest.fixture
    def tool(self):
        return ReadTool()

    @pytest.mark.asyncio
    async def test_read_file(self, tool, sample_file):
        result = await tool.execute(file_path=str(sample_file))
        assert not result.is_error
        assert "Line 1" in result.output
        assert "Line 2" in result.output

    @pytest.mark.asyncio
    async def test_read_nonexistent(self, tool, temp_dir):
        result = await tool.execute(file_path=str(temp_dir / "nonexistent.txt"))
        assert result.is_error
        assert "not found" in result.output.lower()

    @pytest.mark.asyncio
    async def test_read_with_offset(self, tool, sample_file):
        result = await tool.execute(file_path=str(sample_file), offset=2, limit=1)
        assert not result.is_error
        assert "Line 2" in result.output
        assert "Line 1" not in result.output

    def test_is_not_destructive(self, tool):
        assert not tool.is_destructive


class TestWriteTool:
    """Tests for WriteTool."""

    @pytest.fixture
    def tool(self):
        return WriteTool()

    @pytest.mark.asyncio
    async def test_write_file(self, tool, temp_dir):
        file_path = temp_dir / "new_file.txt"
        result = await tool.execute(file_path=str(file_path), content="Hello, world!")
        assert not result.is_error
        assert file_path.exists()
        assert file_path.read_text() == "Hello, world!"

    @pytest.mark.asyncio
    async def test_write_creates_parents(self, tool, temp_dir):
        file_path = temp_dir / "subdir" / "deep" / "file.txt"
        result = await tool.execute(file_path=str(file_path), content="nested")
        assert not result.is_error
        assert file_path.exists()

    def test_is_destructive(self, tool):
        assert tool.is_destructive

    def test_requires_confirmation(self, tool):
        with pytest.raises(ConfirmationRequired) as exc_info:
            tool.check_confirmation(
                skip_confirmation=False,
                file_path="/tmp/test.txt",
                content="test",
            )
        assert "Write to file" in exc_info.value.message

    def test_skip_confirmation(self, tool):
        # Should not raise
        tool.check_confirmation(
            skip_confirmation=True,
            file_path="/tmp/test.txt",
            content="test",
        )


class TestEditTool:
    """Tests for EditTool."""

    @pytest.fixture
    def tool(self):
        return EditTool()

    @pytest.mark.asyncio
    async def test_edit_file(self, tool, sample_file):
        result = await tool.execute(
            file_path=str(sample_file),
            old_string="Line 2",
            new_string="Modified Line 2",
        )
        assert not result.is_error
        assert "Modified Line 2" in sample_file.read_text()

    @pytest.mark.asyncio
    async def test_edit_nonexistent(self, tool, temp_dir):
        result = await tool.execute(
            file_path=str(temp_dir / "nonexistent.txt"),
            old_string="foo",
            new_string="bar",
        )
        assert result.is_error

    @pytest.mark.asyncio
    async def test_edit_string_not_found(self, tool, sample_file):
        result = await tool.execute(
            file_path=str(sample_file),
            old_string="Not in file",
            new_string="replacement",
        )
        assert result.is_error
        assert "not found" in result.output.lower()


class TestGlobTool:
    """Tests for GlobTool."""

    @pytest.fixture
    def tool(self):
        return GlobTool()

    @pytest.mark.asyncio
    async def test_glob_finds_files(self, tool, temp_dir):
        (temp_dir / "file1.py").write_text("# python")
        (temp_dir / "file2.py").write_text("# python")
        (temp_dir / "file3.txt").write_text("text")

        result = await tool.execute(pattern="*.py", path=str(temp_dir))
        assert not result.is_error
        assert "file1.py" in result.output
        assert "file2.py" in result.output
        assert "file3.txt" not in result.output

    @pytest.mark.asyncio
    async def test_glob_no_matches(self, tool, temp_dir):
        result = await tool.execute(pattern="*.xyz", path=str(temp_dir))
        assert not result.is_error
        assert "No files matching" in result.output

    @pytest.mark.asyncio
    async def test_glob_expands_home_prefixed_pattern(self, tool, monkeypatch, temp_dir):
        home_dir = temp_dir / "fake-home"
        animals_dir = home_dir / "Loader" / "animals"
        animals_dir.mkdir(parents=True)
        (animals_dir / "penguins.html").write_text("<h1>Penguins</h1>\n")
        (animals_dir / "wolves.html").write_text("<h1>Wolves</h1>\n")

        monkeypatch.setenv("HOME", str(home_dir))

        result = await tool.execute(pattern="~/Loader/animals/*.html")

        assert not result.is_error
        assert "penguins.html" in result.output
        assert "wolves.html" in result.output
        assert result.metadata["base_path"] == str(animals_dir.resolve())
        assert result.metadata["effective_pattern"] == "*.html"


class TestBashTool:
    """Tests for BashTool."""

    @pytest.fixture
    def tool(self):
        return BashTool()

    @pytest.mark.asyncio
    async def test_bash_simple_command(self, tool):
        result = await tool.execute(command="echo 'hello world'")
        assert not result.is_error
        assert "hello world" in result.output

    @pytest.mark.asyncio
    async def test_bash_pwd(self, tool):
        result = await tool.execute(command="pwd")
        assert not result.is_error
        assert "/" in result.output

    @pytest.mark.asyncio
    async def test_bash_failed_command(self, tool):
        result = await tool.execute(command="exit 1")
        assert result.is_error
        assert "Exit code 1" in result.output

    @pytest.mark.asyncio
    async def test_bash_background_launch_and_wait(self, tool):
        launch = await tool.execute(
            command='python -c "import time; print(\'ready\'); time.sleep(0.1)"',
            background=True,
        )

        assert not launch.is_error
        job_id = launch.metadata["job_id"]
        assert job_id.startswith("bash-")

        await asyncio.sleep(0.05)
        wait_result = await tool.manager.wait_for_job(job_id)

        assert not wait_result.is_error
        assert "ready" in wait_result.output
        assert wait_result.metadata["status"] == "completed"

    @pytest.mark.asyncio
    async def test_bash_background_job_can_be_killed(self, tool):
        launch = await tool.execute(
            command='python -c "import time; print(\'server\'); time.sleep(30)"',
            background=True,
        )
        job_id = launch.metadata["job_id"]

        kill_result = await tool.manager.kill_job(job_id)

        assert not kill_result.is_error
        assert f"bash job {job_id}" in kill_result.output
        assert kill_result.metadata["status"] == "killed"
        assert kill_result.metadata["interrupted"] is False
        assert kill_result.metadata["killed"] is True

    @pytest.mark.asyncio
    async def test_bash_rejects_long_running_foreground_command(self, tool):
        result = await tool.execute(command="python -m http.server 8000")

        assert result.is_error
        assert "background=true" in result.output
        assert result.metadata["suggest_background"] is True

    def test_is_destructive(self, tool):
        assert tool.is_destructive

    def test_safe_command_no_confirmation(self, tool):
        # ls is safe
        tool.check_confirmation(skip_confirmation=False, command="ls -la")
        # git status is safe
        tool.check_confirmation(skip_confirmation=False, command="git status")

    def test_unsafe_command_requires_confirmation(self, tool):
        with pytest.raises(ConfirmationRequired):
            tool.check_confirmation(skip_confirmation=False, command="rm -rf /tmp/test")


class TestGrepTool:
    """Tests for GrepTool."""

    @pytest.fixture
    def tool(self):
        return GrepTool()

    @pytest.mark.asyncio
    async def test_grep_finds_pattern(self, tool, sample_python_file):
        result = await tool.execute(
            pattern="def.*hello",
            path=str(sample_python_file),
        )
        assert not result.is_error
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_grep_no_matches(self, tool, sample_file):
        result = await tool.execute(
            pattern="nonexistent_pattern",
            path=str(sample_file),
        )
        assert not result.is_error
        assert "No matches" in result.output


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_create_default_registry(self):
        registry = create_default_registry()
        assert registry.get("read") is not None
        assert registry.get("write") is not None
        assert registry.get("edit") is not None
        assert registry.get("glob") is not None
        assert registry.get("bash") is not None
        assert registry.get("grep") is not None

    def test_unknown_tool(self):
        registry = create_default_registry()
        assert registry.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        registry = create_default_registry()
        result = await registry.execute("nonexistent")
        assert result.is_error
        assert "Unknown tool" in result.output

    def test_skip_confirmation_flag(self):
        registry = create_default_registry()
        assert not registry.skip_confirmation
        registry.skip_confirmation = True
        assert registry.skip_confirmation
