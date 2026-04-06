"""Tests for the ReAct parsing module."""

from loader.agent.parsing import format_tool_result, parse_tool_calls


class TestParseToolCalls:
    """Tests for parse_tool_calls function."""

    def test_parse_tool_call_xml_style(self):
        text = '''I need to read the file.
<tool_call>
{"name": "read", "arguments": {"file_path": "/tmp/test.txt"}}
</tool_call>
'''
        result = parse_tool_calls(text)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "read"
        assert result.tool_calls[0].arguments == {"file_path": "/tmp/test.txt"}
        assert not result.is_final_answer

    def test_parse_multiple_tool_calls(self):
        text = '''<tool_call>
{"name": "read", "arguments": {"file_path": "a.txt"}}
</tool_call>
<tool_call>
{"name": "read", "arguments": {"file_path": "b.txt"}}
</tool_call>'''
        result = parse_tool_calls(text)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].arguments["file_path"] == "a.txt"
        assert result.tool_calls[1].arguments["file_path"] == "b.txt"

    def test_parse_final_answer(self):
        text = '''Thought: I have all the information needed.
Final Answer: The file contains a hello world program.'''
        result = parse_tool_calls(text)
        assert result.is_final_answer
        assert len(result.tool_calls) == 0
        assert "hello world" in result.content.lower()

    def test_parse_no_tool_calls(self):
        text = "Just some regular text without any tool calls."
        result = parse_tool_calls(text)
        assert len(result.tool_calls) == 0
        assert not result.is_final_answer
        assert "regular text" in result.content

    def test_parse_bare_json(self):
        text = '''Let me read that file.
{"name": "read", "arguments": {"file_path": "/test.txt"}}'''
        result = parse_tool_calls(text)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "read"

    def test_parse_removes_react_labels(self):
        text = '''Thought: I need to check this.
Action: <tool_call>
{"name": "read", "arguments": {"file_path": "test.txt"}}
</tool_call>'''
        result = parse_tool_calls(text)
        assert "Thought:" not in result.content
        assert "Action:" not in result.content

    def test_parse_invalid_json_ignored(self):
        text = '''<tool_call>
{invalid json here}
</tool_call>'''
        result = parse_tool_calls(text)
        assert len(result.tool_calls) == 0

    def test_parse_empty_arguments(self):
        text = '''<tool_call>
{"name": "pwd", "arguments": {}}
</tool_call>'''
        result = parse_tool_calls(text)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {}

    def test_parse_parameters_alias(self):
        """Test that 'parameters' is accepted as alias for 'arguments'."""
        text = '''<tool_call>
{"name": "read", "parameters": {"file_path": "/tmp/test.txt"}}
</tool_call>'''
        result = parse_tool_calls(text)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "read"
        assert result.tool_calls[0].arguments == {"file_path": "/tmp/test.txt"}

    def test_parse_malformed_closing_tag(self):
        """Test handling of malformed </tool_call> at start."""
        text = '''</tool_call> {"name": "read", "parameters": {"file_path": "test.txt"}}
</tool_call>'''
        result = parse_tool_calls(text)
        # Should clean up the malformed tags
        assert "</tool_call>" not in result.content

    def test_parse_cleans_orphaned_tags(self):
        """Test that orphaned tool_call tags are removed from content."""
        text = '''Some text </tool_call> more text <tool_call> end'''
        result = parse_tool_calls(text)
        assert "<tool_call>" not in result.content
        assert "</tool_call>" not in result.content

    def test_parse_bracketed_calls_format(self):
        """Test parsing [calls tool with: key=value] format."""
        text = '''I'll create the file now.
[calls write tool with: file_path=/tmp/test.txt, content="hello world"]
Created the file.'''
        result = parse_tool_calls(text)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "write"
        assert result.tool_calls[0].arguments["file_path"] == "/tmp/test.txt"
        assert result.tool_calls[0].arguments["content"] == "hello world"
        # Bracketed call should be removed from content
        assert "[calls" not in result.content

    def test_parse_bracketed_use_format(self):
        """Test parsing [USE tool: key=value] format."""
        text = '[USE bash tool: command="ls -la"]'
        result = parse_tool_calls(text)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "bash"
        assert result.tool_calls[0].arguments["command"] == "ls -la"

    def test_parse_bracketed_edit_format(self):
        """Test parsing bracketed format with edit tool."""
        text = '[calls edit tool with: file_path="test.py", old_string="foo", new_string="bar"]'
        result = parse_tool_calls(text)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "edit"
        assert result.tool_calls[0].arguments["file_path"] == "test.py"
        assert result.tool_calls[0].arguments["old_string"] == "foo"
        assert result.tool_calls[0].arguments["new_string"] == "bar"


class TestFormatToolResult:
    """Tests for format_tool_result function."""

    def test_format_success(self):
        result = format_tool_result("read", "file contents here")
        assert "Observation" in result
        assert "read" in result
        assert "Result" in result
        assert "file contents here" in result

    def test_format_error(self):
        result = format_tool_result("write", "Permission denied", is_error=True)
        assert "Observation" in result
        assert "write" in result
        assert "Error" in result
        assert "Permission denied" in result
