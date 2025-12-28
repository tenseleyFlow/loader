"""Streaming text widget for LLM responses."""

from rich.text import Text

from textual.reactive import reactive
from textual.widgets import Static


class StreamingText(Static):
    """Widget that displays streaming text content with cursor."""

    content: reactive[str] = reactive("")
    is_streaming: reactive[bool] = reactive(False)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._content_buffer = ""

    def render(self) -> Text:
        """Render the content with optional cursor."""
        # Use Text object to avoid markup interpretation of LLM output
        # Clean any tool_call tags that slipped through filtering
        content = self._clean_tool_tags(self._content_buffer)
        text = Text(content)
        if self.is_streaming:
            text.append("|", style="dim")  # Cursor indicator
        return text

    def _clean_tool_tags(self, content: str) -> str:
        """Remove any tool_call/think tags that weren't filtered during streaming."""
        import re
        # Remove <tool_call>...</tool_call> blocks
        content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL | re.IGNORECASE)
        # Remove orphaned tags
        content = re.sub(r'</?tool_call>', '', content, flags=re.IGNORECASE)
        content = re.sub(r'</?think>', '', content, flags=re.IGNORECASE)
        # Clean up excess newlines from removed blocks
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content

    def append(self, chunk: str) -> None:
        """Append a chunk to the content."""
        self._content_buffer += chunk
        self.refresh(layout=True)  # Force layout recalculation for multi-line

    def start_streaming(self) -> None:
        """Start streaming mode."""
        self._content_buffer = ""
        self.is_streaming = True
        self.add_class("streaming")
        self.refresh(layout=True)

    def stop_streaming(self) -> None:
        """Stop streaming mode."""
        self.is_streaming = False
        self.remove_class("streaming")
        self.refresh(layout=True)

    def get_content(self) -> str:
        """Get the accumulated content."""
        return self._content_buffer

    def clear(self) -> None:
        """Clear the content."""
        self._content_buffer = ""
        self.refresh()
