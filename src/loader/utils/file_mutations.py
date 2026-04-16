"""Shared file-mutation preview models and Rich render helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from ..tools.fs_safety import StructuredPatchHunk, make_structured_patch

FILE_MUTATION_TOOLS = {"write", "edit", "patch"}
DIFF_TRUNCATION_NOTICE = "truncated for display; full result preserved in session"


@dataclass(slots=True)
class FileMutationPreview:
    """Normalized preview data for file-mutation tools."""

    tool_name: str
    file_path: str
    operation: str
    structured_patch: list[StructuredPatchHunk]
    old_text: str | None = None
    new_text: str | None = None
    added_lines: int = 0
    removed_lines: int = 0
    context_lines: int = 0

    @property
    def hunk_count(self) -> int:
        return len(self.structured_patch)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the preview for runtime/UI event payloads."""

        return {
            "tool_name": self.tool_name,
            "file_path": self.file_path,
            "operation": self.operation,
            "structured_patch": [hunk.to_dict() for hunk in self.structured_patch],
            "old_text": self.old_text,
            "new_text": self.new_text,
            "added_lines": self.added_lines,
            "removed_lines": self.removed_lines,
            "context_lines": self.context_lines,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FileMutationPreview:
        """Deserialize one preview payload."""

        return cls(
            tool_name=str(payload.get("tool_name", "")),
            file_path=str(payload.get("file_path", "")),
            operation=str(payload.get("operation", "update")),
            structured_patch=_coerce_patch_hunks(payload.get("structured_patch")),
            old_text=_coerce_optional_text(payload.get("old_text")),
            new_text=_coerce_optional_text(payload.get("new_text")),
            added_lines=int(payload.get("added_lines", 0)),
            removed_lines=int(payload.get("removed_lines", 0)),
            context_lines=int(payload.get("context_lines", 0)),
        )


def is_file_mutation_tool(tool_name: str) -> bool:
    """Return whether the tool mutates file content."""

    return tool_name in FILE_MUTATION_TOOLS


def build_file_mutation_preview(
    tool_name: str,
    *,
    tool_args: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> FileMutationPreview | None:
    """Build a normalized preview from tool args and/or result metadata."""

    if not is_file_mutation_tool(tool_name):
        return None

    args = tool_args or {}
    info = metadata or {}
    file_path = _extract_file_path(info) or _extract_file_path(args)

    structured_patch = (
        _coerce_patch_hunks(info.get("structured_patch"))
        or _coerce_patch_hunks(info.get("structuredPatch"))
        or _coerce_patch_hunks(args.get("structured_patch"))
        or _coerce_patch_hunks(args.get("structuredPatch"))
    )
    if not structured_patch and tool_name == "patch":
        structured_patch = _coerce_patch_hunks(info.get("hunks")) or _coerce_patch_hunks(
            args.get("hunks")
        )

    old_text = _extract_old_text(tool_name, info) or _extract_old_text(tool_name, args)
    new_text = _extract_new_text(tool_name, info) or _extract_new_text(tool_name, args)

    if not structured_patch and tool_name in {"write", "edit"} and new_text is not None:
        structured_patch = make_structured_patch(old_text or "", new_text)

    if not structured_patch:
        return None

    operation = _determine_operation(tool_name, info, old_text)
    added_lines, removed_lines, context_lines = _count_patch_lines(structured_patch)
    return FileMutationPreview(
        tool_name=tool_name,
        file_path=file_path or "",
        operation=operation,
        structured_patch=structured_patch,
        old_text=old_text,
        new_text=new_text,
        added_lines=added_lines,
        removed_lines=removed_lines,
        context_lines=context_lines,
    )


def build_file_mutation_preview_dict(
    tool_name: str,
    *,
    tool_args: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build a serialized preview payload when possible."""

    preview = build_file_mutation_preview(
        tool_name,
        tool_args=tool_args,
        metadata=metadata,
    )
    return preview.to_dict() if preview is not None else None


def render_file_mutation_preview(
    preview: FileMutationPreview | dict[str, Any],
    *,
    border_style: str = "cyan",
    title: str = "Diff",
    max_lines: int = 40,
    max_chars: int = 6_000,
) -> Group:
    """Render one normalized preview as a Rich group."""

    resolved = (
        FileMutationPreview.from_dict(preview)
        if isinstance(preview, dict)
        else preview
    )
    return Group(
        _render_preview_summary(resolved),
        Panel(
            _render_preview_text(
                resolved,
                max_lines=max_lines,
                max_chars=max_chars,
            ),
            title=title,
            border_style=border_style,
            box=box.SQUARE,
            expand=True,
        ),
    )


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _coerce_patch_hunks(value: Any) -> list[StructuredPatchHunk]:
    if not isinstance(value, list):
        return []

    hunks: list[StructuredPatchHunk] = []
    for item in value:
        if isinstance(item, StructuredPatchHunk):
            hunks.append(item)
        elif isinstance(item, dict):
            hunks.append(StructuredPatchHunk.from_dict(item))
    return hunks


def _extract_file_path(payload: dict[str, Any]) -> str | None:
    for key in ("file_path", "filePath", "path", "filename", "file"):
        value = payload.get(key)
        if value:
            return str(value)
    return None


def _extract_old_text(tool_name: str, payload: dict[str, Any]) -> str | None:
    for key in ("original_file", "originalFile", "old_string", "oldString", "old"):
        value = payload.get(key)
        if value is not None:
            return str(value)
    if tool_name == "write":
        return ""
    return None


def _extract_new_text(tool_name: str, payload: dict[str, Any]) -> str | None:
    if tool_name == "write":
        for key in ("content", "contents", "text", "data"):
            value = payload.get(key)
            if value is not None:
                return str(value)
    for key in ("content", "new_string", "newString", "new", "replacement", "replace"):
        value = payload.get(key)
        if value is not None:
            return str(value)
    return None


def _determine_operation(tool_name: str, metadata: dict[str, Any], old_text: str | None) -> str:
    if tool_name == "patch":
        return "patch"
    kind = metadata.get("kind") or metadata.get("type")
    if kind in {"create", "update"}:
        return str(kind)
    if tool_name == "write":
        return "update" if old_text else "create"
    return "update"


def _count_patch_lines(hunks: list[StructuredPatchHunk]) -> tuple[int, int, int]:
    added = 0
    removed = 0
    context = 0
    for hunk in hunks:
        for raw_line in hunk.lines:
            prefix = raw_line[:1]
            if prefix == "+":
                added += 1
            elif prefix == "-":
                removed += 1
            elif prefix == " ":
                context += 1
    return added, removed, context


def _render_preview_summary(preview: FileMutationPreview) -> Text:
    action_label = {
        "create": "Create",
        "update": "Update",
        "patch": "Patch",
    }.get(preview.operation, "Update")
    action_style = {
        "create": "bold green",
        "update": "bold cyan",
        "patch": "bold yellow",
    }.get(preview.operation, "bold cyan")

    text = Text()
    filename = Path(preview.file_path).name if preview.file_path else "(unknown file)"
    text.append(action_label, style=action_style)
    text.append(f"({filename})")
    if preview.file_path:
        text.append(f"\n{preview.file_path}", style="dim")

    stats = []
    if preview.added_lines:
        stats.append(("+%d" % preview.added_lines, "green"))
    if preview.removed_lines:
        stats.append(("-%d" % preview.removed_lines, "red"))
    if preview.context_lines:
        stats.append(("%d context" % preview.context_lines, "dim"))
    stats.append(("%d hunk%s" % (preview.hunk_count, "" if preview.hunk_count == 1 else "s"), "dim"))

    text.append("\n")
    for index, (label, style) in enumerate(stats):
        if index:
            text.append("  ")
        text.append(label, style=style)
    return text


def _render_preview_text(
    preview: FileMutationPreview,
    *,
    max_lines: int,
    max_chars: int,
) -> Text:
    rendered_lines = _iter_rendered_patch_lines(preview.structured_patch)
    output = Text()
    line_count = 0
    char_count = 0
    truncated = False

    for line in rendered_lines:
        next_chars = len(line.plain)
        if line_count >= max_lines or char_count + next_chars > max_chars:
            truncated = True
            break
        output.append_text(line)
        line_count += 1
        char_count += next_chars

    if truncated:
        if output.plain and not output.plain.endswith("\n"):
            output.append("\n")
        output.append(f"... {DIFF_TRUNCATION_NOTICE}", style="dim")

    if not output.plain:
        output.append("No diff available", style="dim")

    return output


def _iter_rendered_patch_lines(
    hunks: list[StructuredPatchHunk],
) -> list[Text]:
    rendered: list[Text] = []
    for hunk in hunks:
        rendered.append(
            Text(
                "@@ -%d,%d +%d,%d @@\n"
                % (hunk.old_start, hunk.old_lines, hunk.new_start, hunk.new_lines),
                style="dim",
            )
        )
        old_line = hunk.old_start
        new_line = hunk.new_start
        for raw_line in hunk.lines:
            prefix = raw_line[:1] if raw_line[:1] in {" ", "+", "-"} else ""
            content = raw_line[1:] if prefix else raw_line
            display = Text()
            old_label = "    "
            new_label = "    "
            if prefix in {" ", "-"}:
                old_label = f"{old_line:>4}"
                old_line += 1
            if prefix in {" ", "+"}:
                new_label = f"{new_line:>4}"
                new_line += 1

            display.append(old_label, style="dim")
            display.append(" ")
            display.append(new_label, style="dim")
            display.append(" ")

            if prefix == "+":
                display.append("+ ", style="green")
                display.append(content, style="green")
            elif prefix == "-":
                display.append("- ", style="red")
                display.append(content, style="red")
            elif prefix == " ":
                display.append("  ", style="dim")
                display.append(content, style="dim")
            else:
                display.append("  ", style="dim")
                display.append(content)
            display.append("\n")
            rendered.append(display)
    return rendered
