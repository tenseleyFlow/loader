"""Durable project memory and working-notepad storage under `.loader/`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(slots=True)
class ProjectMemoryNote:
    """Categorized project-memory note."""

    category: str
    content: str
    timestamp: str

    def to_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "content": self.content,
            "timestamp": self.timestamp,
        }


@dataclass(slots=True)
class ProjectMemoryDirective:
    """Persistent user or project directive."""

    directive: str
    priority: str
    context: str | None
    timestamp: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "directive": self.directive,
            "priority": self.priority,
            "context": self.context,
            "timestamp": self.timestamp,
        }


class MemoryStore:
    """Manage project memory and the working notepad."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.loader_root = project_root / ".loader"
        self.project_memory_path = self.loader_root / "project-memory.json"
        self.notepad_path = self.loader_root / "notepad.md"

    def ensure_layout(self) -> None:
        """Ensure the `.loader/` directory exists."""

        self.loader_root.mkdir(parents=True, exist_ok=True)

    def read_project_memory(self, section: str = "all") -> Any:
        """Read the full project memory or one named section."""

        memory = self._load_project_memory()
        if section == "all":
            return memory
        return memory.get(section)

    def write_project_memory(
        self,
        memory: dict[str, Any],
        *,
        merge: bool = True,
    ) -> dict[str, Any]:
        """Write or merge project memory state."""

        existing = self._load_project_memory() if merge else {}
        merged = {**existing, **memory}
        self._save_project_memory(merged)
        return merged

    def add_project_note(self, category: str, content: str) -> dict[str, Any]:
        """Append one categorized note to project memory."""

        memory = self._load_project_memory()
        notes = list(memory.get("notes", []))
        note = ProjectMemoryNote(
            category=category.strip(),
            content=content.strip(),
            timestamp=_utc_now(),
        )
        notes.append(note.to_dict())
        memory["notes"] = notes
        self._save_project_memory(memory)
        return note.to_dict()

    def add_project_directive(
        self,
        directive: str,
        *,
        priority: str = "normal",
        context: str | None = None,
    ) -> dict[str, Any]:
        """Append one durable directive to project memory."""

        memory = self._load_project_memory()
        directives = list(memory.get("directives", []))
        directive_entry = ProjectMemoryDirective(
            directive=directive.strip(),
            priority=(priority or "normal").strip(),
            context=context.strip() if context else None,
            timestamp=_utc_now(),
        )
        directives.append(directive_entry.to_dict())
        memory["directives"] = directives
        self._save_project_memory(memory)
        return directive_entry.to_dict()

    def read_notepad(self, section: str = "all") -> str:
        """Read the full notepad or one named section."""

        sections = self._load_notepad_sections()
        if section == "all":
            return self._render_notepad(sections)
        return sections[section]

    def write_notepad_priority(self, content: str) -> str:
        """Replace the priority context section."""

        sections = self._load_notepad_sections()
        sections["priority"] = content.strip()
        self._save_notepad_sections(sections)
        return sections["priority"]

    def append_notepad_working(self, content: str) -> str:
        """Append one timestamped working-memory entry."""

        sections = self._load_notepad_sections()
        entry = f"- [{_utc_now()}] {content.strip()}"
        sections["working"] = self._append_markdown_entry(sections["working"], entry)
        self._save_notepad_sections(sections)
        return entry

    def append_notepad_manual(self, content: str) -> str:
        """Append one manual note that should not be auto-pruned."""

        sections = self._load_notepad_sections()
        entry = f"- {content.strip()}"
        sections["manual"] = self._append_markdown_entry(sections["manual"], entry)
        self._save_notepad_sections(sections)
        return entry

    def capture_definition_of_done(self, evidence_summary: str) -> dict[str, Any] | None:
        """Persist a useful evidence summary into project memory."""

        normalized = evidence_summary.strip()
        if not normalized or normalized == "Verification: skipped (no evidence required).":
            return None
        return self.add_project_note("definition_of_done", normalized)

    def _load_project_memory(self) -> dict[str, Any]:
        self.ensure_layout()
        if not self.project_memory_path.exists():
            return {}
        try:
            raw = json.loads(self.project_memory_path.read_text())
        except json.JSONDecodeError:
            return {}
        return raw if isinstance(raw, dict) else {}

    def _save_project_memory(self, memory: dict[str, Any]) -> None:
        self.ensure_layout()
        self.project_memory_path.write_text(json.dumps(memory, indent=2, sort_keys=True))

    def _load_notepad_sections(self) -> dict[str, str]:
        self.ensure_layout()
        sections = {
            "priority": "",
            "working": "",
            "manual": "",
        }
        if not self.notepad_path.exists():
            return sections

        current: str | None = None
        for line in self.notepad_path.read_text().splitlines():
            if line == "## Priority Context":
                current = "priority"
                continue
            if line == "## Working Memory":
                current = "working"
                continue
            if line == "## Manual Notes":
                current = "manual"
                continue
            if current is None or line == "# Loader Notepad":
                continue
            sections[current] = (
                f"{sections[current]}\n{line}".strip()
                if sections[current]
                else line
            )
        return sections

    def _save_notepad_sections(self, sections: dict[str, str]) -> None:
        self.ensure_layout()
        self.notepad_path.write_text(self._render_notepad(sections))

    @staticmethod
    def _append_markdown_entry(existing: str, entry: str) -> str:
        return f"{existing}\n{entry}".strip() if existing else entry

    @staticmethod
    def _render_notepad(sections: dict[str, str]) -> str:
        return "\n".join(
            [
                "# Loader Notepad",
                "",
                "## Priority Context",
                sections.get("priority", "").strip() or "_Empty_",
                "",
                "## Working Memory",
                sections.get("working", "").strip() or "_Empty_",
                "",
                "## Manual Notes",
                sections.get("manual", "").strip() or "_Empty_",
                "",
            ]
        )
