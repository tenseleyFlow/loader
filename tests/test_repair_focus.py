from __future__ import annotations

from pathlib import Path

from loader.llm.base import Message, Role
from loader.runtime.repair_focus import extract_active_repair_context


def test_extract_active_repair_context_parses_write_next_step_target(
    tmp_path: Path,
) -> None:
    repair_target = tmp_path / "guides" / "nginx" / "chapters" / "02-configuration.html"
    context = extract_active_repair_context(
        [
            Message(
                role=Role.ASSISTANT,
                content=(
                    "Repair focus:\n"
                    f"- Continue the declared output set by creating missing planned artifact `{repair_target}`.\n"
                    f"- Immediate next step: write `{repair_target}`.\n"
                    "- Do not rewrite existing aggregate files to match the partial artifact set while these declared outputs are still missing.\n"
                ),
            )
        ]
    )

    assert context is not None
    assert context.artifact_path == str(repair_target.resolve(strict=False))
    assert str(repair_target.resolve(strict=False)) in context.allowed_paths
