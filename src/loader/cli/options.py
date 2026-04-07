"""Reusable CLI argument helpers for Loader."""

from __future__ import annotations


def inject_resume_target(argv: list[str]) -> list[str]:
    """Rewrite `--resume [session-id]` into a hidden click option."""

    rewritten: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg != "--resume":
            rewritten.append(arg)
            index += 1
            continue

        next_arg = argv[index + 1] if index + 1 < len(argv) else None
        if next_arg and not next_arg.startswith("-"):
            rewritten.extend(["--resume-target", next_arg])
            index += 2
        else:
            rewritten.extend(["--resume-target", "__latest__"])
            index += 1
    return rewritten
