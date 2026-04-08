"""Runtime-owned task classification helpers."""

from __future__ import annotations


def is_conversational(message: str) -> bool:
    """Detect if a message is conversational rather than a task."""

    msg = message.lower().strip()

    if len(msg) < 15:
        greetings = [
            "hi",
            "hello",
            "hey",
            "yo",
            "sup",
            "hiya",
            "howdy",
            "ello",
            "hallo",
            "greetings",
            "good morning",
            "good afternoon",
            "good evening",
            "morning",
            "evening",
            "afternoon",
            "what's up",
            "whats up",
            "wassup",
            "how are you",
            "how's it going",
            "hows it going",
        ]
        if any(msg.startswith(greeting) or msg == greeting for greeting in greetings):
            return True

    agent_questions = [
        "who are you",
        "what are you",
        "what can you do",
        "how do you work",
        "what is loader",
        "what's loader",
        "help",
        "what is this",
        "how does this work",
    ]
    if any(question in msg for question in agent_questions):
        return True

    casual = [
        "thanks",
        "thank you",
        "thx",
        "ty",
        "cool",
        "nice",
        "great",
        "awesome",
        "ok",
        "okay",
        "bye",
        "goodbye",
        "see you",
        "later",
        "cya",
        "lol",
        "haha",
        "hehe",
        "lmao",
        "please",
        "sorry",
        "oops",
    ]
    if msg in casual or any(msg == item for item in casual):
        return True

    task_indicators = [
        "create",
        "make",
        "build",
        "write",
        "edit",
        "delete",
        "remove",
        "run",
        "execute",
        "install",
        "fix",
        "debug",
        "test",
        "check",
        "find",
        "search",
        "show",
        "list",
        "read",
        "open",
        "close",
        "add",
        "update",
        "change",
        "modify",
        "refactor",
        "implement",
        "file",
        "folder",
        "directory",
        "code",
        "function",
        "class",
        "git",
        "npm",
        "pip",
        "python",
        "node",
        "bash",
        "command",
    ]
    if any(indicator in msg for indicator in task_indicators):
        return False

    if len(msg) < 30 and not any(char in msg for char in [".", "/", "\\", "`"]):
        return True

    return False


def estimate_complexity(message: str) -> str:
    """Estimate query complexity for token budgeting."""

    msg = message.lower()
    word_count = len(message.split())

    if is_conversational(message) or word_count < 5:
        return "trivial"

    complex_indicators = [
        "project",
        "application",
        "website",
        "api",
        "database",
        "refactor",
        "migrate",
        "upgrade",
        "implement",
        "design",
        "multiple",
        "several",
        "all",
        "entire",
        "whole",
        "and then",
        "after that",
        "also",
        "as well",
    ]
    complex_count = sum(1 for indicator in complex_indicators if indicator in msg)

    if complex_count >= 2 or word_count > 50:
        return "complex"

    simple_indicators = [
        "what is",
        "how do",
        "show me",
        "list",
        "read",
        "single",
        "one",
        "just",
        "only",
        "quick",
    ]
    if any(indicator in msg for indicator in simple_indicators) and word_count < 20:
        return "simple"

    return "moderate"


def get_token_budget(complexity: str) -> tuple[int, int]:
    """Get `(max_tokens, context_tokens)` for a complexity level."""

    budgets = {
        "trivial": (256, 2048),
        "simple": (512, 4096),
        "moderate": (1024, 8192),
        "complex": (2048, 16384),
    }
    return budgets.get(complexity, (1024, 8192))
