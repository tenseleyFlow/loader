"""Internal semantic rule helpers for HTML table-of-contents repair tasks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

HTML_TOC_REPAIR_PATTERNS = (
    r"\bfix(?:ing|ed)?\b",
    r"\bcorrect(?:ing|ed)?\b",
    r"\brepair(?:ing|ed)?\b",
    r"\bupdate(?:d|s|ing)?\b",
    r"\bsync(?:hronize|hronized|ing)?\b",
    r"\balign(?:ed|ing)?\b",
    r"\bmatch(?:es|ed|ing)?\b",
    r"\bwrong\b",
    r"\bincorrect\b",
    r"\binaccurate\b",
    r"\bbroken\b",
    r"\bmismatche?d\b",
    r"\bmissing\b",
)
HTML_TOC_SUBJECT_HINTS = (
    "table of contents",
    "toc",
    "href",
    "hrefs",
    "link text",
    "chapter link",
    "chapter links",
    "chapter title",
    "chapter titles",
)
HTML_TOC_TARGET_HINTS = (
    "index.html",
    "index page",
    "index table of contents",
    "chapters/",
    "/chapters",
    "chapters directory",
    "chapter directory",
)


def task_targets_html_toc(task_text: str | None) -> bool:
    """Return True when task text clearly targets one HTML TOC repair flow."""

    lowered = str(task_text or "").strip().lower()
    if not lowered:
        return False
    has_subject = any(hint in lowered for hint in HTML_TOC_SUBJECT_HINTS)
    has_target = any(hint in lowered for hint in HTML_TOC_TARGET_HINTS)
    has_repair_intent = any(
        re.search(pattern, lowered) is not None for pattern in HTML_TOC_REPAIR_PATTERNS
    )
    return has_subject and has_target and has_repair_intent


def is_html_toc_index_path(path_value: str | Path) -> bool:
    """Return True when one path is the TOC index target."""

    path = Path(path_value).expanduser()
    return path.name == "index.html" and path.suffix.lower() in {".html", ".htm"}


def is_html_toc_chapters_dir(path_value: str | Path) -> bool:
    """Return True when one path is the sibling chapters directory."""

    return Path(path_value).expanduser().name == "chapters"


def is_html_toc_chapter_file(path_value: str | Path) -> bool:
    """Return True when one path is a chapter HTML file beside the TOC index."""

    path = Path(path_value).expanduser()
    return (
        path.suffix.lower() in {".html", ".htm"}
        and path.name != "index.html"
        and path.parent.name == "chapters"
    )


def resolve_html_toc_index_path(path_value: str | Path) -> Path | None:
    """Resolve a related TOC path back to its index target."""

    candidate = Path(path_value).expanduser()
    if is_html_toc_index_path(candidate):
        return candidate
    if is_html_toc_chapters_dir(candidate):
        return candidate.parent / "index.html"
    if is_html_toc_chapter_file(candidate):
        return candidate.parent.parent / "index.html"
    return None


def describe_html_toc_target(path_value: str | Path) -> str:
    """Return one model-facing label for the active TOC target."""

    index = resolve_html_toc_index_path(path_value)
    if index is None:
        return "`the target HTML table-of-contents page`"
    return f"`{index}`"


def describe_html_toc_chapters_dir(path_value: str | Path) -> str:
    """Return one model-facing label for the sibling chapter directory."""

    index = resolve_html_toc_index_path(path_value)
    if index is None:
        return "`the sibling chapter directory`"
    return f"`{index.parent / 'chapters'}`"


def extract_html_title_from_text(payload: str) -> str | None:
    """Extract one human-readable title from raw HTML text."""

    for pattern in (r"<h1[^>]*>(.*?)</h1>", r"<title[^>]*>(.*?)</title>"):
        match = re.search(pattern, payload, re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        title = re.sub(r"<[^>]+>", " ", match.group(1))
        normalized = " ".join(title.split()).strip()
        if normalized:
            return normalized
    return None


def read_html_title(path: Path) -> str:
    """Read one HTML file title for inventory and validation helpers."""

    try:
        return extract_html_title_from_text(path.read_text()) or ""
    except OSError:
        return ""


def format_html_inventory_entry(root: Path, candidate: Path) -> str:
    """Format one exact href/title pair for model-facing guidance."""

    normalized_root = root.expanduser().resolve(strict=False)
    normalized_candidate = candidate.expanduser().resolve(strict=False)
    try:
        href = str(normalized_candidate.relative_to(normalized_root))
    except ValueError:
        href = normalized_candidate.name
    title = read_html_title(candidate)
    if title:
        return f"{href} = {title}"
    return href


def build_validated_html_toc_observation_reason(path_value: str | Path) -> str:
    """Build a duplicate-observation reason for one already validated TOC target."""

    target = describe_html_toc_target(path_value)
    chapters_dir = describe_html_toc_chapters_dir(path_value)
    return (
        f"The HTML table-of-contents target {target} already passed semantic link "
        f"validation; reuse that result instead of rereading {target} or its sibling "
        f"chapter directory {chapters_dir} unless one specific href or label is still "
        "unresolved"
    )


def build_verified_html_inventory_observation_reason(path_value: str | Path) -> str:
    """Build a duplicate-observation reason for one verified chapter inventory."""

    target = describe_html_toc_target(path_value)
    chapters_dir = describe_html_toc_chapters_dir(path_value)
    return (
        f"The verified sibling chapter inventory for {chapters_dir} already contains the "
        f"exact href/title pairs needed for {target}; reuse that inventory instead of "
        "rereading chapter files"
    )


def _collect_html_inventory_entries(index_path: str | Path) -> list[tuple[str, str]]:
    """Return exact href/title pairs for sibling HTML chapters."""

    index = Path(index_path).expanduser()
    if not is_html_toc_index_path(index):
        return []

    chapters_dir = index.parent / "chapters"
    if not chapters_dir.is_dir():
        return []

    entries: list[tuple[str, str]] = []
    for candidate in sorted(chapters_dir.glob("*.html")):
        if not candidate.is_file():
            continue
        title = read_html_title(candidate)
        if not title:
            continue
        href = format_html_inventory_entry(index.parent, candidate).split(" = ", 1)[0]
        entries.append((href, title))
    return entries


def summarize_html_inventory(
    index_path: str | Path,
    *,
    limit: int | None = 12,
) -> str | None:
    """Summarize one sibling HTML inventory for an index page."""

    index = Path(index_path).expanduser()
    if not is_html_toc_index_path(index):
        return None

    entries = [f"{href} = {title}" for href, title in _collect_html_inventory_entries(index)]
    if not entries:
        return None

    if limit is not None and len(entries) > limit:
        return "; ".join(entries[:limit]) + "; ..."
    return "; ".join(entries)


def extract_html_toc_excerpt(
    index_path: str | Path,
    *,
    max_lines: int = 16,
) -> str | None:
    """Extract the current TOC block for recovery guidance."""

    index = Path(index_path).expanduser()
    if not is_html_toc_index_path(index):
        return None

    try:
        text = index.read_text()
    except OSError:
        return None

    match = re.search(
        r"(<h2[^>]*>\s*Table of Contents\s*</h2>.*?</ul>)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        match = re.search(
            r"(<ul[^>]*class=\"[^\"]*chapter-list[^\"]*\"[^>]*>.*?</ul>)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
    if not match:
        return None

    snippet_lines = [line.rstrip() for line in match.group(1).splitlines() if line.strip()]
    if not snippet_lines:
        return None
    if len(snippet_lines) > max_lines:
        snippet_lines = snippet_lines[:max_lines] + ["..."]
    return "\n".join(snippet_lines)


def build_html_toc_replacement_block(index_path: str | Path) -> str | None:
    """Build one exact replacement TOC block from the verified sibling inventory."""

    entries = _collect_html_inventory_entries(index_path)
    if not entries:
        return None

    excerpt = extract_html_toc_excerpt(index_path, max_lines=64)
    excerpt_lines = excerpt.splitlines() if excerpt else []

    heading_line = next(
        (line.rstrip() for line in excerpt_lines if "<h2" in line.lower()),
        "<h2>Table of Contents</h2>",
    )
    ul_line = next(
        (
            line.rstrip()
            for line in excerpt_lines
            if "<ul" in line.lower() and "chapter-list" in line.lower()
        ),
        '        <ul class="chapter-list">',
    )
    li_indent = next(
        (
            re.match(r"^\s*", line).group(0)
            for line in excerpt_lines
            if "<li><a " in line
        ),
        re.match(r"^\s*", ul_line).group(0) + "    ",
    )
    ul_indent = re.match(r"^\s*", ul_line).group(0)
    closing_line = next(
        (line.rstrip() for line in excerpt_lines if "</ul>" in line.lower()),
        f"{ul_indent}</ul>",
    )

    lines = [heading_line, ul_line]
    lines.extend(
        f'{li_indent}<li><a href="{href}">{title}</a></li>'
        for href, title in entries
    )
    lines.append(closing_line)
    return "\n".join(lines)


def build_html_toc_edit_call_template(index_path: str | Path) -> str | None:
    """Build one concrete edit template for replacing the TOC block."""

    index = Path(index_path).expanduser()
    excerpt = extract_html_toc_excerpt(index, max_lines=64)
    replacement = build_html_toc_replacement_block(index)
    if not excerpt or not replacement:
        return None

    return "\n".join(
        [
            "edit(",
            f'  file_path="{index}",',
            '  old_string="""',
            excerpt,
            '""",',
            '  new_string="""',
            replacement,
            '"""',
            ")",
        ]
    )


@dataclass(frozen=True)
class HtmlTocValidationResult:
    """Semantic validation result for one chapter-list table of contents."""

    valid: bool
    link_count: int
    missing: tuple[str, ...] = ()
    mismatched: tuple[str, ...] = ()


def validate_html_toc(index_path: str | Path) -> HtmlTocValidationResult | None:
    """Validate that one HTML TOC points at real chapter files with matching titles."""

    index = Path(index_path).expanduser()
    if not is_html_toc_index_path(index):
        return None

    try:
        text = index.read_text()
    except OSError:
        return None

    section_match = re.search(r'<ul class="chapter-list">(.*?)</ul>', text, re.S)
    if section_match is None:
        return HtmlTocValidationResult(
            valid=False,
            link_count=0,
            missing=("Missing chapter-list table of contents",),
        )

    links = re.findall(r'<a href="([^"]+)">([^<]+)</a>', section_match.group(1))
    if not links:
        return HtmlTocValidationResult(
            valid=False,
            link_count=0,
            missing=("No chapter links found in table of contents",),
        )

    root = index.parent
    missing: list[str] = []
    mismatched: list[str] = []
    for href, label in links:
        target = (root / href).expanduser().resolve(strict=False)
        if not target.exists():
            missing.append(f"{href} -> missing")
            continue
        title = read_html_title(target)
        if title and label.strip() != title:
            mismatched.append(f"{href} -> {label.strip()} != {title}")

    return HtmlTocValidationResult(
        valid=not missing and not mismatched,
        link_count=len(links),
        missing=tuple(missing),
        mismatched=tuple(mismatched),
    )


def build_html_toc_verification_command(index_path: str | Path) -> str | None:
    """Build the semantic verification command for one HTML TOC target."""

    index = Path(index_path).expanduser()
    if not is_html_toc_index_path(index):
        return None
    if not (index.parent / "chapters").is_dir():
        return None

    path_literal = repr(str(index))
    return "\n".join(
        [
            "python3 - <<'PY'",
            "from pathlib import Path",
            "import re",
            "import sys",
            "",
            f"index = Path({path_literal}).expanduser()",
            "root = index.parent",
            "text = index.read_text()",
            "section_match = re.search(r'<ul class=\"chapter-list\">(.*?)</ul>', text, re.S)",
            "if section_match is None:",
            "    print('Missing chapter-list table of contents', file=sys.stderr)",
            "    raise SystemExit(1)",
            "links = re.findall(r'<a href=\"([^\"]+)\">([^<]+)</a>', section_match.group(1))",
            "if not links:",
            "    print('No chapter links found in table of contents', file=sys.stderr)",
            "    raise SystemExit(1)",
            "",
            "missing = []",
            "mismatched = []",
            "for href, label in links:",
            "    target = (root / href).resolve()",
            "    if not target.exists():",
            "        missing.append(f'{href} -> missing')",
            "        continue",
            "    body = target.read_text()",
            "    match = re.search(r'<h1>(.*?)</h1>', body, re.S)",
            "    title = match.group(1).strip() if match else ''",
            "    if title and label.strip() != title:",
            "        mismatched.append(f'{href} -> {label.strip()} != {title}')",
            "",
            "if missing or mismatched:",
            "    if missing:",
            "        print('Missing links:', file=sys.stderr)",
            "        for item in missing:",
            "            print(item, file=sys.stderr)",
            "    if mismatched:",
            "        print('Title mismatches:', file=sys.stderr)",
            "        for item in mismatched:",
            "            print(item, file=sys.stderr)",
            "    raise SystemExit(1)",
            "",
            "print(f'validated {len(links)} toc links in {index.name}')",
            "PY",
        ]
    )


def parse_html_toc_verification_failures(text: str) -> tuple[list[str], list[str]]:
    """Parse missing hrefs and mismatched labels from verifier output."""

    missing: list[str] = []
    mismatches: list[str] = []
    mode: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered == "missing links:":
            mode = "missing"
            continue
        if lowered == "title mismatches:":
            mode = "mismatch"
            continue
        if mode == "missing" and "->" in line:
            href = line.split("->", 1)[0].strip()
            if href and href not in missing:
                missing.append(href)
            continue
        if mode == "mismatch" and "!=" in line and line not in mismatches:
            mismatches.append(line)

    return missing, mismatches


def summarize_html_toc_verification_gap(payload: str, *, max_items: int = 2) -> str | None:
    """Summarize the latest semantic verifier gap from shell output."""

    missing, mismatches = parse_html_toc_verification_failures(payload)

    parts: list[str] = []
    if missing:
        preview = ", ".join(missing[:max_items])
        if len(missing) > max_items:
            preview += ", ..."
        parts.append(f"missing TOC links {preview}")
    if mismatches:
        preview = ", ".join(mismatches[:max_items])
        if len(mismatches) > max_items:
            preview += ", ..."
        parts.append(f"title mismatches {preview}")
    return "; ".join(parts) if parts else None


def summarize_html_file_discovery(payload: str) -> str | None:
    """Summarize a set of discovered HTML filenames from tool output."""

    filenames = re.findall(r"([A-Za-z0-9_.-]+\.html)", payload)
    unique_names: list[str] = []
    for name in filenames:
        if name not in unique_names:
            unique_names.append(name)
    if len(unique_names) < 3:
        return None
    preview = ", ".join(unique_names[:6])
    if len(unique_names) > 6:
        preview += ", ..."
    return f"Existing files include {preview}"
