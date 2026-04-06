"""Runtime safeguards to improve agent behavior.

These safeguards help keep the agent on track when models don't follow
instructions perfectly. They work at runtime to filter, detect, and correct
problematic patterns.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FilterResult:
    """Result of filtering content."""
    content: str  # Filtered content
    was_filtered: bool  # Whether any filtering occurred
    removed_blocks: list[str] = field(default_factory=list)  # What was removed


class CodeBlockFilter:
    """Filters markdown code blocks and bracket tool calls from streamed content.

    Handles both complete blocks (```...```) and partial blocks that span
    multiple stream chunks. Also filters [calls X tool with ...] patterns.
    """

    def __init__(self):
        self._buffer = ""
        self._in_code_block = False
        self._block_lang = ""
        self._current_block = ""
        self._in_bracket = False
        self._bracket_content = ""
        self._in_json_tool = False
        self._json_brace_count = 0

    def reset(self):
        """Reset filter state."""
        self._buffer = ""
        self._in_code_block = False
        self._block_lang = ""
        self._current_block = ""
        self._in_bracket = False
        self._bracket_content = ""
        self._in_json_tool = False
        self._json_brace_count = 0

    def _is_bracket_tool_start(self, text: str) -> bool:
        """Check if text looks like start of a bracket tool call."""
        # Patterns like: [calls, [call, [USE
        return bool(re.match(r'\[(?:calls?|USE)\s', text, re.IGNORECASE))

    def filter_chunk(self, chunk: str) -> FilterResult:
        """Filter a streaming chunk, removing code blocks and bracket tool calls.

        Returns filtered content. Handles partial blocks across chunks.
        """
        if not chunk:
            return FilterResult(content="", was_filtered=False)

        result_parts = []
        removed = []
        was_filtered = False

        # Process character by character to handle streaming
        self._buffer += chunk

        while self._buffer:
            # Handle bracket tool calls: [calls X tool with ...]
            if self._in_bracket:
                # Look for closing ]
                end_idx = self._buffer.find(']')
                if end_idx >= 0:
                    self._bracket_content += self._buffer[:end_idx]
                    removed.append(f"[{self._bracket_content}]")
                    self._buffer = self._buffer[end_idx + 1:]
                    self._in_bracket = False
                    self._bracket_content = ""
                    was_filtered = True
                else:
                    # Still in bracket, consume all
                    self._bracket_content += self._buffer
                    self._buffer = ""
                    was_filtered = True
                continue

            # Check for bracket start: [calls, [USE, or [output (fake outputs)
            bracket_match = re.search(r'\[(?=(?:calls?|USE|output)\s*[:\s])', self._buffer, re.IGNORECASE)
            if bracket_match:
                # Output everything before the bracket
                result_parts.append(self._buffer[:bracket_match.start()])
                self._buffer = self._buffer[bracket_match.start() + 1:]  # Skip the [
                self._in_bracket = True
                was_filtered = True
                continue

            # Handle JSON tool calls: {"name": "write", "arguments": {...}}
            if self._in_json_tool:
                # Track braces to find the end
                for i, char in enumerate(self._buffer):
                    if char == '{':
                        self._json_brace_count += 1
                    elif char == '}':
                        self._json_brace_count -= 1
                        if self._json_brace_count == 0:
                            # Found end of JSON
                            removed.append(self._buffer[:i + 1])
                            self._buffer = self._buffer[i + 1:]
                            self._in_json_tool = False
                            was_filtered = True
                            break
                else:
                    # Still in JSON, consume all
                    self._buffer = ""
                    was_filtered = True
                continue

            # Check for JSON tool call start: {"name": "write" etc
            json_tool_match = re.search(
                r'\{\s*"name"\s*:\s*"(?:write|read|edit|bash|glob|grep)"',
                self._buffer
            )
            if json_tool_match:
                # Output everything before the JSON
                result_parts.append(self._buffer[:json_tool_match.start()])
                self._buffer = self._buffer[json_tool_match.start():]
                self._in_json_tool = True
                self._json_brace_count = 0  # Will count starting from {
                was_filtered = True
                continue

            # Check for hallucinated tool narration and filter the line
            hallucination_match = re.search(
                r'([Uu]sed\s+`?(?:bash|write|read|edit|glob|grep)`?\s+tool|'
                r'[Uu]sing\s+the\s+`?(?:bash|write|read|edit|glob|grep)`?\s+tool|'
                r'with\s+file_path\s*=\s*[`\'"]|'
                r'with\s+command\s*[`\'"]|'
                r'[Hh]ere\s+is\s+what\s+[Ii]\s+did:)',
                self._buffer
            )
            if hallucination_match:
                # Find end of line and remove whole line
                line_start = self._buffer.rfind('\n', 0, hallucination_match.start()) + 1
                line_end = self._buffer.find('\n', hallucination_match.end())
                if line_end == -1:
                    # Line continues to end of buffer - wait for more
                    if line_start > 0:
                        result_parts.append(self._buffer[:line_start])
                    self._buffer = self._buffer[line_start:]
                    break
                else:
                    # Remove the whole line
                    result_parts.append(self._buffer[:line_start])
                    removed.append(self._buffer[line_start:line_end])
                    self._buffer = self._buffer[line_end:]
                    was_filtered = True
                    continue

            # Check for preamble patterns and filter the line
            preamble_match = re.search(
                r'(Here is a JSON response|Here are the function calls|'
                r'Here is the response with|I will respond with|'
                r'The following JSON|Below is the)',
                self._buffer, re.IGNORECASE
            )
            if preamble_match:
                # Find end of line and remove whole line
                line_start = self._buffer.rfind('\n', 0, preamble_match.start()) + 1
                line_end = self._buffer.find('\n', preamble_match.end())
                if line_end == -1:
                    # Line continues to end of buffer - wait for more
                    if line_start > 0:
                        result_parts.append(self._buffer[:line_start])
                    self._buffer = self._buffer[line_start:]
                    break
                else:
                    # Remove the whole line
                    result_parts.append(self._buffer[:line_start])
                    removed.append(self._buffer[line_start:line_end])
                    self._buffer = self._buffer[line_end:]
                    was_filtered = True
                    continue
            if self._in_code_block:
                # Look for closing ```
                end_match = re.search(r'```', self._buffer)
                if end_match:
                    # Found end of code block
                    block_content = self._buffer[:end_match.start()]
                    self._current_block += block_content
                    removed.append(f"```{self._block_lang}\n{self._current_block}```")
                    self._buffer = self._buffer[end_match.end():]
                    self._in_code_block = False
                    self._current_block = ""
                    self._block_lang = ""
                    was_filtered = True
                else:
                    # Still in code block, consume all
                    self._current_block += self._buffer
                    self._buffer = ""
                    was_filtered = True
            else:
                # Look for opening ```
                start_match = re.search(r'```(\w*)\n?', self._buffer)
                if start_match:
                    # Found start of code block
                    # Output everything before the block
                    result_parts.append(self._buffer[:start_match.start()])
                    self._block_lang = start_match.group(1)
                    self._buffer = self._buffer[start_match.end():]
                    self._in_code_block = True
                    was_filtered = True
                else:
                    # Check if buffer ends with partial ``` marker
                    if self._buffer.endswith('`') or self._buffer.endswith('``'):
                        # Hold back potential partial marker
                        split_point = len(self._buffer) - self._buffer[::-1].index('`') - 1
                        if split_point > 0:
                            # Find where backticks start
                            for i in range(len(self._buffer) - 1, -1, -1):
                                if self._buffer[i] != '`':
                                    result_parts.append(self._buffer[:i+1])
                                    self._buffer = self._buffer[i+1:]
                                    break
                        break
                    else:
                        # No code block markers, output all
                        result_parts.append(self._buffer)
                        self._buffer = ""

        return FilterResult(
            content="".join(result_parts),
            was_filtered=was_filtered,
            removed_blocks=removed,
        )

    def filter_complete(self, content: str) -> FilterResult:
        """Filter complete content (non-streaming), removing code blocks, bracket tool calls, and preambles."""
        removed = []

        # Pattern to match code blocks
        code_pattern = r'```\w*\n?[\s\S]*?```'
        removed.extend(re.findall(code_pattern, content))
        filtered = re.sub(code_pattern, '', content)

        # Pattern to match bracket-format tool calls: [calls X tool with ...] and fake outputs
        bracket_patterns = [
            r'\[calls?\s+\w+\s+tool\s+with[:\s][^\]]+\]',
            r'\[USE\s+\w+\s+tool[:\s][^\]]+\]',
            r'\[output[:\s][^\]]+\]',  # Fake outputs from model
        ]
        for pattern in bracket_patterns:
            matches = re.findall(pattern, filtered, re.IGNORECASE)
            removed.extend(matches)
            filtered = re.sub(pattern, '', filtered, flags=re.IGNORECASE)

        # Pattern to match JSON tool calls: {"name": "write", "arguments": {...}}
        # Use a function to handle nested braces properly
        def remove_json_tool_calls(text: str) -> tuple[str, list[str]]:
            json_removed = []
            tool_pattern = r'\{\s*"name"\s*:\s*"(?:write|read|edit|bash|glob|grep)"'
            result = text
            while True:
                match = re.search(tool_pattern, result)
                if not match:
                    break
                # Find matching closing brace
                start = match.start()
                brace_count = 0
                end = start
                for i, char in enumerate(result[start:], start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
                if end > start:
                    json_removed.append(result[start:end])
                    result = result[:start] + result[end:]
                else:
                    break  # Couldn't find matching brace
            return result, json_removed

        filtered, json_matches = remove_json_tool_calls(filtered)
        removed.extend(json_matches)

        # Pattern to match preamble lines (remove entire line)
        preamble_patterns = [
            r'^.*Here is a JSON response.*$',
            r'^.*Here are the function calls.*$',
            r'^.*Here is the response with.*$',
            r'^.*I will respond with.*$',
            r'^.*The following (JSON|function calls|tool calls).*$',
            r'^.*Below (is|are) the (JSON|function|tool).*$',
        ]
        for pattern in preamble_patterns:
            matches = re.findall(pattern, filtered, re.IGNORECASE | re.MULTILINE)
            removed.extend(matches)
            filtered = re.sub(pattern, '', filtered, flags=re.IGNORECASE | re.MULTILINE)

        # Pattern to match hallucinated/narrated tool uses (remove entire line)
        # These are lines where model describes using tools instead of actually calling them
        hallucination_patterns = [
            r'^.*[Uu]sed\s+`?(?:bash|write|read|edit|glob|grep)`?\s+tool.*$',  # "Used bash tool..."
            r'^.*[Uu]sing\s+the\s+`?(?:bash|write|read|edit|glob|grep)`?\s+tool.*$',  # "...using the write tool"
            r'^.*with\s+file_path\s*=\s*[`\'"][^`\'"]+[`\'"].*$',  # Narrated file_path parameter
            r'^.*with\s+command\s*[`\'"][^`\'"]+[`\'"].*$',  # Narrated bash command
            r'^\s*\*\s*[Uu]sed\s+`.*$',  # "* Used `bash`..." (bullet point narration)
            r'^.*[Hh]ere\s+is\s+what\s+[Ii]\s+did:.*$',  # "Here is what I did:"
            r'^\s*\d+\.\s+[Uu]sed\s+.*tool.*$',  # "1. Used bash tool..."
            r'^\s*\d+\.\s+[Cc]reated\s+.*using\s+the\s+.*tool.*$',  # "1. Created... using the write tool"
        ]
        for pattern in hallucination_patterns:
            matches = re.findall(pattern, filtered, re.MULTILINE)
            removed.extend(matches)
            filtered = re.sub(pattern, '', filtered, flags=re.MULTILINE)

        # Filter internal recovery/system prompts (multiline blocks)
        internal_prompt_patterns = [
            # Recovery prompts
            r'## TOOL FAILURE - INVESTIGATE AND ADAPT[\s\S]*?What will you do\?',
            r'## REQUIRED: Choose ONE[\s\S]*?(?=\n\n|\Z)',
            r'## CRITICAL RULES:[\s\S]*?(?=\n\n|\Z)',
            r'## Current attempt:.*$',
            r'\*\*Your next action should gather information[\s\S]*?What will you do\?',
            # Observation prefixes
            r'^Observation \[[\w]+\]:.*$',
        ]
        for pattern in internal_prompt_patterns:
            matches = re.findall(pattern, filtered, re.MULTILINE)
            removed.extend(matches)
            filtered = re.sub(pattern, '', filtered, flags=re.MULTILINE)

        # Clean up multiple blank lines left behind
        filtered = re.sub(r'\n{3,}', '\n\n', filtered)

        return FilterResult(
            content=filtered.strip(),
            was_filtered=bool(removed),
            removed_blocks=removed,
        )


@dataclass
class PatternMatch:
    """A detected problematic pattern."""
    pattern_type: str  # 'code_block', 'narration', 'preview', 'repetition'
    match_text: str
    severity: str  # 'low', 'medium', 'high'


class PatternDetector:
    """Detects problematic patterns in agent output.

    Patterns include:
    - Code blocks (which should be tool calls instead)
    - Narration ("I will call...", "Now I'll...")
    - Previews ("The file will look like:", "After editing:")
    - Repetitive commands
    """

    # Narration patterns - model announcing what it will do instead of doing it
    NARRATION_PATTERNS = [
        (r"I('ll| will) (use|call|execute|run) the (\w+) tool", "narration", "high"),
        (r"Let me (use|call|execute|run) the (\w+) tool", "narration", "high"),
        (r"Now I('ll| will) (create|write|edit|run|execute)", "narration", "medium"),
        (r"I('m going to| am going to) (use|call|create|write)", "narration", "medium"),
        (r"First,? I('ll| will) (use|call|create)", "narration", "medium"),
        (r"Next,? I('ll| will) (use|call|create)", "narration", "medium"),
    ]

    # Preview patterns - model showing content instead of using tools
    PREVIEW_PATTERNS = [
        (r"(The|This) file will (look like|contain|have):", "preview", "high"),
        (r"After editing,? (the file|it) will (look like|contain):", "preview", "high"),
        (r"Here('s| is) (the|what) (content|code|file):", "preview", "high"),
        (r"Save this (to|as|in) [\w./]+:", "preview", "high"),
        (r"Create a file (with|containing):", "preview", "medium"),
        (r"(The|Your) [\w./]+ (should|will) (look like|contain):", "preview", "medium"),
    ]

    # Preamble patterns - model describing JSON/function calls instead of using them
    PREAMBLE_PATTERNS = [
        (r"Here is a JSON response", "preamble", "high"),
        (r"Here are the function calls", "preamble", "high"),
        (r"Here is the response with", "preamble", "high"),
        (r"I will respond with", "preamble", "high"),
        (r"The following (JSON|function calls|tool calls)", "preamble", "high"),
        (r"Below (is|are) the (JSON|function|tool)", "preamble", "high"),
    ]

    # Code block patterns
    CODE_BLOCK_PATTERNS = [
        (r'```\w+\n', "code_block", "high"),
        (r'```\n', "code_block", "medium"),
    ]

    def __init__(self):
        self._all_patterns = (
            self.NARRATION_PATTERNS +
            self.PREVIEW_PATTERNS +
            self.PREAMBLE_PATTERNS +
            self.CODE_BLOCK_PATTERNS
        )
        self._recent_detections: list[PatternMatch] = []

    def reset(self):
        """Reset detection state."""
        self._recent_detections = []

    def detect(self, content: str) -> list[PatternMatch]:
        """Detect problematic patterns in content."""
        matches = []

        for pattern, ptype, severity in self._all_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                matches.append(PatternMatch(
                    pattern_type=ptype,
                    match_text=match.group(0),
                    severity=severity,
                ))

        self._recent_detections.extend(matches)
        return matches

    def has_high_severity(self, content: str) -> bool:
        """Check if content has high-severity patterns."""
        matches = self.detect(content)
        return any(m.severity == "high" for m in matches)

    def get_steering_message(self, matches: list[PatternMatch]) -> str | None:
        """Generate a steering message based on detected patterns.

        Returns None if no steering needed.
        """
        if not matches:
            return None

        # Prioritize high severity
        high_severity = [m for m in matches if m.severity == "high"]
        if not high_severity:
            return None

        # Generate appropriate steering message
        pattern_types = set(m.pattern_type for m in high_severity)

        if "preamble" in pattern_types:
            return (
                "[STOP] Do not describe JSON or function calls. "
                "Just USE the tools directly. No preambles."
            )
        elif "code_block" in pattern_types or "preview" in pattern_types:
            return (
                "[REMINDER] Do not show code blocks or previews. "
                "Use tools directly to create/edit files. "
                "No ```code```, just call the tool."
            )
        elif "narration" in pattern_types:
            return (
                "[REMINDER] Don't announce tool calls. "
                "Just use the tool directly without narration."
            )

        return None


class ActionTracker:
    """Tracks completed actions to prevent duplicates and detect loops.

    Tracks:
    - Files created (by path)
    - Files edited (by path + edit signature)
    - Commands executed (by command string)
    - Directories created (by path)
    - Action sequence for loop detection
    - Response hashes for text loop detection
    """

    MAX_SEQUENCE_LENGTH = 20  # Track last N actions
    LOOP_PATTERN_MIN = 2  # Minimum pattern length to detect
    LOOP_REPEAT_THRESHOLD = 2  # How many times pattern must repeat
    MAX_RESPONSE_HISTORY = 5  # Track last N responses for text loops

    def __init__(self):
        self._file_writes: dict[str, list[str]] = {}
        self._files_edited: dict[str, list[str]] = {}  # path -> list of edit sigs
        self._commands_run: set[str] = set()
        self._dirs_created: set[str] = set()
        self._action_sequence: list[str] = []  # For loop detection
        self._response_history: list[str] = []  # For text loop detection

    def reset(self):
        """Reset all tracking."""
        self._file_writes.clear()
        self._files_edited.clear()
        self._commands_run.clear()
        self._dirs_created.clear()
        self._action_sequence.clear()
        self._response_history.clear()

    def _normalize_path(self, path: str) -> str:
        """Normalize a file path for comparison."""
        # Expand ~ and resolve to absolute
        expanded = Path(path).expanduser()
        try:
            return str(expanded.resolve())
        except Exception:
            return str(expanded)

    def _make_edit_signature(self, old_string: str, new_string: str) -> str:
        """Create a signature for an edit operation."""
        # Use hash of old+new to detect same edit
        return f"{hash(old_string)}:{hash(new_string)}"

    def _make_write_signature(self, content: str) -> str:
        """Create a signature for a write operation."""
        return str(hash(content))

    def would_duplicate_file_create(self, file_path: str, content: str) -> bool:
        """Check if writing the same content to a file would be a duplicate."""
        norm_path = self._normalize_path(file_path)
        sig = self._make_write_signature(content)
        return sig in self._file_writes.get(norm_path, [])

    def would_duplicate_edit(self, file_path: str, old_string: str, new_string: str) -> bool:
        """Check if this edit would be a duplicate."""
        norm_path = self._normalize_path(file_path)
        sig = self._make_edit_signature(old_string, new_string)
        return sig in self._files_edited.get(norm_path, [])

    def would_duplicate_command(self, command: str) -> bool:
        """Check if this command would be a duplicate."""
        # Normalize whitespace
        norm_cmd = " ".join(command.split())
        return norm_cmd in self._commands_run

    def would_duplicate_mkdir(self, dir_path: str) -> bool:
        """Check if creating this directory would be a duplicate."""
        norm_path = self._normalize_path(dir_path)
        return norm_path in self._dirs_created

    def record_file_create(self, file_path: str, content: str) -> None:
        """Record that a file write completed."""
        norm_path = self._normalize_path(file_path)
        sig = self._make_write_signature(content)
        if norm_path not in self._file_writes:
            self._file_writes[norm_path] = []
        self._file_writes[norm_path].append(sig)

    def record_edit(self, file_path: str, old_string: str, new_string: str) -> None:
        """Record that an edit was made."""
        norm_path = self._normalize_path(file_path)
        sig = self._make_edit_signature(old_string, new_string)
        if norm_path not in self._files_edited:
            self._files_edited[norm_path] = []
        self._files_edited[norm_path].append(sig)

    def record_command(self, command: str) -> None:
        """Record that a command was run."""
        norm_cmd = " ".join(command.split())
        self._commands_run.add(norm_cmd)

        # Also track mkdir commands specially
        mkdir_match = re.match(r'mkdir\s+(-p\s+)?(.+)', norm_cmd)
        if mkdir_match:
            dir_path = mkdir_match.group(2).strip().strip('"\'')
            self._dirs_created.add(self._normalize_path(dir_path))

    def record_mkdir(self, dir_path: str) -> None:
        """Record that a directory was created."""
        norm_path = self._normalize_path(dir_path)
        self._dirs_created.add(norm_path)

    def check_tool_call(self, tool_name: str, arguments: dict) -> tuple[bool, str]:
        """Check if a tool call would be a duplicate.

        Returns (is_duplicate, reason).
        """
        if tool_name == "write":
            file_path = arguments.get("file_path", "")
            content = arguments.get("content", "")
            if self.would_duplicate_file_create(file_path, content):
                return True, f"Same file content already written: {file_path}"

        elif tool_name == "edit":
            file_path = arguments.get("file_path", "")
            old_string = arguments.get("old_string", "")
            new_string = arguments.get("new_string", "")
            if self.would_duplicate_edit(file_path, old_string, new_string):
                return True, f"Same edit already applied to: {file_path}"

        elif tool_name == "bash":
            command = arguments.get("command", "")
            if self.would_duplicate_command(command):
                return True, f"Command already executed: {command[:50]}..."

        return False, ""

    def record_tool_call(self, tool_name: str, arguments: dict) -> None:
        """Record a tool call as completed."""
        # Track in action sequence for loop detection
        self._action_sequence.append(tool_name)
        if len(self._action_sequence) > self.MAX_SEQUENCE_LENGTH:
            self._action_sequence.pop(0)

        if tool_name == "write":
            file_path = arguments.get("file_path", "")
            content = arguments.get("content", "")
            if file_path:
                self.record_file_create(file_path, content)

        elif tool_name == "edit":
            file_path = arguments.get("file_path", "")
            old_string = arguments.get("old_string", "")
            new_string = arguments.get("new_string", "")
            if file_path:
                self.record_edit(file_path, old_string, new_string)

        elif tool_name == "bash":
            command = arguments.get("command", "")
            if command:
                self.record_command(command)

    def detect_loop(self) -> tuple[bool, str]:
        """Detect if the agent is in a repetitive loop.

        Returns (is_loop, pattern_description).
        """
        seq = self._action_sequence
        if len(seq) < self.LOOP_PATTERN_MIN * self.LOOP_REPEAT_THRESHOLD:
            return False, ""

        # Check for repeating patterns of length 2, 3, 4
        for pattern_len in range(self.LOOP_PATTERN_MIN, min(6, len(seq) // 2 + 1)):
            # Get the most recent pattern
            pattern = seq[-pattern_len:]

            # Count how many times this pattern appears consecutively
            repeats = 1
            for i in range(len(seq) - pattern_len * 2, -1, -pattern_len):
                if seq[i:i + pattern_len] == pattern:
                    repeats += 1
                else:
                    break

            if repeats >= self.LOOP_REPEAT_THRESHOLD:
                pattern_str = " → ".join(pattern)
                return True, f"Repeating pattern detected ({repeats}x): {pattern_str}"

        return False, ""

    def _normalize_response(self, response: str) -> str:
        """Normalize a response for comparison.

        Strips whitespace, lowercases, and takes first ~200 chars
        to create a signature for detecting similar responses.
        """
        # Take first part of response for comparison
        normalized = response.strip().lower()[:200]
        # Remove common variable parts like paths, numbers
        normalized = re.sub(r'/[\w/.-]+', '<PATH>', normalized)
        normalized = re.sub(r'\d+', '<NUM>', normalized)
        return normalized

    def record_response(self, response: str) -> None:
        """Record a response for text loop detection."""
        normalized = self._normalize_response(response)
        self._response_history.append(normalized)
        if len(self._response_history) > self.MAX_RESPONSE_HISTORY:
            self._response_history.pop(0)

    def detect_text_loop(self, response: str) -> tuple[bool, str]:
        """Detect if the agent is repeating the same response.

        Returns (is_loop, description).
        """
        if len(self._response_history) < 1:
            return False, ""

        normalized = self._normalize_response(response)

        # Check if this response matches recent ones (exact match)
        exact_matches = sum(1 for r in self._response_history if r == normalized)
        if exact_matches >= 1:
            return True, f"Agent repeated the same response {exact_matches + 1} times"

        # Check for common repetitive phrases that indicate looping
        repetitive_phrases = [
            "apologies for any confusion",
            "let me proceed",
            "i will now use the",
            "let's proceed with creating",
            "i'll create the",
        ]
        response_lower = response.lower()
        for phrase in repetitive_phrases:
            if phrase in response_lower:
                # Check if this phrase appeared in recent responses
                phrase_count = sum(1 for r in self._response_history if phrase in r)
                if phrase_count >= 1:
                    return True, f"Agent is stuck repeating '{phrase}'"

        # Check for high similarity (not exact match)
        current_words = set(normalized.split())
        similarity_matches = 0
        for prev in self._response_history[-3:]:
            prev_words = set(prev.split())
            if len(current_words) > 5 and len(prev_words) > 5:
                overlap = len(current_words & prev_words)
                similarity = overlap / max(len(current_words), len(prev_words))
                if similarity > 0.7:  # Lower threshold
                    similarity_matches += 1

        if similarity_matches >= 1:
            return True, "Agent responses are highly repetitive"

        return False, ""


@dataclass
class ValidationResult:
    """Result of pre-action validation."""
    valid: bool
    reason: str = ""
    suggestion: str = ""
    severity: str = "warning"  # 'warning', 'error', 'block'


class PreActionValidator:
    """Validates tool calls before execution to catch problematic actions.

    Catches:
    - Empty/missing required arguments
    - Invalid file paths
    - Dangerous bash commands
    - Writing empty content
    - Nonsensical operations
    """

    # Dangerous bash patterns that should be blocked
    DANGEROUS_PATTERNS = [
        (r'rm\s+(-[rf]+\s+)?/', "Dangerous: removing from root directory"),
        (r'rm\s+-rf\s+~', "Dangerous: removing home directory"),
        (r'>\s*/dev/sd[a-z]', "Dangerous: writing directly to disk device"),
        (r'mkfs\.', "Dangerous: formatting filesystem"),
        (r'dd\s+.*of=/dev/', "Dangerous: dd to device"),
        (r'chmod\s+-R\s+777\s+/', "Dangerous: making everything world-writable"),
        (r':\(\)\s*\{\s*:\|:\s*&\s*\}\s*;', "Dangerous: fork bomb"),
    ]

    # Suspicious patterns that warrant a warning
    SUSPICIOUS_PATTERNS = [
        (r'rm\s+-rf\s+', "Warning: recursive force delete"),
        (r'>\s*/etc/', "Warning: overwriting system config"),
        (r'curl\s+.*\|\s*sh', "Warning: piping curl to shell"),
        (r'wget\s+.*\|\s*sh', "Warning: piping wget to shell"),
        (r'eval\s+', "Warning: using eval"),
        (r'sudo\s+', "Warning: using sudo"),
    ]

    def validate(self, tool_name: str, arguments: dict) -> ValidationResult:
        """Validate a tool call before execution.

        Returns ValidationResult indicating if the action is valid.
        """
        if tool_name == "bash":
            return self._validate_bash(arguments)
        elif tool_name == "write":
            return self._validate_write(arguments)
        elif tool_name == "edit":
            return self._validate_edit(arguments)
        elif tool_name == "read":
            return self._validate_read(arguments)
        elif tool_name in ("glob", "grep"):
            return self._validate_search(tool_name, arguments)

        return ValidationResult(valid=True)

    def _validate_bash(self, arguments: dict) -> ValidationResult:
        """Validate bash command."""
        command = arguments.get("command", "")

        if not command or not command.strip():
            return ValidationResult(
                valid=False,
                reason="Empty command",
                suggestion="Provide a valid command to execute",
                severity="error",
            )

        # Check for dangerous patterns
        for pattern, reason in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command):
                return ValidationResult(
                    valid=False,
                    reason=reason,
                    suggestion="This command is too dangerous to execute",
                    severity="block",
                )

        # Check for suspicious patterns (allow but warn)
        for pattern, reason in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, command):
                return ValidationResult(
                    valid=True,  # Allow but flag
                    reason=reason,
                    severity="warning",
                )

        # Check for commands that won't work in non-interactive mode
        interactive_patterns = [
            (r'\bnano\b', "nano requires interactive terminal"),
            (r'\bvim?\b', "vim requires interactive terminal"),
            (r'\bemacs\b', "emacs requires interactive terminal"),
            (r'\bless\b', "less requires interactive terminal"),
            (r'\bmore\b', "more requires interactive terminal"),
            (r'\btop\b', "top requires interactive terminal"),
            (r'\bhtop\b', "htop requires interactive terminal"),
        ]
        for pattern, reason in interactive_patterns:
            if re.search(pattern, command):
                return ValidationResult(
                    valid=False,
                    reason=reason,
                    suggestion="Use non-interactive alternatives (cat, head, tail for viewing; sed for editing)",
                    severity="error",
                )

        return ValidationResult(valid=True)

    def _validate_write(self, arguments: dict) -> ValidationResult:
        """Validate write operation."""
        file_path = arguments.get("file_path", "")
        content = arguments.get("content", "")

        if not file_path or not file_path.strip():
            return ValidationResult(
                valid=False,
                reason="Empty file path",
                suggestion="Provide a valid file path",
                severity="error",
            )

        # Check for path issues
        path_result = self._validate_path(file_path)
        if not path_result.valid:
            return path_result

        # Warn about empty content (might be intentional)
        if content is None or (isinstance(content, str) and not content.strip()):
            return ValidationResult(
                valid=True,  # Allow but warn
                reason="Writing empty content to file",
                severity="warning",
            )

        # Check for writing to sensitive locations
        sensitive_paths = ['/etc/', '/usr/', '/bin/', '/sbin/', '/boot/', '/sys/', '/proc/']
        for sensitive in sensitive_paths:
            if file_path.startswith(sensitive):
                return ValidationResult(
                    valid=False,
                    reason=f"Cannot write to system directory: {sensitive}",
                    suggestion="Write to a user directory instead",
                    severity="block",
                )

        return ValidationResult(valid=True)

    def _validate_edit(self, arguments: dict) -> ValidationResult:
        """Validate edit operation."""
        file_path = arguments.get("file_path", "")
        old_string = arguments.get("old_string", "")
        new_string = arguments.get("new_string", "")

        if not file_path or not file_path.strip():
            return ValidationResult(
                valid=False,
                reason="Empty file path",
                suggestion="Provide a valid file path",
                severity="error",
            )

        # Check for path issues
        path_result = self._validate_path(file_path)
        if not path_result.valid:
            return path_result

        # old_string can be empty (for prepending), but warn
        if old_string is None:
            return ValidationResult(
                valid=False,
                reason="old_string is None",
                suggestion="Provide the text to replace (can be empty string for prepend)",
                severity="error",
            )

        # new_string can legitimately be empty (for deletion)
        if new_string is None:
            return ValidationResult(
                valid=False,
                reason="new_string is None",
                suggestion="Provide the replacement text (can be empty string for deletion)",
                severity="error",
            )

        # Check if old and new are identical
        if old_string == new_string:
            return ValidationResult(
                valid=False,
                reason="old_string and new_string are identical - no change would occur",
                suggestion="Provide different old and new strings",
                severity="error",
            )

        return ValidationResult(valid=True)

    def _validate_read(self, arguments: dict) -> ValidationResult:
        """Validate read operation."""
        file_path = arguments.get("file_path", "")

        if not file_path or not file_path.strip():
            return ValidationResult(
                valid=False,
                reason="Empty file path",
                suggestion="Provide a valid file path",
                severity="error",
            )

        return self._validate_path(file_path)

    def _validate_search(self, tool_name: str, arguments: dict) -> ValidationResult:
        """Validate glob/grep operations."""
        pattern = arguments.get("pattern", "")

        if not pattern or not pattern.strip():
            return ValidationResult(
                valid=False,
                reason=f"Empty {tool_name} pattern",
                suggestion="Provide a valid search pattern",
                severity="error",
            )

        return ValidationResult(valid=True)

    def _validate_path(self, file_path: str) -> ValidationResult:
        """Validate a file path for common issues."""
        # Check for null bytes (security issue)
        if '\x00' in file_path:
            return ValidationResult(
                valid=False,
                reason="Path contains null byte",
                suggestion="Remove null bytes from path",
                severity="block",
            )

        # Check for path traversal attempts outside reasonable bounds
        # (Some traversal is fine for relative paths)
        if '/../../../' in file_path or file_path.count('..') > 5:
            return ValidationResult(
                valid=False,
                reason="Excessive path traversal",
                suggestion="Use a direct path instead",
                severity="warning",
            )

        return ValidationResult(valid=True)


class RuntimeSafeguards:
    """Combined runtime safeguards for the agent.

    Usage:
        safeguards = RuntimeSafeguards()

        # For streaming:
        filtered = safeguards.filter_stream_chunk(chunk)
        if safeguards.should_steer():
            steering_msg = safeguards.get_steering_message()

        # Before tool execution:
        is_dup, reason = safeguards.check_duplicate(tool_name, args)
        if is_dup:
            skip this tool call

        # Pre-action validation:
        validation = safeguards.validate_action(tool_name, args)
        if not validation.valid:
            skip or warn

        # After tool execution:
        safeguards.record_action(tool_name, args)
    """

    def __init__(self):
        self.code_filter = CodeBlockFilter()
        self.pattern_detector = PatternDetector()
        self.action_tracker = ActionTracker()
        self.validator = PreActionValidator()
        self._pending_steering: str | None = None
        self._accumulated_content = ""

    def reset(self):
        """Reset all safeguards for a new conversation."""
        self.code_filter.reset()
        self.pattern_detector.reset()
        self.action_tracker.reset()
        self._pending_steering = None
        self._accumulated_content = ""

    def filter_stream_chunk(self, chunk: str) -> str:
        """Filter a streaming chunk, removing code blocks.

        Also detects patterns for potential steering.
        """
        # Filter code blocks
        result = self.code_filter.filter_chunk(chunk)

        # Accumulate for pattern detection
        self._accumulated_content += chunk

        # Check for patterns periodically (every 200 chars)
        if len(self._accumulated_content) > 200:
            matches = self.pattern_detector.detect(self._accumulated_content)
            if matches:
                steering = self.pattern_detector.get_steering_message(matches)
                if steering:
                    self._pending_steering = steering
            self._accumulated_content = self._accumulated_content[-100:]  # Keep last 100 chars for context

        return result.content

    def filter_complete_content(self, content: str) -> str:
        """Filter complete content (non-streaming)."""
        result = self.code_filter.filter_complete(content)

        # Also detect patterns
        matches = self.pattern_detector.detect(content)
        if matches:
            steering = self.pattern_detector.get_steering_message(matches)
            if steering:
                self._pending_steering = steering

        return result.content

    def should_steer(self) -> bool:
        """Check if we should inject a steering message."""
        return self._pending_steering is not None

    def get_steering_message(self) -> str | None:
        """Get pending steering message and clear it."""
        msg = self._pending_steering
        self._pending_steering = None
        return msg

    def check_duplicate(self, tool_name: str, arguments: dict) -> tuple[bool, str]:
        """Check if a tool call would be a duplicate."""
        return self.action_tracker.check_tool_call(tool_name, arguments)

    def record_action(self, tool_name: str, arguments: dict) -> None:
        """Record a completed tool action."""
        self.action_tracker.record_tool_call(tool_name, arguments)

    def detect_loop(self) -> tuple[bool, str]:
        """Detect if the agent is in a repetitive loop.

        Returns (is_loop, pattern_description).
        """
        return self.action_tracker.detect_loop()

    def validate_action(self, tool_name: str, arguments: dict) -> ValidationResult:
        """Validate a tool action before execution.

        Returns ValidationResult with validity and any warnings/errors.
        """
        return self.validator.validate(tool_name, arguments)

    def record_response(self, response: str) -> None:
        """Record a response for text loop detection."""
        self.action_tracker.record_response(response)

    def detect_text_loop(self, response: str) -> tuple[bool, str]:
        """Detect if the agent is repeating the same response.

        Returns (is_loop, description).
        """
        return self.action_tracker.detect_text_loop(response)
